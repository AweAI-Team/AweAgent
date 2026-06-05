#!/usr/bin/env python3
"""Patch swebench==4.1.0 Docker evaluation containers to honor CPU specs.

The public Python constants define ``SPECS_PYLINT[version]["nano_cpus"]`` for
several Pylint versions, but the installed harness does not pass that value to
``docker.containers.create()``. Without an explicit CPU quota, Pylint's own
``_cpu_count()`` can infer one core from cgroup ``cpu.shares=1024`` and skip
tests marked ``needs_two_cores``.

This patch does not change predictions, images, test commands, or grading. It
only forwards the existing SWE-bench resource spec to Docker when starting the
evaluation container.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


MARKER = "# AWEAGENT_SWEBENCH_CONTAINER_RESOURCE_LIMITS"


def main() -> None:
    spec = importlib.util.find_spec("swebench.harness.docker_build")
    if spec is None or spec.origin is None:
        raise SystemExit("Could not locate swebench.harness.docker_build")

    path = Path(spec.origin)
    text = path.read_text()
    if MARKER in text:
        print(f"Already patched: {path}")
        return

    import_anchor = "    INSTANCE_IMAGE_BUILD_DIR,\n    UTF8,\n)"
    import_replacement = (
        "    INSTANCE_IMAGE_BUILD_DIR,\n"
        "    MAP_REPO_VERSION_TO_SPECS,\n"
        "    UTF8,\n"
        ")"
    )
    if "MAP_REPO_VERSION_TO_SPECS" not in text:
        if import_anchor not in text:
            raise SystemExit(f"Could not find constants import anchor in {path}")
        text = text.replace(import_anchor, import_replacement, 1)

    target = """\
        # Define arguments for running the container
        run_args = test_spec.docker_specs.get("run_args", {})
        cap_add = run_args.get("cap_add", [])

        container = client.containers.create(
            image=test_spec.instance_image_key,
            name=test_spec.get_instance_container_name(run_id),
            user=DOCKER_USER,
            detach=True,
            command="tail -f /dev/null",
            platform=test_spec.platform,
            cap_add=cap_add,
        )
"""
    replacement = f"""\
        # Define arguments for running the container
        run_args = test_spec.docker_specs.get("run_args", {{}})
        cap_add = run_args.get("cap_add", [])
        specs = MAP_REPO_VERSION_TO_SPECS.get(test_spec.repo, {{}}).get(
            test_spec.version, {{}}
        )
        nano_cpus = run_args.get(
            "nano_cpus",
            test_spec.docker_specs.get("nano_cpus", specs.get("nano_cpus")),
        )
        create_kwargs = {{}}
        if nano_cpus is not None:
            # {MARKER}
            create_kwargs["nano_cpus"] = int(nano_cpus)
            logger.info(
                f"Using nano_cpus={{int(nano_cpus)}} for {{test_spec.instance_id}}"
            )

        container = client.containers.create(
            image=test_spec.instance_image_key,
            name=test_spec.get_instance_container_name(run_id),
            user=DOCKER_USER,
            detach=True,
            command="tail -f /dev/null",
            platform=test_spec.platform,
            cap_add=cap_add,
            **create_kwargs,
        )
"""
    if target not in text:
        raise SystemExit(f"Could not find container create target in {path}")

    path.write_text(text.replace(target, replacement, 1))
    print(f"Patched: {path}")


if __name__ == "__main__":
    main()
