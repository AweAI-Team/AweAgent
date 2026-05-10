"""SWEBenchProEvaluator — isolated evaluation flow for SWE-bench-Pro."""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from awe_agent.core.runtime.protocol import Runtime
from awe_agent.core.task.protocol import Evaluator
from awe_agent.core.task.types import EvalResult, Instance
from awe_agent.tasks.swe_bench_pro.eval_assets import (
    parse_swebench_expected_tests,
    strip_binary_hunks,
)

if TYPE_CHECKING:
    from awe_agent.core.runtime.protocol import RuntimeSession

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SWEBenchProEvaluationArtifacts:
    """Workspace files and captured outputs for a single evaluation."""

    patch_diff: str
    entryscript_sh: str
    run_script_sh: str
    parser_py: str
    stdout_log: str = ""
    stderr_log: str = ""
    output_json: str = ""


class SWEBenchProEvaluator(Evaluator):
    """Evaluate a patch in a fresh container using swalm-style eval assets."""

    def __init__(self, timeout: int = 3600) -> None:
        self._timeout = timeout

    async def evaluate(
        self,
        instance: Instance,
        patch: str,
        runtime: Runtime,
    ) -> EvalResult:
        result, _ = await self.evaluate_with_artifacts(instance, patch, runtime)
        return result

    async def evaluate_with_artifacts(
        self,
        instance: Instance,
        patch: str,
        runtime: Runtime,
    ) -> tuple[EvalResult, SWEBenchProEvaluationArtifacts]:
        start = time.monotonic()
        artifacts = _build_artifacts(instance, patch)
        eval_error = instance.metadata.get("eval_assets_error")
        if eval_error:
            return (
                EvalResult(
                    accepted=False,
                    score=0.0,
                    details={"error": str(eval_error)},
                    duration=time.monotonic() - start,
                ),
                artifacts,
            )

        if not artifacts.entryscript_sh or not artifacts.run_script_sh or not artifacts.parser_py:
            return (
                EvalResult(
                    accepted=False,
                    score=0.0,
                    details={"error": "missing_eval_assets"},
                    duration=time.monotonic() - start,
                ),
                artifacts,
            )

        try:
            if getattr(getattr(runtime, "config", None), "backend", "") == "docker":
                return await self._evaluate_with_local_docker(
                    instance,
                    patch,
                    artifacts,
                    runtime,
                    start,
                )
            async with runtime.session(instance.image) as session:
                await session.execute(
                    "mkdir -p /workspace",
                    cwd=instance.workdir,
                    timeout=30,
                )
                await self._upload_eval_files(session, artifacts)
                exec_result = await session.execute(
                    "/bin/bash /workspace/entryscript.sh",
                    cwd=instance.workdir,
                    timeout=self._timeout,
                )

                stdout_text = await _safe_download_text(session, "/workspace/stdout.log")
                stderr_text = await _safe_download_text(session, "/workspace/stderr.log")
                output_json = await _safe_download_text(session, "/workspace/output.json")
                test_exit_code_text = await _safe_download_text(
                    session, "/workspace/test_exit_code.txt"
                )
                artifacts = replace(
                    artifacts,
                    stdout_log=stdout_text,
                    stderr_log=stderr_text,
                    output_json=output_json,
                )
                return (
                    _build_eval_result(
                        instance=instance,
                        patch=patch,
                        stdout_text=stdout_text,
                        stderr_text=stderr_text or exec_result.stderr,
                        output_json=output_json,
                        entryscript_exit_code=exec_result.exit_code,
                        test_exit_code=_parse_optional_int(test_exit_code_text),
                        start=start,
                    ),
                    artifacts,
                )
        except Exception as exc:
            logger.error("SWE-bench-Pro evaluation failed for %s: %s", instance.id, exc)
            return (
                EvalResult(
                    accepted=False,
                    score=0.0,
                    details={"error": str(exc)},
                    duration=time.monotonic() - start,
                ),
                artifacts,
            )

    async def _upload_eval_files(
        self,
        session: RuntimeSession,
        artifacts: SWEBenchProEvaluationArtifacts,
    ) -> None:
        files = {
            "/workspace/patch.diff": artifacts.patch_diff,
            "/workspace/entryscript.sh": artifacts.entryscript_sh,
            "/workspace/run_script.sh": artifacts.run_script_sh,
            "/workspace/parser.py": artifacts.parser_py,
        }
        for remote_path, content in files.items():
            await session.upload_file(remote_path, content.encode())

    async def _evaluate_with_local_docker(
        self,
        instance: Instance,
        patch: str,
        artifacts: SWEBenchProEvaluationArtifacts,
        runtime: Runtime,
        start: float,
    ) -> tuple[EvalResult, SWEBenchProEvaluationArtifacts]:
        import docker

        image = instance.image
        docker_cfg = runtime.config.docker
        workspace = Path(tempfile.mkdtemp(prefix="awe_swebench_pro_eval_"))

        files = {
            "patch.diff": artifacts.patch_diff,
            "entryscript.sh": artifacts.entryscript_sh,
            "run_script.sh": artifacts.run_script_sh,
            "parser.py": artifacts.parser_py,
        }
        for rel_path, content in files.items():
            (workspace / rel_path).write_text(content, encoding="utf-8")

        def _run_container() -> tuple[int | None, str, str, str]:
            client = docker.from_env()
            if docker_cfg.pull_policy == "always":
                client.images.pull(image)
            elif docker_cfg.pull_policy == "if_not_present":
                try:
                    client.images.get(image)
                except Exception:
                    client.images.pull(image)

            volumes = {str(workspace): {"bind": "/workspace", "mode": "rw"}}
            run_kwargs: dict[str, object] = {
                "volumes": volumes,
                "detach": True,
                "remove": True,
                "entrypoint": "/bin/bash",
                "command": ["-c", "bash /workspace/entryscript.sh"],
                "working_dir": runtime.config.workdir,
                "environment": docker_cfg.environment or None,
            }
            if docker_cfg.network == "none":
                run_kwargs["network_mode"] = "none"
            elif docker_cfg.network:
                run_kwargs["network"] = docker_cfg.network

            container = client.containers.run(image, **run_kwargs)
            result = container.wait(timeout=self._timeout)
            status_code = result.get("StatusCode") if isinstance(result, dict) else None
            stdout_text = _read_workspace_file(workspace / "stdout.log")
            stderr_text = _read_workspace_file(workspace / "stderr.log")
            output_json = _read_workspace_file(workspace / "output.json")
            return status_code, stdout_text, stderr_text, output_json

        try:
            loop = asyncio.get_running_loop()
            exit_code, stdout_text, stderr_text, output_json = await loop.run_in_executor(
                None,
                _run_container,
            )
        finally:
            for path in workspace.glob("*"):
                try:
                    path.unlink()
                except OSError:
                    pass
            try:
                workspace.rmdir()
            except OSError:
                pass

        artifacts = replace(
            artifacts,
            stdout_log=stdout_text,
            stderr_log=stderr_text,
            output_json=output_json,
        )
        return (
            _build_eval_result(
                instance=instance,
                patch=patch,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
                output_json=output_json,
                entryscript_exit_code=exit_code,
                test_exit_code=None,
                start=start,
            ),
            artifacts,
        )


def _build_artifacts(instance: Instance, patch: str) -> SWEBenchProEvaluationArtifacts:
    return SWEBenchProEvaluationArtifacts(
        patch_diff=strip_binary_hunks(patch),
        entryscript_sh=str(instance.metadata.get("entryscript_sh", "")),
        run_script_sh=str(instance.metadata.get("run_script_sh", "")),
        parser_py=str(instance.metadata.get("parser_py", "")),
    )


def _build_eval_result(
    *,
    instance: Instance,
    patch: str,
    stdout_text: str,
    stderr_text: str,
    output_json: str,
    entryscript_exit_code: int | None,
    test_exit_code: int | None,
    start: float,
) -> EvalResult:
    details: dict[str, Any] = {
        "patch_is_empty": not bool(patch.strip()),
        "patch_is_not_applicable": _patch_is_not_applicable(stderr_text),
        "entryscript_exit_code": entryscript_exit_code,
        "test_exit_code": test_exit_code,
        "stdout": stdout_text[-2000:],
        "stderr": stderr_text[-2000:],
    }

    if not output_json:
        details["error"] = (
            "entryscript_failed"
            if entryscript_exit_code not in (None, 0)
            else "missing_output_json"
        )
        return EvalResult(
            accepted=False,
            score=0.0,
            details=details,
            duration=time.monotonic() - start,
        )

    try:
        parsed = json.loads(output_json)
    except json.JSONDecodeError as exc:
        details["error"] = "invalid_output_json"
        details["output_json_error"] = str(exc)
        details["output_json"] = output_json[-2000:]
        return EvalResult(
            accepted=False,
            score=0.0,
            details=details,
            duration=time.monotonic() - start,
        )

    passed_tests = {
        test.get("name", "")
        for test in parsed.get("tests", [])
        if test.get("status") == "PASSED"
    }
    expected_tests = set(
        parse_swebench_expected_tests(instance.metadata.get("FAIL_TO_PASS"))
        + parse_swebench_expected_tests(instance.metadata.get("PASS_TO_PASS"))
    )
    accepted = expected_tests <= passed_tests
    details.update(
        {
            "expected_tests": sorted(expected_tests),
            "passed_tests": sorted(passed_tests),
            "missing_tests": sorted(expected_tests - passed_tests),
        }
    )
    return EvalResult(
        accepted=accepted,
        score=1.0 if accepted else 0.0,
        details=details,
        duration=time.monotonic() - start,
    )


async def _safe_download_text(session: RuntimeSession, remote_path: str) -> str:
    try:
        data = await session.download_file(remote_path)
    except FileNotFoundError:
        return ""
    return data.decode("utf-8", errors="replace")


def _patch_is_not_applicable(stderr_text: str) -> bool:
    lowered = stderr_text.lower()
    is_corrupt = "corrupt patch at line" in lowered or "error: corrupt patch" in lowered
    not_apply = "git apply" in lowered and (
        "does not apply" in lowered or "failed" in lowered or "error:" in lowered
    )
    return bool(is_corrupt or not_apply)


def _parse_optional_int(text: str) -> int | None:
    text = text.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _read_workspace_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
