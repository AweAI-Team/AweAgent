"""Official-style patch-only batch evaluator for SWE-bench-Pro."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from awe_agent.core.runtime.config import RuntimeConfig
from awe_agent.core.task.runner import runtime_registry
from awe_agent.core.task.types import Instance
from awe_agent.tasks.swe_bench_pro.eval_assets import parse_swebench_expected_tests
from awe_agent.tasks.swe_bench_pro.evaluator import SWEBenchProEvaluator
from awe_agent.tasks.swe_bench_pro.task import SWEBenchProTask

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SWEBenchProPatchSample:
    """A single patch submission from the official patch JSON."""

    instance_id: str
    patch: str
    prefix: str = ""


def load_patch_samples(path: str | Path) -> list[SWEBenchProPatchSample]:
    """Load the official patch JSON format."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of patch samples in {path}")

    samples: list[SWEBenchProPatchSample] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid patch sample in {path}: expected an object")
        instance_id = str(item.get("instance_id", "")).strip()
        if not instance_id:
            raise ValueError(f"Invalid patch sample in {path}: missing instance_id")
        patch = str(item.get("model_patch", item.get("patch", "")))
        prefix = str(item.get("prefix", ""))
        samples.append(SWEBenchProPatchSample(instance_id=instance_id, patch=patch, prefix=prefix))
    return samples


async def run_patch_evaluation(
    *,
    task: SWEBenchProTask,
    patch_samples: list[SWEBenchProPatchSample],
    runtime_config: RuntimeConfig,
    output_dir: str | Path,
    timeout: int = 3600,
    max_concurrent: int = 50,
    redo: bool = False,
) -> dict[str, bool]:
    """Evaluate a patch JSON against raw SWE-bench-Pro samples."""
    instances = {instance.id: instance for instance in task.get_instances()}
    valid_samples: list[SWEBenchProPatchSample] = []
    missing_instances: list[str] = []
    for sample in patch_samples:
        if sample.instance_id in instances:
            valid_samples.append(sample)
        else:
            missing_instances.append(sample.instance_id)

    if missing_instances:
        logger.warning(
            "Found %d patch instances not in raw samples; evaluating %d/%d patches",
            len(missing_instances),
            len(valid_samples),
            len(patch_samples),
        )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(max(1, max_concurrent))
    evaluator = SWEBenchProEvaluator(timeout=timeout)
    results: dict[str, bool] = {}

    async def _run_one(sample: SWEBenchProPatchSample) -> tuple[str, bool]:
        async with semaphore:
            accepted = await _evaluate_patch_sample(
                instance=instances[sample.instance_id],
                sample=sample,
                evaluator=evaluator,
                runtime_config=runtime_config,
                output_root=output_root,
                redo=redo,
            )
            return sample.instance_id, accepted

    tasks = [asyncio.create_task(_run_one(sample)) for sample in valid_samples]
    total = len(tasks)
    completed = 0
    for future in asyncio.as_completed(tasks):
        instance_id, accepted = await future
        results[instance_id] = accepted
        completed += 1
        accuracy = sum(results.values()) / len(results) if results else 0.0
        logger.info(
            "Progress: %d/%d evaluated (accuracy %.2f%%)",
            completed,
            total,
            accuracy * 100,
        )

    write_eval_results(output_root, results)
    return results


def write_eval_results(output_dir: str | Path, results: dict[str, bool]) -> Path:
    """Write the official eval_results.json aggregate."""
    output_path = Path(output_dir) / "eval_results.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    return output_path


async def _evaluate_patch_sample(
    *,
    instance: Instance,
    sample: SWEBenchProPatchSample,
    evaluator: SWEBenchProEvaluator,
    runtime_config: RuntimeConfig,
    output_root: Path,
    redo: bool,
) -> bool:
    instance_dir = output_root / sample.instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)

    output_path = instance_dir / _artifact_name(sample.prefix, "output.json")
    if output_path.exists() and not redo:
        return _accepted_from_output_json(instance, output_path.read_text(encoding="utf-8"))

    patch_path = instance_dir / _artifact_name(sample.prefix, "patch.diff")
    entryscript_path = instance_dir / _artifact_name(sample.prefix, "entryscript.sh")
    stdout_path = instance_dir / _artifact_name(sample.prefix, "stdout.log")
    stderr_path = instance_dir / _artifact_name(sample.prefix, "stderr.log")

    patch_path.write_text(sample.patch, encoding="utf-8")
    entryscript_path.write_text(str(instance.metadata.get("entryscript_sh", "")), encoding="utf-8")

    instance_runtime_config = runtime_config.model_copy(
        update={"image": instance.image, "workdir": instance.workdir}
    )
    runtime_cls = runtime_registry.get(instance_runtime_config.backend)
    runtime = runtime_cls(instance_runtime_config)

    result, artifacts = await evaluator.evaluate_with_artifacts(instance, sample.patch, runtime)
    entryscript_path.write_text(artifacts.entryscript_sh, encoding="utf-8")
    stdout_path.write_text(artifacts.stdout_log, encoding="utf-8")
    stderr_path.write_text(artifacts.stderr_log, encoding="utf-8")
    if artifacts.output_json:
        output_path.write_text(artifacts.output_json, encoding="utf-8")
    return result.accepted


def _accepted_from_output_json(instance: Instance, output_json_text: str) -> bool:
    try:
        parsed = json.loads(output_json_text)
    except json.JSONDecodeError:
        return False

    passed_tests = {
        test.get("name", "")
        for test in parsed.get("tests", [])
        if test.get("status") == "PASSED"
    }
    expected_tests = set(
        parse_swebench_expected_tests(instance.metadata.get("FAIL_TO_PASS"))
        + parse_swebench_expected_tests(instance.metadata.get("PASS_TO_PASS"))
    )
    return expected_tests <= passed_tests


def _artifact_name(prefix: str, filename: str) -> str:
    return f"{prefix}_{filename}"
