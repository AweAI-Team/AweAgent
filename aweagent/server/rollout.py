"""Rollout server — programmatic entry point for collecting teacher trajectories.

``rollout()`` is the training-side counterpart of ``server.suite.evaluate()``,
but its shape is deliberately different. Eval keeps a *fixed* harness (hidden in
``BENCHES``) and only varies the checkpoint under test, so it takes an endpoint
argument. Rollout is the opposite: the harness *is* the thing being varied —
teacher endpoint, system prompt, skills, sampling, how many samples per instance
— and that whole experiment definition is a config file. So rollout takes a
**config path**, not a pile of keyword knobs. One yaml = "this data × this
setup"; the same dataset can have many configs, and one config can point at
different data. Turning trajectories into SFT data is a downstream concern —
this layer only guarantees the trajectories exist, are grouped per rollout, and
carry a verifier verdict when eval is on.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from aweagent.core.config.loader import load_config
from aweagent.core.task.pipeline import run_pipeline
from aweagent.core.task.types import TaskResult
from aweagent.server.normalize import _INFRA_KINDS, _error_kind

logger = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    """One rollout run's output: where the trajectories are and which are usable.

    Deliberately not a ``BenchScore``: rollout reports trajectory provenance and
    verdict tallies, not pass@k. Each rollout row is an independent SFT sample —
    no group-by-instance averaging, no dropping of all-infra instances.
    """

    run_dir: str
    # Per rollout k (0-based, matching runner._rollout_paths): the files it wrote.
    # N==1 → files live directly under run_dir; N>1 → under run_dir/rollout_k/.
    rollout_paths: list[dict] = field(default_factory=list)
    num_rollouts: int = 1
    n_instances: int = 0          # distinct instance_ids attempted
    n_trajectories: int = 0       # total result rows (incl. infra rows; ≤ n_instances × N)
    n_accepted: int = 0           # rows whose verifier accepted (meaningless if eval_skipped)
    n_infra: int = 0              # INFRA_ERROR / TIMEOUT / CONTEXT_LENGTH — unusable
    error_kind_counts: dict[str, int] = field(default_factory=dict)
    eval_skipped: bool = False


def _enumerate_trajectories(run_dir: Path, num_rollouts: int) -> list[dict]:
    """List each rollout's output files by reconstructing runner's layout.

    The runner does not return a file manifest, so we mirror ``_rollout_paths``
    (runner.py): N==1 writes directly under run_dir, N>1 under rollout_k/.
    """
    out: list[dict] = []
    for k in range(num_rollouts):
        base = run_dir if num_rollouts == 1 else run_dir / f"rollout_{k}"
        traj = base / "trajectories.jsonl"
        res = base / "results.jsonl"
        if traj.exists() or res.exists():  # tolerate a rollout that emitted nothing
            out.append(
                {
                    "rollout": k,
                    "trajectories": str(traj) if traj.exists() else None,
                    "results": str(res) if res.exists() else None,
                }
            )
    return out


def _summarize(results: list[TaskResult]) -> tuple[int, int, dict[str, int]]:
    """Flat per-trajectory tally: (n_accepted, n_infra, error_kind_counts).

    Reuses ``normalize._error_kind`` so OK / TASK_FAILURE / INFRA classification
    matches eval exactly, but does not call ``aggregate`` (pass@k semantics don't
    apply — each rollout is an independent sample).
    """
    counts: Counter = Counter()
    n_acc = 0
    n_infra = 0
    for r in results:
        ek = _error_kind(r)
        counts[ek] += 1
        if r.success:
            n_acc += 1
        if ek in _INFRA_KINDS:
            n_infra += 1
    return n_acc, n_infra, dict(counts)


async def rollout(
    config_path: str | Path,
    *,
    data_file: str | None = None,
    instance_ids: list[str] | None = None,
    output_root: str | Path | None = None,
    overrides: dict | None = None,
) -> RolloutResult:
    """Collect teacher trajectories for one rollout config.

    The config file *is* the experiment: it names the teacher endpoint
    (``llm.*``), the scaffold + system prompt / skills under test
    (``agent.*``), the dataset (``task.*``), how many samples per instance
    (``execution.num_rollouts``), and whether to run the verifier
    (``eval.enabled``). To sweep prompts/teachers, write several configs for the
    same dataset; to reuse one setup on different data, override ``data_file``.

    Args:
        config_path: path to the rollout config yaml. Must exist (unlike
            ``load_config``, a missing path is an error, not a silent default).
        data_file: override ``task.data_file`` for this run (training configs
            typically use ``data_file: ${DATA_FILE}``); None keeps the config's.
        instance_ids: optional instance-id subset filter.
        output_root: override ``execution.output_path``; None keeps the config's.
        overrides: escape hatch — an extra override dict deep-merged last (after
            data_file / output_root), for one-off knobs without a new config file.

    Returns:
        RolloutResult: run_dir, per-rollout trajectory file paths, and usability
        tallies (n_accepted is meaningful only when the verifier ran, i.e.
        ``eval_skipped`` is False).

    Notes:
        * Sampling temperature comes from the config's ``llm.params`` — set it
          > 0 there for ``num_rollouts > 1`` (greedy sampling yields identical
          trajectories). rollout does not silently inject a temperature.
        * ``save_trajectories`` is forced True regardless of the config —
          trajectories are rollout's only output.
        * The run directory is second-resolution timestamped; two rollout()
          calls in the same second with the same output_path + model name reuse
          the directory and interleave. Give each shard a distinct output_root
          (or model name) when sharding a pool into rapid calls.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"rollout config not found: {config_path}")

    # Assemble overrides: data_file / output_root / save_trajectories, then the
    # caller's escape-hatch overrides on top.
    ov: dict = {"execution": {"save_trajectories": True}}
    if output_root is not None:
        ov["execution"]["output_path"] = str(output_root)
    if data_file is not None:
        ov["task"] = {"data_file": data_file}
    if overrides:
        from aweagent.core.config.loader import _deep_merge

        ov = _deep_merge(ov, overrides)

    config = load_config(config_path, overrides=ov)

    skip_eval = not config.eval.enabled
    num_rollouts = config.execution.num_rollouts

    logger.info(
        "Rollout config=%s (num_rollouts=%d, max_concurrent=%d, eval=%s)",
        config_path, num_rollouts, config.execution.max_concurrent,
        config.eval.enabled,
    )
    runner, results = await run_pipeline(
        config, instance_ids=instance_ids, save_trajectories=True
    )

    n_acc, n_infra, ek_counts = _summarize(results)
    return RolloutResult(
        run_dir=str(runner.run_dir),
        rollout_paths=_enumerate_trajectories(Path(runner.run_dir), num_rollouts),
        num_rollouts=num_rollouts,
        n_instances=len({r.instance_id for r in results}),
        n_trajectories=len(results),
        n_accepted=n_acc,
        n_infra=n_infra,
        error_kind_counts=ek_counts,
        eval_skipped=skip_eval,
    )
