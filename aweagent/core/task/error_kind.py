"""Centralized error-kind inference.

The eval layer scatters failure signals across non-uniform ``EvalResult.details``
keys (PatchTest ``error``, SWE-bench-Pro ``entryscript_failed`` /
``missing_output_json``, Terminal-Bench-v2 ``verifier_timed_out``, eval-proxy
``exception`` / ``raw_status``) plus the agent's ``finish_reason``. Rather than
teach every evaluator to fill ``error_kind``, we classify once, here, reading the
signals that already exist. The raw ``details`` / ``finish_reason`` remain
persisted, so classification is always re-derivable offline.

Policy: only tag infra when a clear infra signal is present; otherwise a run that
completed but did not pass is a genuine ``task_failure``. Erring toward
``task_failure`` keeps pass-rate denominators honest (over-excluding would inflate
pass rate).
"""

from __future__ import annotations

from aweagent.core.task.types import ErrorKind, EvalResult


def infer_error_kind(
    *,
    finish_reason: str | None,
    eval_result: EvalResult | None,
    task_error: str | None,
) -> str:
    """Classify an instance outcome into an :class:`ErrorKind` value.

    Args:
        finish_reason: ``AgentResult.finish_reason`` (finish/max_steps/
            context_length/error/timeout), or None if the agent never ran.
        eval_result: the evaluator's result, or None if evaluation never
            happened (e.g. retries exhausted, runner-level exception).
        task_error: ``TaskResult.error`` — a runner-level infra failure string.
    """
    # 1. Hard infra failure — no evaluation happened at all.
    if task_error and eval_result is None:
        return ErrorKind.INFRA_ERROR.value

    # 2. Agent-lifecycle signals from finish_reason.
    if finish_reason == "timeout":
        return ErrorKind.TIMEOUT.value
    if finish_reason == "context_length":
        return ErrorKind.CONTEXT_LENGTH.value
    if finish_reason == "error":
        return ErrorKind.INFRA_ERROR.value

    # 3. Evaluator-reported infra signals in details (union across evaluators).
    details = (eval_result.details if eval_result else {}) or {}
    if details.get("raw_status") == 2 or "exception" in details:
        # eval-proxy internal exception (status==2) — infra.
        return ErrorKind.INFRA_ERROR.value
    if details.get("entryscript_failed") or details.get("missing_output_json"):
        # SWE-bench-Pro: the eval harness itself did not run.
        return ErrorKind.INFRA_ERROR.value
    if details.get("verifier_timed_out"):
        # Terminal-Bench-v2: the verifier died rather than judged.
        return ErrorKind.INFRA_ERROR.value
    if (
        details.get("error")
        and eval_result is not None
        and not eval_result.accepted
        and eval_result.score == 0.0
    ):
        # PatchTest patch_apply_failed / str(exc), eval-proxy status==1 app error.
        return ErrorKind.INFRA_ERROR.value

    # 4. Ran to completion.
    if eval_result is not None and eval_result.accepted:
        return ErrorKind.OK.value
    return ErrorKind.TASK_FAILURE.value
