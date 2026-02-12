"""SWEBenchEvaluator — evaluates SWE-Bench family datasets.

Evaluation logic:

1. Parse FAIL_TO_PASS and PASS_TO_PASS test IDs from instance metadata.
2. Run FAIL_TO_PASS tests — all must now pass (the bug is fixed).
3. Run PASS_TO_PASS tests — none should regress (no new breakage).
4. ``accepted = f2p_all_pass AND p2p_all_pass``.

Compatible with: ``swe_bench_verified``, ``ScaleSWE``, ``multi_swe_bench``,
and any dataset that stores test identifiers in the ``FAIL_TO_PASS`` /
``PASS_TO_PASS`` instance metadata fields.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from awe_agent.core.eval.base import PatchTestEvaluator
from awe_agent.core.eval.utils import run_f2p_p2p_eval
from awe_agent.core.task.types import EvalResult, Instance

if TYPE_CHECKING:
    from awe_agent.core.runtime.protocol import RuntimeSession

logger = logging.getLogger(__name__)


class SWEBenchEvaluator(PatchTestEvaluator):
    """Evaluator for SWE-Bench family datasets.

    The base class handles container creation, patch application, and
    test-file restoration.  This subclass only needs to run the F2P / P2P
    tests via the shared :func:`run_f2p_p2p_eval` helper.

    Example::

        evaluator = SWEBenchEvaluator(timeout=1800)
        result = await evaluator.evaluate(instance, patch, runtime)
        print(result.accepted, result.details)
    """

    async def run_tests(
        self,
        instance: Instance,
        session: RuntimeSession,
    ) -> EvalResult:
        """Run FAIL_TO_PASS and PASS_TO_PASS tests."""
        return await run_f2p_p2p_eval(
            session, instance, timeout=self._timeout,
        )
