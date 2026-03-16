"""Terminal Bench V2 Evaluator — same-session verification.

Runs bash /tests/test.sh in the container, reads reward.txt or reward.json.
The evaluator runs in the SAME container the agent used.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time

from awe_agent.core.runtime.protocol import Runtime, RuntimeSession
from awe_agent.core.task.protocol import Evaluator
from awe_agent.core.task.types import EvalResult, Instance
from awe_agent.tasks.terminal_bench_v2.task import (
    INTERNAL_PYPI_INDEX,
    PROXY_ENV,
    TerminalBenchInstance,
)

logger = logging.getLogger(__name__)


class TerminalBenchV2Evaluator(Evaluator):
    """Evaluator for Terminal-Bench 2.0.

    requires_same_session: When True, the runner must evaluate in the same
    container as the agent (no patch to apply — agent modifies state directly).

    Flow:
    1. mkdir -p /logs/agent /logs/verifier
    2. Upload test files to /tests/
    3. Run bash /tests/test.sh (with cwd=workdir, env)
    4. Read reward.txt or reward.json
    5. accepted = reward > 0
    """

    @property
    def requires_same_session(self) -> bool:  # noqa: D102
        return True

    def __init__(self, timeout: int | None = None, **kwargs: object) -> None:
        super().__init__()
        self._timeout = timeout

    async def evaluate(
        self,
        instance: Instance,
        patch: str,
        runtime: Runtime,
    ) -> EvalResult:
        start = time.monotonic()

        try:
            async with runtime.session(instance.image) as session:
                return await self._run_verifier(instance, session, start)
        except Exception as exc:
            logger.error("Evaluation failed for %s: %s", instance.id, exc)
            return EvalResult(
                accepted=False,
                score=0.0,
                details={"error": str(exc)},
                duration=time.monotonic() - start,
            )

    async def _run_verifier(
        self,
        instance: Instance,
        session: RuntimeSession,
        start: float,
    ) -> EvalResult:
        inst = TerminalBenchInstance.from_instance(instance)
        workdir = inst.workdir
        test_files = inst.test_files
        verifier_timeout = int(inst.verifier_timeout_sec)

        # 1. Create log dirs
        await session.execute("mkdir -p /logs/agent /logs/verifier", timeout=30)

        # 2. Upload test files
        for remote_path, b64_content in test_files.items():
            content = base64.b64decode(b64_content)
            parent = os.path.dirname(remote_path)
            if parent and parent != "/":
                await session.execute(f"mkdir -p {parent}", timeout=30)
            await session.upload_file(remote_path, content)

        # 3. Build verifier env
        env = {
            "TEST_DIR": "/tests",
            "PIP_INDEX_URL": INTERNAL_PYPI_INDEX,
            **PROXY_ENV,
        }

        # 4. Run bash /tests/test.sh
        post_test_pane = ""
        reward_value = 0.0
        reward_source = "none"
        verifier_timed_out = False

        try:
            result = await session.execute(
                "bash /tests/test.sh 2>&1 | tee /logs/verifier/test-stdout.txt",
                cwd=workdir,
                timeout=verifier_timeout,
                env=env,
            )
            post_test_pane = result.stdout or ""
            logger.info(
                "Verifier finished for %s, exit_code=%d",
                instance.id, result.exit_code,
            )
        except TimeoutError:
            verifier_timed_out = True
            logger.warning(
                "Verifier timeout (%ds) for %s", verifier_timeout, instance.id
            )
            reward_source = "timeout"
        except Exception as e:
            logger.warning("Verifier failed for %s: %s", instance.id, e)
            post_test_pane = str(e)

        # 5. Read reward
        if not verifier_timed_out:
            # Try reward.txt
            try:
                txt_result = await session.execute(
                    "cat /logs/verifier/reward.txt 2>/dev/null", timeout=10,
                )
                content = (txt_result.stdout or "").strip()
                if content:
                    reward_value = float(content)
                    reward_source = "reward.txt"
                    logger.info(
                        "Read reward.txt for %s: %s", instance.id, reward_value
                    )
            except Exception:
                pass

            # Fallback to reward.json
            if reward_source == "none":
                try:
                    json_result = await session.execute(
                        "cat /logs/verifier/reward.json 2>/dev/null", timeout=10,
                    )
                    content = (json_result.stdout or "").strip()
                    if content:
                        data = json.loads(content)
                        reward_value = float(data.get("reward", 0))
                        reward_source = "reward.json"
                        logger.info(
                            "Read reward.json for %s: %s", instance.id, reward_value
                        )
                except Exception:
                    pass

            if reward_source == "none":
                logger.warning(
                    "No reward file found for %s, using 0.0", instance.id
                )

        accepted = reward_value > 0
        duration = time.monotonic() - start

        return EvalResult(
            accepted=accepted,
            score=reward_value,
            details={
                "reward_source": reward_source,
                "verifier_timed_out": verifier_timed_out,
                "output_tail": post_test_pane if post_test_pane else "",
            },
            duration=duration,
        )
