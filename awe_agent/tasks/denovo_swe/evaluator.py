"""DeNovoSWEEvaluator — evaluates DeNovoSWE doc2repo tasks.

Evaluation workflow:
1. Apply agent's code patch (from PatchTestEvaluator base).
2. Delete any test files that the agent may have created (to avoid conflicts).
3. Apply the unit-test patch (``test_patch`` from instance metadata).
4. Run ``pip install -e .`` to install the agent's code.
5. Execute all unit tests (passed_ptp + failed_ptp) via pytest.
6. Score based on pass_rate.

Validate-run mode:
- Skips agent patch (code is already in the image since we don't clean it).
- Still deletes test files and applies test patch.
- Runs evaluation to verify the test infrastructure works.

Most non-orchestration helpers live in :mod:`._helpers` and the embedded
sandbox scripts in :mod:`._scripts` so this file stays focused on the
end-to-end flow.
"""

from __future__ import annotations

import logging
import os
import shlex
import time
from typing import TYPE_CHECKING, Any

from awe_agent.core.eval.base import PatchTestEvaluator
from awe_agent.core.eval.utils import (
    parse_pytest_summary,
    parse_test_ids,
    run_tests_with_runner,
    sanitize_test_ids,
)
from awe_agent.core.runtime.protocol import Runtime
from awe_agent.core.task.types import EvalResult, Instance
from awe_agent.tasks.denovo_swe._helpers import (
    aggregate_iterations,
    apply_binary_archive,
    clean_all_test_files,
    collect_test_ids,
    delete_docker_image,
    files_added_by_test_patch,
    group_tests_by_file,
    head_tail_slice,
    parse_patch_files,
    remove_failed_tests,
)
from awe_agent.tasks.denovo_swe._scripts import UNINSTALL_PKG_SCRIPT

if TYPE_CHECKING:
    from awe_agent.core.runtime.protocol import RuntimeSession

logger = logging.getLogger(__name__)

_EVAL_TIMEOUT = 1800  # 30 min


class DeNovoSWEEvaluator(PatchTestEvaluator):
    """Evaluator for the DeNovoSWE benchmark.

    Applies the unit-test patch after the agent's code, then runs all
    tests to compute pass_rate.

    In ``validate_run`` mode, the agent patch is skipped (source code
    remains in the image) but tests are still applied and executed.

    Multi-run averaging:
        Pass ``eval_iters > 1`` to run the full evaluate cycle that
        many times in *independent* sandboxes and aggregate the
        scores.  This mitigates pytest / install / network flakiness
        when comparing model variants — single runs are noisy enough
        that small fixes can disappear in the noise.
    """

    def __init__(
        self,
        timeout: int = 3600,
        validate_run: bool = False,
        del_done_images: bool = False,
        eval_iters: int = 1,
    ) -> None:
        super().__init__(timeout=timeout, restore_tests=False)
        self._validate_run = validate_run
        self._del_done_images = del_done_images
        if eval_iters < 1:
            raise ValueError(f"eval_iters must be >= 1, got {eval_iters}")
        self._eval_iters = eval_iters

    async def evaluate(
        self,
        instance: Instance,
        patch: str,
        runtime: Runtime,
    ) -> EvalResult:
        """Public entry point — fans out to multiple iterations when
        ``eval_iters > 1`` and aggregates the resulting scores."""
        if self._eval_iters == 1:
            return await self._evaluate_once(instance, patch, runtime)

        outer_start = time.monotonic()
        results: list[EvalResult] = []
        for i in range(self._eval_iters):
            logger.info(
                "Eval: starting iteration %d/%d for %s",
                i + 1, self._eval_iters, instance.id,
            )
            try:
                er = await self._evaluate_once(instance, patch, runtime)
            except Exception as exc:
                logger.error(
                    "Eval: iteration %d for %s crashed: %s",
                    i + 1, instance.id, exc,
                )
                er = EvalResult(
                    accepted=False,
                    score=0.0,
                    details={"error": f"iteration_crash:{exc!r}"},
                    duration=0.0,
                )
            results.append(er)
            logger.info(
                "Eval: iteration %d/%d for %s → score=%.3f accepted=%s",
                i + 1, self._eval_iters, instance.id,
                er.score or 0.0, er.accepted,
            )

        return aggregate_iterations(
            instance.id, results, time.monotonic() - outer_start,
        )

    async def _evaluate_once(
        self,
        instance: Instance,
        patch: str,
        runtime: Runtime,
    ) -> EvalResult:
        """Single evaluation pass — full lifecycle in one sandbox.

        In validate_run mode, we skip agent patch application but still
        apply the test patch and run evaluation.
        """
        start = time.monotonic()
        image = instance.image

        try:
            async with runtime.session(image) as session:
                workdir = instance.workdir

                # 1. Checkout base commit
                if instance.base_commit:
                    checkout_result = await session.execute(
                        f"git checkout -f {instance.base_commit}",
                        cwd=workdir,
                    )
                    if not checkout_result.success:
                        logger.error(
                            "git checkout failed for %s: %s",
                            instance.id, checkout_result.stderr[:500],
                        )
                        return EvalResult(
                            accepted=False,
                            score=0.0,
                            details={
                                "error": "checkout_failed",
                                "stderr": checkout_result.stderr[-2000:],
                            },
                            duration=time.monotonic() - start,
                        )

                # 1.5. Run clean.sh to match the state the agent saw
                if not self._validate_run:
                    # ``abspath`` collapses ``../..`` segments at lookup
                    # time.  Without it we've seen sporadic
                    # ``FileNotFoundError`` on busy NFS volumes when the
                    # cwd transiently doesn't expose the parent dirs the
                    # relative path traverses (e.g.
                    # ``firefighterblu3_python-pam_pr-1`` once failed
                    # with ``/.../recipes/denovo_swe/../../awe_agent/...
                    # clean.sh`` despite the file being present).
                    clean_sh = os.path.abspath(os.path.join(
                        os.path.dirname(__file__), "clean.sh",
                    ))
                    logger.info("Eval: running clean.sh for %s", instance.id)
                    with open(clean_sh, "rb") as f:
                        clean_script = f.read()
                    await session.upload_file(
                        f"{workdir}/clean.sh", clean_script,
                    )
                    clean_result = await session.execute(
                        f"bash {workdir}/clean.sh {workdir}",
                        cwd=workdir, timeout=300,
                    )
                    await session.execute(
                        f"rm -f {workdir}/clean.sh", cwd=workdir,
                    )
                    logger.info(
                        "Eval: clean.sh for %s exit=%d",
                        instance.id, clean_result.exit_code,
                    )
                    if not clean_result.success:
                        logger.error(
                            "clean.sh failed in eval for %s: %s",
                            instance.id, clean_result.stderr[:500],
                        )
                        return EvalResult(
                            accepted=False,
                            score=0.0,
                            details={
                                "error": "eval_clean_failed",
                                "stderr": clean_result.stderr[-2000:],
                            },
                            duration=time.monotonic() - start,
                        )

                    # 1.6. Re-inject README.md (the spec) — symmetric with
                    #      ``prepare_session`` step 2.  clean.sh just
                    #      deleted every ``*.md`` to scrub upstream docs,
                    #      which also removed the spec README the agent
                    #      had access to.  Many ``setup.py`` files
                    #      (``long_description = open("README.md").read()``)
                    #      crash ``pip install -e .`` when README is missing
                    #      — since the README was folded into the agent's
                    #      baseline commit (no longer in their patch),
                    #      this re-upload is required for symmetry.
                    document = instance.metadata.get("document", "") or ""
                    if document:
                        await session.upload_file(
                            f"{workdir}/README.md", document.encode(),
                        )

                # 2. In validate_run mode, skip agent patch; otherwise apply it
                if not self._validate_run:
                    if not patch or not patch.strip():
                        logger.error("Eval: empty patch for %s", instance.id)
                        return EvalResult(
                            accepted=False,
                            score=0.0,
                            details={"error": "empty_patch"},
                            duration=time.monotonic() - start,
                        )
                    logger.info(
                        "Eval: applying agent patch for %s (%d bytes)",
                        instance.id, len(patch),
                    )
                    apply_result = await session.apply_patch(workdir, patch)
                    logger.info(
                        "Eval: agent patch apply for %s exit=%d",
                        instance.id, apply_result.exit_code,
                    )
                    if not apply_result.success:
                        return EvalResult(
                            accepted=False,
                            score=0.0,
                            details={
                                "error": "patch_apply_failed",
                                "stderr": apply_result.stderr[-2000:],
                            },
                            duration=time.monotonic() - start,
                        )

                # 2.5. Verify source files exist after patch apply
                verify_result = await session.execute(
                    "find . -name '*.py' -not -path './.git/*' | head -10",
                    cwd=workdir,
                )
                logger.info(
                    "Eval: files after patch apply for %s: %s",
                    instance.id,
                    verify_result.stdout[:300] if verify_result.stdout else "NONE",
                )

                # 3. Nuke ALL test-related files in the repo.
                #    This removes:
                #    - test files the agent may have created (verify_*.py, tests/, etc.)
                #    - original test files that might conflict with test_patch
                #    After this, test_patch will create the correct tests from scratch.
                await clean_all_test_files(session, workdir)

                # 4. Parse test_patch to find files it will create, ensure dirs exist
                test_patch = instance.metadata.get("test_patch", "")
                if test_patch:
                    patch_files = parse_patch_files(test_patch)
                    for pf in patch_files:
                        parent_dir = "/".join(pf.rsplit("/", 1)[:-1]) if "/" in pf else ""
                        if parent_dir:
                            await session.execute(
                                f"mkdir -p {parent_dir}",
                                cwd=workdir,
                            )

                    # 4.5 Pre-clean files that test_patch will NEWLY add.  Without
                    # this, an agent-created file at the same path causes the
                    # corresponding "new file" hunk in test_patch to be rejected.
                    # We use the strict add-only helper, so files merely modified
                    # by test_patch (e.g. setup.py / .gitignore preserved by
                    # clean.sh) are NOT removed — only files where the diff is
                    # `--- /dev/null` -> `+++ b/<path>`.  rm -f is per-file and
                    # never recursive: a missing file is a no-op, a directory
                    # is never matched.  Score semantics are unchanged: any file
                    # we delete here is overwritten 1:1 by test_patch on the
                    # next step, so an agent gets the same credit as before for
                    # the surrounding source files.
                    files_added = files_added_by_test_patch(test_patch)
                    if files_added:
                        logger.info(
                            "Eval: pre-cleaning %d add-only paths from test_patch for %s",
                            len(files_added), instance.id,
                        )
                        for pf in files_added:
                            await session.execute(
                                f"rm -f -- {shlex.quote(pf)}",
                                cwd=workdir,
                            )

                # 5. Apply the unit-test patch (must fully succeed — no partial apply)
                if test_patch:
                    # Clear any ``.rej`` files left over from the
                    # agent's own ``apply_patch`` (step 2).  Without
                    # this, the post-apply ``find -name '*.rej'``
                    # check below would mis-attribute agent's
                    # rejected hunks to test_patch and short-circuit
                    # to ``test_patch_partial_apply``.  Confirmed
                    # bite: ``dedupeio_pylbfgs_pr19`` failed with
                    # score=0 because the agent's patch had
                    # ``setup.py.rej`` / ``MANIFEST.in.rej`` /
                    # ``.gitignore.rej`` / ``lbfgs/_lowlevel.c.rej``
                    # left over — none of which test_patch even
                    # touches (it only adds README.rst and
                    # tests/test_lbfgs.py).
                    await session.execute(
                        "find . -name '*.rej' -type f -delete "
                        "2>/dev/null || true",
                        cwd=workdir,
                    )
                    logger.info(
                        "Eval: applying test_patch for %s (%d bytes)",
                        instance.id, len(test_patch),
                    )
                    apply_result = await session.apply_patch(workdir, test_patch)
                    # Check for .rej files — indicates partial apply via --reject strategy.
                    # Only files created AFTER we cleared above are attributable to
                    # test_patch's apply, so this check is now precise.
                    rej_check = await session.execute(
                        "find . -name '*.rej' -type f 2>/dev/null | head -5",
                        cwd=workdir,
                    )
                    if rej_check.stdout.strip():
                        logger.error(
                            "test_patch partially failed for %s (rejected hunks): %s",
                            instance.id, rej_check.stdout.strip(),
                        )
                        return EvalResult(
                            accepted=False,
                            score=0.0,
                            details={
                                "error": "test_patch_partial_apply",
                                "rejected_files": rej_check.stdout.strip(),
                                "stderr": apply_result.stderr[-2000:],
                            },
                            duration=time.monotonic() - start,
                        )
                    logger.info(
                        "Eval: test_patch apply for %s exit=%d",
                        instance.id, apply_result.exit_code,
                    )
                    if not apply_result.success:
                        logger.error(
                            "test_patch failed for %s: %s",
                            instance.id,
                            apply_result.stderr[:500],
                        )
                        return EvalResult(
                            accepted=False,
                            score=0.0,
                            details={
                                "error": "test_patch_apply_failed",
                                "stderr": apply_result.stderr[-2000:],
                            },
                            duration=time.monotonic() - start,
                        )
                else:
                    # If we have only binary fixtures and no text patch
                    # (e.g. a test suite consisting purely of data files
                    # to be read), proceed without failing.  Otherwise
                    # this is a real error.
                    if not instance.metadata.get("test_binary_archive_b64"):
                        logger.warning("No test_patch for instance %s", instance.id)
                        return EvalResult(
                            accepted=False,
                            score=0.0,
                            details={"error": "no_test_patch"},
                            duration=time.monotonic() - start,
                        )

                # 5.4. Surface Bug B cheating-risk warning.  When
                #      extract_patch's sibling-helper whitelist captures
                #      a path that shares its top-level segment with the
                #      agent's expected package, the eval might overlay
                #      a reference implementation onto agent code.  We
                #      log it here once per run; the field also lands in
                #      ``details`` for downstream audit.
                bug_b_overrides = instance.metadata.get(
                    "bug_b_overrides_agent_code", [],
                ) or []
                if bug_b_overrides:
                    logger.warning(
                        "Eval: instance %s has %d Bug B paths that may "
                        "shadow agent code (see "
                        "bug_b_overrides_agent_code in details)",
                        instance.id, len(bug_b_overrides),
                    )

                # 5.5. Apply binary fixtures (Bug C).  Unified text diffs
                #      cannot carry binary content, so extract_patch.py
                #      packs binary test files into a base64-encoded
                #      tar.gz which we now upload + extract.
                binary_archive_b64 = instance.metadata.get(
                    "test_binary_archive_b64", "",
                ) or ""
                if binary_archive_b64:
                    apply_err = await apply_binary_archive(
                        session, workdir, binary_archive_b64,
                        instance.metadata.get("test_binary_files", []) or [],
                        instance.id,
                    )
                    if apply_err:
                        return EvalResult(
                            accepted=False,
                            score=0.0,
                            details={
                                "error": "binary_archive_apply_failed",
                                "reason": apply_err,
                            },
                            duration=time.monotonic() - start,
                        )

                # 6. Remove failed_ptp tests from files to prevent collection crashes
                failed_ptp = instance.metadata.get("failed_ptp", [])
                if isinstance(failed_ptp, str):
                    failed_ptp = parse_test_ids(failed_ptp)
                if failed_ptp:
                    await remove_failed_tests(
                        session, workdir, failed_ptp,
                    )

                # 7. Uninstall old package then reinstall in editable mode.
                #    The image may have had `pip install .` (non-editable),
                #    which caches code in site-packages. Without uninstall,
                #    `pip install -e .` might not override it, causing the
                #    agent's new code to be invisible at import time.
                await session.upload_file(
                    "/tmp/_awe_uninstall_pkg.py",
                    UNINSTALL_PKG_SCRIPT.encode(),
                )
                await session.execute(
                    "python /tmp/_awe_uninstall_pkg.py",
                    cwd=workdir, timeout=60,
                )
                logger.info("Eval: pip install -e . for %s", instance.id)
                install_result = await session.execute(
                    "pip install -e . 2>&1", cwd=workdir, timeout=300,
                )
                logger.info(
                    "Eval: pip install for %s exit=%d, stdout_tail=%s",
                    instance.id,
                    install_result.exit_code,
                    install_result.stdout[-200:] if install_result.stdout else "",
                )
                if not install_result.success:
                    # Hard fail: a package that won't install editably
                    # makes every subsequent test fail for the wrong
                    # reason.  Previously this was a warning, which
                    # produced misleading low scores attributed to the
                    # agent rather than to a broken install.
                    logger.error(
                        "pip install -e . failed for %s: %s",
                        instance.id,
                        install_result.stderr[-500:],
                    )
                    return EvalResult(
                        accepted=False,
                        score=0.0,
                        details={
                            "error": "pip_install_failed",
                            "exit_code": install_result.exit_code,
                            "stderr": install_result.stderr[-2000:],
                            "stdout_tail": install_result.stdout[-500:]
                            if install_result.stdout else "",
                        },
                        duration=time.monotonic() - start,
                    )

                # 8. Run tests
                eval_result = await self._run_eval_tests(instance, session)
                eval_result.duration = time.monotonic() - start
                return eval_result

        except Exception as exc:
            logger.error("Evaluation failed for %s: %s", instance.id, exc)
            return EvalResult(
                accepted=False,
                score=0.0,
                details={"error": str(exc)},
                duration=time.monotonic() - start,
            )
        finally:
            # Delete image after use if requested
            if self._del_done_images and image:
                await delete_docker_image(image)

    async def run_tests(
        self,
        instance: Instance,
        session: RuntimeSession,
    ) -> EvalResult:
        """Required by PatchTestEvaluator but we override evaluate() directly."""
        return await self._run_eval_tests(instance, session)

    async def _run_eval_tests(
        self,
        instance: Instance,
        session: RuntimeSession,
    ) -> EvalResult:
        """Run unit tests per-file and compute pass_rate.

        Only ``passed_ptp`` tests are executed and scored.
        ``failed_ptp`` tests are known-broken and excluded entirely.

        Tests are grouped by file and each file is run in a separate
        pytest invocation.  If one file crashes (import error, collection
        error, etc.), the other files still run normally.
        """
        workdir = instance.workdir

        # Only run passed_ptp; failed_ptp are known-broken, skip them
        passed_ptp = instance.metadata.get("passed_ptp", [])
        failed_ptp = instance.metadata.get("failed_ptp", [])

        if isinstance(passed_ptp, str):
            passed_ptp = parse_test_ids(passed_ptp)
        if isinstance(failed_ptp, str):
            failed_ptp = parse_test_ids(failed_ptp)

        # Safety-net Bug E sanitizer.  extract_patch.py already rewrites
        # ``<dir>/TestX.py::<rest>`` → ``<dir>.py::TestX::<rest>`` upstream,
        # but applying it here too means we are resilient to old datasets
        # being re-evaluated without re-extraction.
        sanitized, rewrites = await sanitize_test_ids(
            list(passed_ptp), session, workdir,
        )
        if rewrites:
            logger.info(
                "Instance %s: eval-side sanitizer rewrote %d test ids "
                "(Bug E safety net)",
                instance.id, len(rewrites),
            )
            passed_ptp = sanitized

        tests_to_run = list(passed_ptp)
        total_expected = len(tests_to_run)

        if not tests_to_run:
            logger.warning("Instance %s has no passed_ptp test IDs", instance.id)
            return EvalResult(
                accepted=False,
                score=0.0,
                details={"error": "no_test_ids"},
            )

        logger.info(
            "Instance %s: running %d passed_ptp tests (skipping %d failed_ptp)",
            instance.id, total_expected, len(failed_ptp),
        )

        timeout = min(self._timeout, _EVAL_TIMEOUT)

        # Group by file and run each file separately for isolation
        file_groups = group_tests_by_file(tests_to_run)
        total_passed = 0
        total_failed = 0
        total_errors = 0
        all_outputs: list[str] = []
        per_file_details: dict[str, dict] = {}

        for filepath, file_tests in file_groups.items():
            try:
                # ── Pre-flight collection ──────────────────────────
                # ``passed_ptp`` was recorded at extract time.  By
                # eval time, parametrize expansions may have shifted
                # (different Python version, different ``dir(<Class>)``,
                # hypothesis seeds, etc.) so the recorded ids no
                # longer match what pytest collects.
                #
                # Pytest aborts the **entire batch** the moment ONE
                # node id can't be resolved (``ERROR: not found:
                # tests/test_x.py::test_y ... no match in any of …``
                # → exit 4 with ``no tests ran``).  So we'd lose 95
                # passing tests because the 96th id had a slightly
                # different ``[param]`` label.  Real example:
                # ``infinidat_munch_pr-1`` collected 95 / expected 96
                # → score 0 instead of 95/96.
                #
                # Fix: run ``pytest --collect-only`` first, intersect
                # the discovered ids with ``file_tests``, run only the
                # intersection, and account the missing ids as errors
                # individually (instead of as one cliff-edge zero).
                collected_ids = await collect_test_ids(
                    session, workdir, filepath, timeout=120,
                )
                if collected_ids is None:
                    # Pre-flight collection itself failed (import
                    # error in the test file, syntax error, etc.).
                    # Fall through with the original id list — the
                    # ensuing ``run_tests_with_runner`` will surface
                    # the import error in its output, and every test
                    # in the file will count as an error which is
                    # exactly what we want.
                    runnable = list(file_tests)
                    missing: list[str] = []
                    collection_error = "collection_failed"
                else:
                    runnable = [t for t in file_tests if t in collected_ids]
                    missing = [t for t in file_tests if t not in collected_ids]
                    collection_error = None

                if runnable:
                    file_passed, raw_output, file_details = await run_tests_with_runner(
                        session, workdir, runnable, timeout=timeout,
                    )
                    summary = parse_pytest_summary(raw_output)
                else:
                    # Nothing pytest can run — usually means every
                    # ``passed_ptp`` id is stale; record per-file
                    # diagnostic so the downstream report can tell
                    # the difference between "collection skew" and
                    # "all tests genuinely failed".
                    file_passed = False
                    raw_output = (
                        f"[no runnable test IDs for {filepath}; "
                        f"expected {len(file_tests)}, collected "
                        f"{0 if collected_ids is None else len(collected_ids)}, "
                        f"missing {len(missing)}]"
                    )
                    summary = parse_pytest_summary("")
                    file_details = {}

                total_passed += summary.passed
                total_failed += summary.failed
                # Missing ids are accounted as errors here so the
                # final ``pass_rate = passed / total_expected``
                # denominator is still ``total_expected`` (no score
                # inflation).
                total_errors += summary.errors + len(missing)
                # Keep both head and tail — fatal SIGSEGV / collection
                # errors emit the diagnostic at the TOP of the output,
                # which a pure ``-1000:`` slice would drop.  Net cost is
                # at most ~2.5 KB per file group, vs. the alternative of
                # an "unknown crash" mystery in every report.
                file_output_slice = head_tail_slice(raw_output)
                all_outputs.append(file_output_slice)
                per_file_details[filepath] = {
                    "all_passed": file_passed and not missing,
                    "passed": summary.passed,
                    "failed": summary.failed,
                    "errors": summary.errors + len(missing),
                    "expected": len(file_tests),
                    # New diagnostic fields — only present when
                    # collection skew is at play, kept short.
                    "collected": (
                        0 if collected_ids is None else len(collected_ids)
                    ),
                    "missing_ids": len(missing),
                    "collection_error": collection_error,
                    # Per-file output — bounded by ``head_tail_slice``
                    # to head+tail (~3 KB). Stored here so multi-file
                    # runs don't lose per-file root cause when the
                    # combined ``details.output`` tail-truncates.
                    "output": file_output_slice,
                }
            except Exception as exc:
                # File-level crash — count all tests in this file as errors,
                # but keep going with other files
                logger.warning(
                    "Instance %s: file %s crashed: %s",
                    instance.id, filepath, exc,
                )
                total_errors += len(file_tests)
                crash_line = f"[CRASH] {filepath}: {exc}"
                all_outputs.append(crash_line)
                per_file_details[filepath] = {
                    "all_passed": False,
                    "passed": 0,
                    "failed": 0,
                    "errors": len(file_tests),
                    "expected": len(file_tests),
                    "crash": str(exc),
                    "output": crash_line,
                }

        # Sanity check: total_ran should equal total_expected.
        # If not, some tests were silently lost (not collected, skipped, etc.)
        total_ran = total_passed + total_failed + total_errors
        if total_ran != total_expected:
            logger.warning(
                "Instance %s: test count mismatch! expected=%d, "
                "ran=%d (passed=%d, failed=%d, errors=%d). "
                "Unaccounted tests: %d",
                instance.id, total_expected, total_ran,
                total_passed, total_failed, total_errors,
                total_expected - total_ran,
            )
            # Count unaccounted tests as errors to avoid score inflation
            unaccounted = total_expected - total_ran
            if unaccounted > 0:
                total_errors += unaccounted

        pass_rate = total_passed / total_expected if total_expected > 0 else 0.0
        all_passed = (
            total_passed == total_expected
            and total_failed == 0
            and total_errors == 0
        )

        combined_output = "\n\n".join(all_outputs)

        details: dict[str, Any] = {
            "total_expected": total_expected,
            "passed_ptp_count": len(passed_ptp),
            "failed_ptp_count_skipped": len(failed_ptp),
            "passed": total_passed,
            "failed": total_failed,
            "errors": total_errors,
            "pass_rate": pass_rate,
            "num_test_files": len(file_groups),
            "per_file": per_file_details,
            # Head+tail slice (was ``combined_output[-3000:]``).  Pure
            # tail-truncation hid collection / import errors which
            # always appear at the head of pytest output — see
            # ``head_tail_slice`` docstring.  Bumped budget keeps the
            # combined report useful even when several test files fail
            # for different reasons.
            "output": head_tail_slice(combined_output, head_bytes=8000, tail_bytes=4000),
            "validate_run": self._validate_run,
            # Forwarded from extract_patch — non-empty means the test
            # patch may shadow files the agent was expected to write.
            "bug_b_overrides_agent_code": instance.metadata.get(
                "bug_b_overrides_agent_code", [],
            ) or [],
        }

        return EvalResult(
            accepted=all_passed,
            score=1.0 if all_passed else pass_rate,
            details=details,
        )
