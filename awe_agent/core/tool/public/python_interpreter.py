"""PythonInterpreterTool — execute Python in a remote SandboxFusion service.

This is a thin HTTP client. It does not run code locally or inside a
:class:`RuntimeSession`; it submits code to one of several SandboxFusion
endpoints (deployed separately, see https://github.com/bytedance/SandboxFusion)
through the ``sandbox_fusion`` client SDK and returns the captured stdout/stderr.

Endpoints are resolved (first match wins):

1. The ``endpoints`` constructor argument (a list, or a comma-separated string).
2. ``SANDBOX_FUSION_ENDPOINTS`` — comma-separated list (the multi-endpoint form).
3. ``SANDBOX_FUSION_ENDPOINT`` — a single endpoint (legacy fallback).
4. The built-in default list.

Each attempt picks a random endpoint, so a pool spreads load and tolerates a
dead node. ``sandbox_fusion`` and its ``requests`` dependency are optional and
imported lazily, so importing this module never requires them; install with
``pip install sandbox_fusion`` (or the ``iterresearch`` extra) to use the tool.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
from typing import Any

from awe_agent.core.runtime.protocol import RuntimeSession
from awe_agent.core.tool.protocol import Tool

logger = logging.getLogger(__name__)

# Imported lazily so the module loads even when sandbox_fusion is absent.
try:  # pragma: no cover - import guard
    from sandbox_fusion import RunCodeRequest as _RunCodeRequest
    from sandbox_fusion import run_code as _run_code
except Exception:  # pragma: no cover - sandbox_fusion not installed
    _RunCodeRequest = None
    _run_code = None

_DEFAULT_ENDPOINTS = [
    "http://22.13.220.234:8080",
    "http://22.13.220.235:8080",
    "http://22.13.220.232:8080",
]

_DEFAULT_RUN_TIMEOUT = 50
_DEFAULT_MAX_ATTEMPTS = 8

# Matches a fenced code block: ```lang\n<code>``` — captures <code>.
_CODE_FENCE_RE = re.compile(r"```[^\n]*\n(.+?)```", re.DOTALL)


def _resolve_endpoints(endpoints: list[str] | str | None) -> list[str]:
    """Resolve the endpoint pool from an explicit value or environment."""
    raw: list[str] | str | None = endpoints
    if raw is None:
        raw = os.environ.get("SANDBOX_FUSION_ENDPOINTS") or os.environ.get(
            "SANDBOX_FUSION_ENDPOINT"
        )
    if raw is None:
        return list(_DEFAULT_ENDPOINTS)
    if isinstance(raw, str):
        raw = raw.split(",")
    return [e.strip() for e in raw if e and e.strip()]


class PythonInterpreterTool(Tool):
    """Run Python code in a remote SandboxFusion sandbox.

    Args:
        endpoints: Endpoint pool (list or comma-separated string). Falls back to
            ``SANDBOX_FUSION_ENDPOINTS`` / ``SANDBOX_FUSION_ENDPOINT`` env vars,
            then a built-in default list.
        timeout: Client-side request timeout (seconds) per attempt.
        run_timeout: Server-side execution timeout (seconds) for the code.
        max_attempts: Number of attempts, each on a randomly chosen endpoint.
        run_code: Override the ``sandbox_fusion.run_code`` callable (for tests).
        request_cls: Override the ``sandbox_fusion.RunCodeRequest`` class (for tests).
    """

    def __init__(
        self,
        endpoints: list[str] | str | None = None,
        timeout: int = _DEFAULT_RUN_TIMEOUT,
        run_timeout: int = _DEFAULT_RUN_TIMEOUT,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        run_code: Any = None,
        request_cls: Any = None,
    ) -> None:
        self._endpoints = _resolve_endpoints(endpoints)
        self._timeout = timeout
        self._run_timeout = run_timeout
        self._max_attempts = max(int(max_attempts), 1)
        self._run_code = run_code or _run_code
        self._request_cls = request_cls or _RunCodeRequest

    @property
    def name(self) -> str:
        return "python_interpreter"

    @property
    def description(self) -> str:
        return (
            "Executes arbitrary Python code in a secure, sandboxed environment. "
            "This tool is designed for performing complex calculations, data "
            "manipulations, string processing, logical operations, and general "
            "programming tasks that require programmatic logic. It can define "
            "variables, use built-in functions, and print results to standard "
            "output. It operates with a limited set of pre-installed libraries and "
            "cannot access local files, external networks, or system resources."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "The Python code to execute. This should be a complete and "
                        "runnable Python script or series of statements. All output "
                        "should be explicitly printed using `print()` functions."
                    ),
                }
            },
            "required": ["code"],
        }

    async def execute(
        self,
        params: dict[str, Any],
        session: RuntimeSession | None = None,
    ) -> str:
        if self._run_code is None or self._request_cls is None:
            return (
                "[Python Interpreter Error]: sandbox_fusion is not installed. "
                "Install it with `pip install sandbox_fusion`."
            )

        code = self._extract_code(params)
        if not code.strip():
            return "[Python Interpreter Error]: Empty code."

        if not self._endpoints:
            return (
                "[Python Interpreter Error]: No SandboxFusion endpoint configured. "
                "Set SANDBOX_FUSION_ENDPOINTS (comma-separated) or "
                "SANDBOX_FUSION_ENDPOINT."
            )

        last_error: str | None = None
        for attempt in range(self._max_attempts):
            endpoint = random.choice(self._endpoints)
            try:
                code_result = await asyncio.to_thread(
                    self._run_code,
                    self._request_cls(
                        code=code, language="python", run_timeout=self._run_timeout
                    ),
                    max_attempts=1,
                    client_timeout=self._timeout,
                    endpoint=endpoint,
                )
                return self._format_result(code_result)
            except Exception as exc:  # noqa: BLE001 - report the failure as observation
                # Match requests' Timeout (and its ConnectTimeout/ReadTimeout
                # subclasses) by name, without importing requests here.
                is_timeout = any(c.__name__ == "Timeout" for c in type(exc).__mro__)
                if is_timeout:
                    last_error = (
                        "[Python Interpreter Error] TimeoutError: Execution timed "
                        f"out on endpoint {endpoint}."
                    )
                else:
                    last_error = f"[Python Interpreter Error]: {exc} on endpoint {endpoint}"
                logger.warning(
                    "PythonInterpreter attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_attempts,
                    last_error,
                )

        return last_error or "[Python Interpreter Error]: All attempts failed."

    # ── Internal ─────────────────────────────────────────────────────────

    def _extract_code(self, params: dict[str, Any] | str) -> str:
        """Pull the code string out of ``params``, unwrapping a fenced block."""
        if isinstance(params, str):
            params = self._loads_loose(params)
        code = ""
        if isinstance(params, dict):
            code = params.get("code", "") or params.get("raw", "") or ""
        if not isinstance(code, str):
            code = str(code)
        match = _CODE_FENCE_RE.search(code)
        if match:
            code = match.group(1)
        return code

    @staticmethod
    def _loads_loose(text: str) -> Any:
        """Parse a JSON(5) string; tolerate malformed input by returning {}."""
        import json

        try:
            return json.loads(text)
        except Exception:
            try:
                import json5  # optional, lenient parsing

                return json5.loads(text)
            except Exception:
                return {}

    def _format_result(self, code_result: Any) -> str:
        run_result = code_result.run_result
        parts: list[str] = []
        if run_result.stdout:
            parts.append(f"stdout:\n{run_result.stdout}")
        if run_result.stderr:
            parts.append(f"stderr:\n{run_result.stderr}")
        execution_time = getattr(run_result, "execution_time", 0) or 0
        if execution_time >= self._run_timeout - 1:
            parts.append("[PythonInterpreter Error] TimeoutError: Execution timed out.")
        result = "\n".join(parts)
        return result if result.strip() else "Finished execution."
