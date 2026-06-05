"""Tests for the public PythonInterpreterTool (SandboxFusion HTTP client).

``sandbox_fusion`` is an optional dependency, so these tests inject a fake
``run_code`` / ``request_cls`` instead of contacting a real sandbox.
"""

from __future__ import annotations

from typing import Any

import pytest

from awe_agent.core.tool.public.python_interpreter import (
    PythonInterpreterTool,
    _resolve_endpoints,
)


class _FakeRunResult:
    def __init__(self, stdout: str = "", stderr: str = "", execution_time: float = 0.0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.execution_time = execution_time


class _FakeCodeResult:
    def __init__(self, run_result: _FakeRunResult) -> None:
        self.run_result = run_result


class _FakeRequest:
    """Stand-in for sandbox_fusion.RunCodeRequest; records the code it received."""

    last: dict[str, Any] = {}

    def __init__(self, code: str, language: str, run_timeout: int) -> None:
        self.code = code
        self.language = language
        self.run_timeout = run_timeout
        _FakeRequest.last = {"code": code, "language": language, "run_timeout": run_timeout}


def _ok_run_code(stdout: str = "2\n", stderr: str = "", execution_time: float = 0.1):
    def run_code(request, *, max_attempts, client_timeout, endpoint):  # noqa: ANN001
        return _FakeCodeResult(
            _FakeRunResult(stdout=stdout, stderr=stderr, execution_time=execution_time)
        )

    return run_code


def _tool(**kwargs: Any) -> PythonInterpreterTool:
    kwargs.setdefault("endpoints", ["http://sandbox:8080"])
    kwargs.setdefault("request_cls", _FakeRequest)
    return PythonInterpreterTool(**kwargs)


# ── Endpoint resolution ──────────────────────────────────────────────────


def test_resolve_endpoints_splits_comma_list():
    assert _resolve_endpoints("http://a:8080, http://b:8080 ,http://c:8080") == [
        "http://a:8080",
        "http://b:8080",
        "http://c:8080",
    ]


def test_resolve_endpoints_accepts_list():
    assert _resolve_endpoints(["http://a", "http://b"]) == ["http://a", "http://b"]


def test_resolve_endpoints_prefers_plural_env(monkeypatch):
    monkeypatch.setenv("SANDBOX_FUSION_ENDPOINTS", "http://x:8080,http://y:8080")
    monkeypatch.setenv("SANDBOX_FUSION_ENDPOINT", "http://legacy:8080")
    assert _resolve_endpoints(None) == ["http://x:8080", "http://y:8080"]


def test_resolve_endpoints_falls_back_to_singular_env(monkeypatch):
    monkeypatch.delenv("SANDBOX_FUSION_ENDPOINTS", raising=False)
    monkeypatch.setenv("SANDBOX_FUSION_ENDPOINT", "http://legacy:8080")
    assert _resolve_endpoints(None) == ["http://legacy:8080"]


def test_resolve_endpoints_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("SANDBOX_FUSION_ENDPOINTS", raising=False)
    monkeypatch.delenv("SANDBOX_FUSION_ENDPOINT", raising=False)
    assert _resolve_endpoints(None)  # built-in default list is non-empty


# ── Execution ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_returns_stdout():
    tool = _tool(run_code=_ok_run_code(stdout="4\n"))
    obs = await tool.execute({"code": "print(2 + 2)"})
    assert "stdout:\n4" in obs


@pytest.mark.asyncio
async def test_execute_unwraps_fenced_code():
    tool = _tool(run_code=_ok_run_code())
    await tool.execute({"code": "```python\nprint('hi')\n```"})
    assert _FakeRequest.last["code"] == "print('hi')\n"


@pytest.mark.asyncio
async def test_execute_empty_code_short_circuits():
    tool = _tool(run_code=_ok_run_code())
    assert await tool.execute({"code": "   "}) == "[Python Interpreter Error]: Empty code."


@pytest.mark.asyncio
async def test_execute_flags_timeout():
    # execution_time >= run_timeout - 1 → timeout marker appended.
    tool = _tool(run_code=_ok_run_code(stdout="x\n", execution_time=50), run_timeout=50)
    obs = await tool.execute({"code": "print('x')"})
    assert "TimeoutError" in obs


@pytest.mark.asyncio
async def test_execute_retries_then_succeeds():
    calls = {"n": 0}

    def flaky(request, *, max_attempts, client_timeout, endpoint):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("connection refused")
        return _FakeCodeResult(_FakeRunResult(stdout="ok\n"))

    tool = _tool(run_code=flaky, max_attempts=8)
    obs = await tool.execute({"code": "print('ok')"})
    assert "stdout:\nok" in obs
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_execute_all_attempts_fail_returns_last_error():
    attempts = {"n": 0}

    def always_fail(request, *, max_attempts, client_timeout, endpoint):  # noqa: ANN001
        attempts["n"] += 1
        raise RuntimeError("boom")

    tool = _tool(run_code=always_fail, max_attempts=3)
    obs = await tool.execute({"code": "print(1)"})
    assert obs.startswith("[Python Interpreter Error]")
    assert "boom" in obs
    assert attempts["n"] == 3  # all attempts used (no early exit)


@pytest.mark.asyncio
async def test_execute_without_sandbox_fusion_reports_clearly():
    tool = _tool(run_code=_ok_run_code())
    tool._run_code = None  # simulate sandbox_fusion not installed  # noqa: SLF001
    tool._request_cls = None  # noqa: SLF001
    obs = await tool.execute({"code": "print(1)"})
    assert "sandbox_fusion is not installed" in obs


@pytest.mark.asyncio
async def test_execute_without_endpoints_reports_clearly():
    tool = _tool(run_code=_ok_run_code(), endpoints=[])
    obs = await tool.execute({"code": "print(1)"})
    assert "No SandboxFusion endpoint configured" in obs


def test_schema_uses_internal_name():
    tool = _tool(run_code=_ok_run_code())
    assert tool.name == "python_interpreter"
    assert tool.parameters["required"] == ["code"]
