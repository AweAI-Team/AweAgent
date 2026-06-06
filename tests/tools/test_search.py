"""Tests for web tools: SearchConstraints, WebSearchTool, WebFetchRawTool, WebFetchTool."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aweagent.core.tool.search.backends.reader import get_reader_backend
from aweagent.core.tool.search.backends.reader.jina import JinaReaderBackend
from aweagent.core.tool.search.backends.search import get_search_backend
from aweagent.core.tool.search.backends.search.serpapi import SerpAPIBackend
from aweagent.core.tool.search.constraints import SearchConstraints
from aweagent.core.tool.search.web_fetch_raw_tool import WebFetchRawTool
from aweagent.core.tool.search.web_fetch_tool import WebFetchTool
from aweagent.core.tool.search.web_search_tool import WebSearchTool

# ── SearchConstraints ───────────────────────────────────────────────────────


class TestSearchConstraints:

    def test_from_repo_with_owner(self):
        c = SearchConstraints.from_repo("django/django")
        assert c._repo_owner == "django"
        assert c._repo_name == "django"
        assert len(c.blocked_patterns["url"]) == 3  # github, gitlab, raw

    def test_from_repo_without_owner(self):
        c = SearchConstraints.from_repo("flask")
        assert c._repo_owner is None
        assert c._repo_name == "flask"
        # Patterns use [^/]+ wildcard for owner
        assert any("[^/]+" in p for p in c.blocked_patterns["url"])

    def test_from_repo_special_chars(self):
        """Repo names with regex-special chars should be escaped."""
        c = SearchConstraints.from_repo("owner/my.repo+plus")
        assert c.is_url_blocked("https://github.com/owner/my.repo+plus/issues")
        # Should NOT match without the dot (regex . would match any char)
        assert not c.is_url_blocked("https://github.com/owner/myXrepo+plus/issues")

    def test_is_url_blocked(self):
        c = SearchConstraints.from_repo("django/django")
        assert c.is_url_blocked("https://github.com/django/django/pull/42")
        assert c.is_url_blocked("https://GITHUB.COM/django/django")  # case insensitive
        assert c.is_url_blocked("https://gitlab.com/django/django/issues/1")
        assert not c.is_url_blocked("https://stackoverflow.com/questions/django")
        assert not c.is_url_blocked("https://github.com/django/django-extensions")

    def test_is_url_blocked_invalid_regex(self):
        """Invalid regex patterns should not crash, just log warning."""
        bad_pattern = "[invalid"
        c = SearchConstraints(blocked_patterns={"url": [bad_pattern]})
        assert not c.is_url_blocked("https://example.com")

    def test_filter_search_results(self):
        c = SearchConstraints.from_repo("django/django")
        results = [
            {"url": "https://github.com/django/django/pull/42", "title": "Fix"},
            {"url": "https://stackoverflow.com/q/123", "title": "Help"},
            {"url": "https://gitlab.com/django/django/issues/1", "title": "Bug"},
            {"url": "https://docs.djangoproject.com/en/5.0/", "title": "Docs"},
        ]
        filtered, count = c.filter_search_results(results)
        assert count == 2
        assert len(filtered) == 2
        assert filtered[0]["url"] == "https://stackoverflow.com/q/123"
        assert filtered[1]["url"] == "https://docs.djangoproject.com/en/5.0/"

    def test_filter_empty_patterns(self):
        c = SearchConstraints()
        results = [{"url": "https://example.com"}]
        filtered, count = c.filter_search_results(results)
        assert count == 0
        assert filtered == results

    def test_filter_multiple_fields(self):
        c = SearchConstraints(blocked_patterns={
            "url": [r".*blocked\.com.*"],
            "title": [r".*SECRET.*"],
        })
        results = [
            {"url": "https://ok.com", "title": "SECRET doc"},
            {"url": "https://blocked.com/page", "title": "Fine"},
            {"url": "https://ok.com", "title": "Fine"},
        ]
        filtered, count = c.filter_search_results(results)
        assert count == 2
        assert len(filtered) == 1
        assert filtered[0]["title"] == "Fine"

    def test_get_bash_blocklist_patterns(self):
        c = SearchConstraints.from_repo("django/django")
        patterns = c.get_bash_blocklist_patterns()
        assert any("git\\s+clone" in p for p in patterns)
        assert any("api\\.github\\.com" in p for p in patterns)

    def test_get_bash_blocklist_empty(self):
        c = SearchConstraints()
        assert c.get_bash_blocklist_patterns() == []

    def test_merge(self):
        c1 = SearchConstraints.from_repo("django/django")
        c2 = SearchConstraints(blocked_patterns={
            "url": [r".*extra\.com.*"],
            "title": [r".*BLOCKED.*"],
        })
        merged = c1.merge(c2)
        # Has both url sets
        assert len(merged.blocked_patterns["url"]) == len(c1.blocked_patterns["url"]) + 1
        assert "title" in merged.blocked_patterns
        # Original objects unchanged
        assert "title" not in c1.blocked_patterns

    def test_merge_deduplicates(self):
        c1 = SearchConstraints(blocked_patterns={"url": ["pattern_a"]})
        c2 = SearchConstraints(blocked_patterns={"url": ["pattern_a", "pattern_b"]})
        merged = c1.merge(c2)
        assert merged.blocked_patterns["url"] == ["pattern_a", "pattern_b"]


# ── WebSearchTool ───────────────────────────────────────────────────────────


class TestWebSearchTool:

    @pytest.mark.asyncio
    async def test_single_query(self):
        async def fake_search(**kwargs):
            return [
                {"title": "Result 1", "url": "https://a.com", "description": "Desc 1"},
                {"title": "Result 2", "url": "https://b.com", "description": "Desc 2"},
            ]

        tool = WebSearchTool(search_fn=fake_search)
        result = await tool.execute({"query": "python async"})
        assert "python async" in result
        assert "Result 1" in result
        assert "Result 2" in result

    @pytest.mark.asyncio
    async def test_batch_query(self):
        calls: list[str] = []

        async def fake_search(**kwargs):
            calls.append(kwargs["query"])
            return [{"title": f"For: {kwargs['query']}", "url": "https://x.com"}]

        tool = WebSearchTool(search_fn=fake_search)
        result = await tool.execute({"query": ["query1", "query2"]})
        assert len(calls) == 2
        assert "query1" in result
        assert "query2" in result

    @pytest.mark.asyncio
    async def test_constraint_filtering(self):
        async def fake_search(**kwargs):
            return [
                {"title": "Repo PR", "url": "https://github.com/django/django/pull/1"},
                {"title": "SO Answer", "url": "https://stackoverflow.com/q/123"},
            ]

        constraints = SearchConstraints.from_repo("django/django")
        tool = WebSearchTool(search_fn=fake_search, constraints=constraints)
        result = await tool.execute({"query": "django bug"})
        assert "SO Answer" in result
        assert "Repo PR" not in result
        assert "1 result(s) filtered" in result

    @pytest.mark.asyncio
    async def test_empty_query(self):
        tool = WebSearchTool()
        result = await tool.execute({"query": ""})
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_search_fn_returns_json_string(self):
        """search_fn returns a JSON string instead of a dict/list."""
        import json

        async def fake_search(**kwargs):
            return json.dumps({"results": [{"title": "Found", "url": "https://x.com"}]})

        tool = WebSearchTool(search_fn=fake_search)
        result = await tool.execute({"query": "test"})
        assert "Found" in result

    @pytest.mark.asyncio
    async def test_search_fn_raises(self):
        """search_fn raises an exception — should not crash, returns no results."""
        async def failing_search(**kwargs):
            raise ConnectionError("timeout")

        tool = WebSearchTool(search_fn=failing_search, max_attempts=2)
        result = await tool.execute({"query": "test"})
        assert "No results found" in result

    @pytest.mark.asyncio
    async def test_num_and_start_passed(self):
        received: dict[str, Any] = {}

        async def capture_search(**kwargs):
            received.update(kwargs)
            return []

        tool = WebSearchTool(search_fn=capture_search)
        await tool.execute({"query": "test", "num": 5, "start": 10})
        assert received["num"] == 5
        assert received["start"] == 10


class TestBackendAutodiscovery:

    def test_default_backend_autodiscovery_picks_public_defaults(self, monkeypatch):
        monkeypatch.delenv("SEARCH_BACKEND", raising=False)
        monkeypatch.delenv("READER_BACKEND", raising=False)

        assert isinstance(get_search_backend(), SerpAPIBackend)
        assert isinstance(get_reader_backend(), JinaReaderBackend)


# ── WebFetchRawTool ─────────────────────────────────────────────────────────


class TestWebFetchRawTool:

    @pytest.mark.asyncio
    async def test_fetch_plain_text(self):
        async def fake_reader(url):
            return "Hello, this is the page content."

        tool = WebFetchRawTool(reader_fn=fake_reader)
        result = await tool.execute({"url": "https://example.com"})
        assert "Hello, this is the page content." in result

    @pytest.mark.asyncio
    async def test_fetch_json_response(self):
        async def fake_reader(url):
            return {"content": "Extracted content here"}

        tool = WebFetchRawTool(reader_fn=fake_reader)
        result = await tool.execute({"url": "https://example.com"})
        assert "Extracted content here" in result

    @pytest.mark.asyncio
    async def test_fetch_json_string(self):
        import json

        async def fake_reader(url):
            return json.dumps({"content": "From JSON string"})

        tool = WebFetchRawTool(reader_fn=fake_reader)
        result = await tool.execute({"url": "https://example.com"})
        assert "From JSON string" in result

    @pytest.mark.asyncio
    async def test_url_blocked(self):
        constraints = SearchConstraints.from_repo("django/django")
        tool = WebFetchRawTool(constraints=constraints, reader_fn=AsyncMock())
        result = await tool.execute({"url": "https://github.com/django/django/blob/main/README.md"})
        assert "ACCESS DENIED" in result

    @pytest.mark.asyncio
    async def test_empty_url(self):
        tool = WebFetchRawTool()
        result = await tool.execute({"url": ""})
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_truncation(self):
        long_content = "word " * 100000  # very long

        async def fake_reader(url):
            return long_content

        tool = WebFetchRawTool(reader_fn=fake_reader, max_content_tokens=100)
        result = await tool.execute({"url": "https://example.com"})
        assert "truncated" in result
        assert len(result) < len(long_content)

    @pytest.mark.asyncio
    async def test_reader_fn_raises(self):
        async def failing_reader(url):
            raise OSError("network error")

        tool = WebFetchRawTool(reader_fn=failing_reader, max_attempts=1)
        result = await tool.execute({"url": "https://example.com"})
        assert "Error: failed to fetch" in result


# ── WebFetchTool ────────────────────────────────────────────────────────────


def test_web_fetch_yaml_config_inherits_all_fields(tmp_path):
    """WebFetchTool._ensure_llm_loaded builds a full LLMConfig from YAML.

    Verifies that fields like thinking, reasoning, extra are NOT silently
    dropped when loading from a YAML config file.
    """
    import yaml

    yaml_content = {
        "backend": "openai",
        "model": "gpt-4o-mini",
        "api_key": "test-key",
        "thinking": True,
        "thinking_budget": 5000,
        "reasoning": {"preserve": True, "format": "reasoning_content"},
        "extra": {"enable_thinking": True},
        "params": {"temperature": 0.5},
    }
    config_path = tmp_path / "test_llm.yaml"
    config_path.write_text(yaml.dump(yaml_content))

    tool = WebFetchTool(llm_config_path=str(config_path))
    tool._ensure_llm_loaded()

    assert tool._llm is not None
    cfg = tool._llm.config
    assert cfg.backend == "openai"
    assert cfg.model == "gpt-4o-mini"
    assert cfg.thinking is True
    assert cfg.thinking_budget == 5000
    assert cfg.reasoning.preserve is True
    assert cfg.extra.get("enable_thinking") is True


def _make_mock_llm(summary_text: str = "This is the summary.") -> MagicMock:
    """Create a mock LLMClient for WebFetchTool tests."""
    from aweagent.core.llm.types import LLMResponse

    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(return_value=LLMResponse(content=summary_text))
    return mock_llm


class TestWebFetchTool:

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """End-to-end: fetch + summarize via LLMClient."""
        async def fake_reader(url):
            return "Django is a Python web framework."

        reader = WebFetchRawTool(reader_fn=fake_reader)
        mock_llm = _make_mock_llm("Django is a high-level web framework for Python.")

        tool = WebFetchTool(
            llm=mock_llm,
            reader=reader,
        )
        result = await tool.execute({
            "url": "https://docs.djangoproject.com",
            "prompt": "What is Django?",
        })
        assert "Summary of" in result
        assert "high-level web framework" in result

        # Verify LLMClient.chat was called with Message objects
        call_args = mock_llm.chat.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert "What is Django?" in messages[1].content

    @pytest.mark.asyncio
    async def test_url_blocked(self):
        constraints = SearchConstraints.from_repo("django/django")
        tool = WebFetchTool(constraints=constraints)
        result = await tool.execute({
            "url": "https://github.com/django/django/issues/123",
            "prompt": "read issue",
        })
        assert "ACCESS DENIED" in result

    @pytest.mark.asyncio
    async def test_empty_prompt(self):
        tool = WebFetchTool()
        result = await tool.execute({"url": "https://example.com", "prompt": ""})
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_llm_params_passed(self):
        """Verify custom llm_params are forwarded to LLMClient.chat()."""
        async def fake_reader(url):
            return "content"

        reader = WebFetchRawTool(reader_fn=fake_reader)
        mock_llm = _make_mock_llm()

        tool = WebFetchTool(
            llm=mock_llm,
            reader=reader,
            llm_params={"temperature": 0.7, "max_tokens": 2048},
        )
        await tool.execute({"url": "https://example.com", "prompt": "summarize"})

        call_kwargs = mock_llm.chat.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_fallback_when_no_llm(self):
        """Without LLM client, should return raw content."""
        async def fake_reader(url):
            return "Raw page content"

        reader = WebFetchRawTool(reader_fn=fake_reader)
        tool = WebFetchTool(reader=reader)
        result = await tool.execute({
            "url": "https://example.com",
            "prompt": "summarize",
        })
        assert "no LLM configured" in result
        assert "Raw page content" in result

    @pytest.mark.asyncio
    async def test_llm_failure_returns_raw_content(self):
        """When LLMClient.chat() fails, should return raw content with error note."""
        async def fake_reader(url):
            return "Fallback content"

        reader = WebFetchRawTool(reader_fn=fake_reader)
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=RuntimeError("LLM down"))

        tool = WebFetchTool(
            llm=mock_llm,
            reader=reader,
            max_attempts=1,
        )
        result = await tool.execute({
            "url": "https://example.com",
            "prompt": "summarize",
        })
        assert "Failed to summarize" in result
        assert "Fallback content" in result

    @pytest.mark.asyncio
    async def test_fetch_error_propagated(self):
        """When reader fails, error should be returned directly."""
        async def failing_reader(url):
            raise OSError("network error")

        reader = WebFetchRawTool(reader_fn=failing_reader, max_attempts=1)
        tool = WebFetchTool(reader=reader)
        result = await tool.execute({
            "url": "https://example.com",
            "prompt": "summarize",
        })
        assert "Error: failed to fetch" in result
