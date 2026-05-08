"""Debug: WebFetchTool — verify full pipeline (fetch + summarize) with real backends.

Uses WebFetchRawTool with aiohttp reader_fn and real LLM (loaded from YAML config).

Before running, set environment variables:
    export SERPAPI_API_KEY=your_key_here  (for search tests)

    # For LLM summarization, one of:
    export WEB_FETCH_CONFIG_PATH=configs/llm/web_fetch/azure.yaml
    # Or:
    export WEB_FETCH_MODEL=gpt-4o-mini
    export OPENAI_API_KEY=...
    export OPENAI_BASE_URL=...
"""

import asyncio
import os
from pathlib import Path

import aiohttp

from awe_agent.core.tool.search.constraints import SearchConstraints
from awe_agent.core.tool.search.web_fetch_raw_tool import WebFetchRawTool
from awe_agent.core.tool.search.web_fetch_tool import WebFetchTool


# ── Helpers ─────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "llm" / "web_fetch" / "azure.yaml"


async def simple_aiohttp_reader(url: str) -> str:
    """Minimal URL fetcher using aiohttp."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            resp.raise_for_status()
            return await resp.text()


def ensure_config():
    """Set WEB_FETCH_CONFIG_PATH if not already set."""
    if not os.environ.get("WEB_FETCH_CONFIG_PATH") and not os.environ.get("WEB_FETCH_MODEL"):
        if _DEFAULT_CONFIG.exists():
            os.environ["WEB_FETCH_CONFIG_PATH"] = str(_DEFAULT_CONFIG)
            print(f"  Auto-detected config: {_DEFAULT_CONFIG}")
        else:
            print(f"  WARNING: No config found at {_DEFAULT_CONFIG}")
            print("  Set WEB_FETCH_CONFIG_PATH or WEB_FETCH_MODEL env var.")


def make_reader() -> WebFetchRawTool:
    return WebFetchRawTool(reader_fn=simple_aiohttp_reader)


# ── Test scenarios ──────────────────────────────────────────────────────────


async def test_full_pipeline():
    """Full pipeline: real fetch + real LLM summarize."""
    print("=" * 60)
    print("1. Full pipeline: real fetch + real LLM summarize")
    print("=" * 60)
    ensure_config()

    tool = WebFetchTool(reader=make_reader())

    result = await tool.execute({
        "url": "https://docs.djangoproject.com/en/5.0/ref/models/querysets/",
        "prompt": "How does QuerySet lazy evaluation work?",
    })
    print(f"  Result length: {len(result)} chars")
    print(f"  First 500 chars:\n{result[:500]}")
    print()


async def test_url_blocked():
    """Blocked URLs should be rejected without making any request."""
    print("=" * 60)
    print("2. URL blocked")
    print("=" * 60)

    constraints = SearchConstraints.from_repo("django/django")
    tool = WebFetchTool(constraints=constraints, reader=make_reader())

    urls = [
        ("https://github.com/django/django/issues/100", "read issue"),
        ("https://gitlab.com/django/django/merge_requests/1", "read MR"),
        ("https://httpbin.org/html", "read docs"),  # allowed
    ]
    for url, prompt in urls:
        result = await tool.execute({"url": url, "prompt": prompt})
        tag = "BLOCKED" if "ACCESS DENIED" in result else "PASSED"
        print(f"  [{tag}] {url}")
    print()


async def test_no_llm_fallback():
    """Without LLM config, should return raw fetched content."""
    print("=" * 60)
    print("3. No LLM configured (raw content fallback)")
    print("=" * 60)

    saved_model = os.environ.pop("WEB_FETCH_MODEL", None)
    saved_config = os.environ.pop("WEB_FETCH_CONFIG_PATH", None)

    try:
        tool = WebFetchTool(reader=make_reader())
        result = await tool.execute({
            "url": "https://httpbin.org/html",
            "prompt": "What is on this page?",
        })
        print(f"  Contains 'no LLM configured': {'no LLM configured' in result}")
        print(f"  Result length: {len(result)} chars")
        print(f"  First 200 chars: {result[:200]}")
    finally:
        if saved_model is not None:
            os.environ["WEB_FETCH_MODEL"] = saved_model
        if saved_config is not None:
            os.environ["WEB_FETCH_CONFIG_PATH"] = saved_config
    print()


async def test_empty_inputs():
    print("=" * 60)
    print("4. Empty inputs")
    print("=" * 60)

    tool = WebFetchTool(reader=make_reader())
    r1 = await tool.execute({"url": "", "prompt": "test"})
    print(f"  Empty URL:    {r1}")

    r2 = await tool.execute({"url": "https://example.com", "prompt": ""})
    print(f"  Empty prompt: {r2}")
    print()


async def main():
    await test_empty_inputs()
    await test_url_blocked()
    await test_no_llm_fallback()
    await test_full_pipeline()
    print("All WebFetchTool tests done.")


if __name__ == "__main__":
    asyncio.run(main())
