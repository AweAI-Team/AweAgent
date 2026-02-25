"""Debug: LinkSummaryTool — verify full pipeline (fetch + summarize) with real backends.

Uses LinkReaderTool with aiohttp reader_fn and real LLM (loaded from YAML config).

Before running, set environment variables:
    export SERPAPI_API_KEY=your_key_here  (for search tests)

    # For LLM summarization, one of:
    export LINK_SUMMARY_CONFIG_PATH=configs/llm/link_summary/azure.yaml
    # Or:
    export LINK_SUMMARY_MODEL=gpt-4o-mini
    export OPENAI_API_KEY=...
    export OPENAI_BASE_URL=...
"""

import asyncio
import os
from pathlib import Path

import aiohttp

from awe_agent.core.tool.search.constraints import SearchConstraints
from awe_agent.core.tool.search.link_reader_tool import LinkReaderTool
from awe_agent.core.tool.search.link_summary_tool import LinkSummaryTool


# ── Helpers ─────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "llm" / "link_summary" / "azure.yaml"


async def simple_aiohttp_reader(url: str) -> str:
    """Minimal URL fetcher using aiohttp."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            resp.raise_for_status()
            return await resp.text()


def ensure_config():
    """Set LINK_SUMMARY_CONFIG_PATH if not already set."""
    if not os.environ.get("LINK_SUMMARY_CONFIG_PATH") and not os.environ.get("LINK_SUMMARY_MODEL"):
        if _DEFAULT_CONFIG.exists():
            os.environ["LINK_SUMMARY_CONFIG_PATH"] = str(_DEFAULT_CONFIG)
            print(f"  Auto-detected config: {_DEFAULT_CONFIG}")
        else:
            print(f"  WARNING: No config found at {_DEFAULT_CONFIG}")
            print("  Set LINK_SUMMARY_CONFIG_PATH or LINK_SUMMARY_MODEL env var.")


def make_reader() -> LinkReaderTool:
    return LinkReaderTool(reader_fn=simple_aiohttp_reader)


# ── Test scenarios ──────────────────────────────────────────────────────────


async def test_full_pipeline():
    """Full pipeline: real fetch + real LLM summarize."""
    print("=" * 60)
    print("1. Full pipeline: real fetch + real LLM summarize")
    print("=" * 60)
    ensure_config()

    tool = LinkSummaryTool(reader=make_reader())

    result = await tool.execute({
        "url": "https://docs.djangoproject.com/en/5.0/ref/models/querysets/",
        "goal": "How does QuerySet lazy evaluation work?",
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
    tool = LinkSummaryTool(constraints=constraints, reader=make_reader())

    urls = [
        ("https://github.com/django/django/issues/100", "read issue"),
        ("https://gitlab.com/django/django/merge_requests/1", "read MR"),
        ("https://httpbin.org/html", "read docs"),  # allowed
    ]
    for url, goal in urls:
        result = await tool.execute({"url": url, "goal": goal})
        tag = "BLOCKED" if "ACCESS DENIED" in result else "PASSED"
        print(f"  [{tag}] {url}")
    print()


async def test_no_llm_fallback():
    """Without LLM config, should return raw fetched content."""
    print("=" * 60)
    print("3. No LLM configured (raw content fallback)")
    print("=" * 60)

    saved_model = os.environ.pop("LINK_SUMMARY_MODEL", None)
    saved_config = os.environ.pop("LINK_SUMMARY_CONFIG_PATH", None)

    try:
        tool = LinkSummaryTool(reader=make_reader())
        result = await tool.execute({
            "url": "https://httpbin.org/html",
            "goal": "What is on this page?",
        })
        print(f"  Contains 'no LLM configured': {'no LLM configured' in result}")
        print(f"  Result length: {len(result)} chars")
        print(f"  First 200 chars: {result[:200]}")
    finally:
        if saved_model is not None:
            os.environ["LINK_SUMMARY_MODEL"] = saved_model
        if saved_config is not None:
            os.environ["LINK_SUMMARY_CONFIG_PATH"] = saved_config
    print()


async def test_empty_inputs():
    print("=" * 60)
    print("4. Empty inputs")
    print("=" * 60)

    tool = LinkSummaryTool(reader=make_reader())
    r1 = await tool.execute({"url": "", "goal": "test"})
    print(f"  Empty URL:  {r1}")

    r2 = await tool.execute({"url": "https://example.com", "goal": ""})
    print(f"  Empty goal: {r2}")
    print()


async def main():
    await test_empty_inputs()
    await test_url_blocked()
    await test_no_llm_fallback()
    await test_full_pipeline()
    print("All LinkSummaryTool tests done.")


if __name__ == "__main__":
    asyncio.run(main())
