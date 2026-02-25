"""Debug: SearchTool — verify real search via SerpAPI backend.

Before running, set:
    export SERPAPI_API_KEY=your_key_here
"""

import asyncio
import os

from awe_agent.core.tool.search.backends.search.serpapi import SerpAPIBackend
from awe_agent.core.tool.search.constraints import SearchConstraints
from awe_agent.core.tool.search.search_tool import SearchTool


# ── 1. Standalone SerpAPIBackend ──────────────────────────────────────────


async def test_serpapi_backend_directly():
    """Directly call SerpAPIBackend.search() — bypass SearchTool entirely."""
    print("=" * 60)
    print("1. SerpAPIBackend standalone")
    print("=" * 60)

    backend = SerpAPIBackend()
    results = await backend.search("python asyncio tutorial", num=3)

    print(f"  Got {len(results)} results")
    for r in results:
        print(f"  [{r['position']}] {r['title']}")
        print(f"      {r['url']}")
        print(f"      {r['description'][:80]}...")
    print()


# ── 2. SearchTool with SerpAPIBackend ─────────────────────────────────────


async def test_search_tool_single_query():
    """Single query through SearchTool (auto-discovers SerpAPIBackend)."""
    print("=" * 60)
    print("2. SearchTool — single query (auto-discover backend)")
    print("=" * 60)

    tool = SearchTool()
    result = await tool.execute({"query": "django queryset lazy evaluation"})
    print(result)
    print()


async def test_search_tool_with_backend_name():
    """Explicitly pass backend name 'serpapi'."""
    print("=" * 60)
    print("3. SearchTool — explicit backend='serpapi'")
    print("=" * 60)

    tool = SearchTool(backend="serpapi")
    result = await tool.execute({"query": "python type hints best practices", "num": 3})
    print(result)
    print()


async def test_search_tool_with_backend_instance():
    """Inject a SerpAPIBackend instance directly."""
    print("=" * 60)
    print("4. SearchTool — inject backend instance")
    print("=" * 60)

    backend = SerpAPIBackend()
    tool = SearchTool(backend=backend)
    result = await tool.execute({"query": "aiohttp client session example", "num": 3})
    print(result)
    print()


async def test_search_tool_with_constraints():
    """Search with constraint filtering — github.com/django/* should be filtered."""
    print("=" * 60)
    print("5. SearchTool — with constraints (django/django)")
    print("=" * 60)

    constraints = SearchConstraints.from_repo("django/django")
    tool = SearchTool(constraints=constraints)
    result = await tool.execute({"query": "django queryset filter github"})
    print(result)
    print()


async def test_search_tool_batch_query():
    """Batch query — multiple queries in one call."""
    print("=" * 60)
    print("6. SearchTool — batch query")
    print("=" * 60)

    tool = SearchTool()
    result = await tool.execute({
        "query": ["python asyncio tutorial", "pytorch nn.Linear usage"],
        "num": 3,
    })
    print(result)
    print()


async def test_search_tool_empty_query():
    """Empty query should return error without calling backend."""
    print("=" * 60)
    print("7. SearchTool — empty query")
    print("=" * 60)

    tool = SearchTool()
    result = await tool.execute({"query": ""})
    print(f"  Result: {result}")
    print()


async def test_search_tool_no_api_key():
    """Without API key, should get empty results (not crash)."""
    print("=" * 60)
    print("8. SerpAPIBackend — no API key")
    print("=" * 60)

    saved = os.environ.pop("SERPAPI_API_KEY", None)
    try:
        backend = SerpAPIBackend()
        results = await backend.search("test query")
        print(f"  Results: {results}")
    finally:
        if saved is not None:
            os.environ["SERPAPI_API_KEY"] = saved
    print()


async def main():
    api_key = os.environ.get("SERPAPI_API_KEY", "")
    if not api_key:
        print("WARNING: SERPAPI_API_KEY not set. Real search tests will fail.")
        print("Set it with: export SERPAPI_API_KEY=your_key_here\n")

    await test_search_tool_empty_query()
    await test_search_tool_no_api_key()
    await test_serpapi_backend_directly()
    await test_search_tool_single_query()
    await test_search_tool_with_backend_name()
    await test_search_tool_with_backend_instance()
    await test_search_tool_with_constraints()
    await test_search_tool_batch_query()
    print("All SearchTool tests done.")


if __name__ == "__main__":
    asyncio.run(main())
