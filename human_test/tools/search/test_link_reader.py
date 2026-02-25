"""Debug: LinkReaderTool — test with Jina Reader backend and reader_fn injection.

Before running, optionally set:
    export JINA_API_KEY=your_key_here   (optional, 20 RPM without key)
"""

import asyncio
import os

from awe_agent.core.tool.search.constraints import SearchConstraints
from awe_agent.core.tool.search.link_reader_tool import LinkReaderTool
from awe_agent.core.tool.search.backends.reader.jina import JinaReaderBackend


# ── 1. Standalone JinaReaderBackend ───────────────────────────────────────


async def test_jina_backend_directly():
    """Directly call JinaReaderBackend.read_link() — bypass LinkReaderTool."""
    print("=" * 60)
    print("1. JinaReaderBackend standalone")
    print("=" * 60)

    backend = JinaReaderBackend()
    content = await backend.read_link("https://httpbin.org/html")
    print(f"  Length: {len(content)} chars")
    print(f"  First 300 chars:\n{content[:300]}")
    print()


async def test_jina_backend_real_page():
    """Fetch a real documentation page."""
    print("=" * 60)
    print("2. JinaReaderBackend — real documentation page")
    print("=" * 60)

    backend = JinaReaderBackend()
    content = await backend.read_link("https://docs.python.org/3/library/asyncio.html")
    print(f"  Length: {len(content)} chars")
    print(f"  First 500 chars:\n{content[:500]}")
    print()


async def test_jina_backend_pdf():
    """Fetch a PDF — Jina Reader natively supports PDFs."""
    print("=" * 60)
    print("3. JinaReaderBackend — PDF")
    print("=" * 60)

    backend = JinaReaderBackend()
    content = await backend.read_link("https://arxiv.org/pdf/1706.03762")
    print(f"  Length: {len(content)} chars")
    print(f"  First 500 chars:\n{content[:500]}")
    print()


async def test_jina_backend_with_target_selector():
    """Use CSS target selector to extract specific content."""
    print("=" * 60)
    print("4. JinaReaderBackend — with target selector")
    print("=" * 60)

    # Use .mw-parser-output — Wikipedia's actual content container
    backend = JinaReaderBackend(target_selector=".mw-parser-output")
    content = await backend.read_link("https://en.wikipedia.org/wiki/Python_(programming_language)")
    print(f"  Length: {len(content)} chars")
    print(f"  First 500 chars:\n{content[:500]}")
    print()


# ── 2. LinkReaderTool with backend ────────────────────────────────────────


async def test_link_reader_auto_discover():
    """LinkReaderTool with auto-discovered Jina backend."""
    print("=" * 60)
    print("5. LinkReaderTool — auto-discover backend")
    print("=" * 60)

    tool = LinkReaderTool()
    result = await tool.execute({"url": "https://httpbin.org/html"})
    print(f"  Length: {len(result)} chars")
    print(f"  First 300 chars:\n{result[:300]}")
    print()


async def test_link_reader_explicit_backend():
    """LinkReaderTool with explicit backend='jina'."""
    print("=" * 60)
    print("6. LinkReaderTool — explicit backend='jina'")
    print("=" * 60)

    tool = LinkReaderTool(backend="jina")
    result = await tool.execute({"url": "https://httpbin.org/html"})
    print(f"  Length: {len(result)} chars")
    print(f"  First 300 chars:\n{result[:300]}")
    print()


async def test_link_reader_inject_backend_instance():
    """LinkReaderTool with injected JinaReaderBackend instance."""
    print("=" * 60)
    print("7. LinkReaderTool — inject backend instance")
    print("=" * 60)

    backend = JinaReaderBackend()
    tool = LinkReaderTool(backend=backend)
    result = await tool.execute({"url": "https://httpbin.org/html"})
    print(f"  Length: {len(result)} chars")
    print(f"  First 300 chars:\n{result[:300]}")
    print()


async def test_link_reader_with_constraints():
    """Blocked URLs should be rejected without making any request."""
    print("=" * 60)
    print("8. LinkReaderTool — URL blocked by constraints")
    print("=" * 60)

    constraints = SearchConstraints.from_repo("django/django")
    tool = LinkReaderTool(constraints=constraints)

    blocked_urls = [
        "https://github.com/django/django/blob/main/README.rst",
        "https://github.com/django/django/issues/12345",
    ]
    for url in blocked_urls:
        result = await tool.execute({"url": url})
        print(f"  [BLOCKED] {url}")

    allowed_url = "https://httpbin.org/html"
    result = await tool.execute({"url": allowed_url})
    print(f"  [ALLOWED] {allowed_url} -> {len(result)} chars")
    print()


async def test_link_reader_truncation():
    """Verify token truncation with real content."""
    print("=" * 60)
    print("9. LinkReaderTool — truncation (max_content_tokens=200)")
    print("=" * 60)

    tool = LinkReaderTool(max_content_tokens=200)
    result = await tool.execute({"url": "https://docs.python.org/3/library/asyncio.html"})
    print(f"  Length: {len(result)} chars")
    print(f"  Truncated: {'truncated' in result}")
    print()


async def test_link_reader_empty_url():
    print("=" * 60)
    print("10. LinkReaderTool — empty URL")
    print("=" * 60)

    tool = LinkReaderTool()
    result = await tool.execute({"url": ""})
    print(f"  Result: {result}")
    print()


async def test_link_reader_no_backend():
    """Without API key — still works at 20 RPM but some domains may be blocked."""
    print("=" * 60)
    print("11. JinaReaderBackend — no API key (20 RPM)")
    print("=" * 60)

    saved = os.environ.pop("JINA_API_KEY", None)
    try:
        backend = JinaReaderBackend()
        content = await backend.read_link("https://example.com")
        print(f"  Length: {len(content)} chars")
        print(f"  Works without key: {len(content) > 0}")
    finally:
        if saved is not None:
            os.environ["JINA_API_KEY"] = saved
    print()


async def main():
    api_key = os.environ.get("JINA_API_KEY", "")
    if api_key:
        print(f"JINA_API_KEY set (500 RPM)")
    else:
        print("JINA_API_KEY not set (20 RPM, still works)")
    print()

    await test_link_reader_empty_url()
    await test_link_reader_with_constraints()
    await test_link_reader_no_backend()
    await test_jina_backend_directly()
    await test_jina_backend_real_page()
    await test_jina_backend_pdf()
    await test_jina_backend_with_target_selector()
    await test_link_reader_auto_discover()
    await test_link_reader_explicit_backend()
    await test_link_reader_inject_backend_instance()
    await test_link_reader_truncation()
    print("All LinkReaderTool tests done.")


if __name__ == "__main__":
    asyncio.run(main())
