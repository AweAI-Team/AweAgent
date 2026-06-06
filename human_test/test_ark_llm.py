"""Quick smoke test for Ark LLM backend and WebFetchTool.

Usage:
    export WEB_FETCH_ARK_BASE_URL="https://your-ark-endpoint/api/v3"
    export WEB_FETCH_ARK_API_KEY="your-key"
    export WEB_FETCH_ARK_MODEL="your-model"

    # Test 1: raw Ark client
    python human_test/test_ark_llm.py

    # Test 2: also test WebFetchTool (needs network for URL fetch)
    python human_test/test_ark_llm.py --with-tool
"""

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "llm" / "web_fetch" / "ark.yaml"


async def test_raw_client():
    """Test 1: create AsyncArk client and send a simple chat request."""
    print("=" * 60)
    print("Test 1: Raw Ark client (create_async_client)")
    print("=" * 60)

    from aweagent.core.config.loader import load_yaml
    from aweagent.core.llm.client import create_async_client

    config = load_yaml(str(CONFIG_PATH))
    print(f"  Config loaded: backend={config.get('backend')}, model={config.get('model')}")
    print(f"  base_url: {config.get('base_url')}")

    client = create_async_client(
        backend=config.get("backend", "openai"),
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
    )
    print(f"  Client type: {type(client).__name__}")

    model = config.get("model")
    print(f"  Sending chat request to model={model} ...")

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say 'hello' in one word."}],
        max_completion_tokens=64,
    )

    content = response.choices[0].message.content
    print(f"  Response: {content}")
    print(f"  Usage: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")
    print("  PASSED\n")


async def test_web_fetch_tool():
    """Test 2: WebFetchTool with Ark backend."""
    print("=" * 60)
    print("Test 2: WebFetchTool (Ark backend)")
    print("=" * 60)

    os.environ.setdefault("WEB_FETCH_CONFIG_PATH", str(CONFIG_PATH))

    from aweagent.core.tool.search.web_fetch_tool import WebFetchTool

    tool = WebFetchTool()
    # Force lazy-load to verify config is picked up
    tool._ensure_llm_loaded()
    print(f"  LLM client: {type(tool._llm).__name__}")
    print(f"  Params: {tool._llm_params}")

    result = await tool.execute({
        "url": "https://httpbin.org/html",
        "prompt": "What is on this page?",
    })
    print(f"  Result ({len(result)} chars): {result[:300]}")
    print("  PASSED\n")


async def main():
    # Validate env vars
    missing = []
    for var in ["WEB_FETCH_ARK_BASE_URL", "WEB_FETCH_ARK_API_KEY", "WEB_FETCH_ARK_MODEL"]:
        if not os.environ.get(var):
            missing.append(var)
    if missing:
        print(f"ERROR: Missing env vars: {', '.join(missing)}")
        print("Set them before running this script.")
        sys.exit(1)

    if not CONFIG_PATH.exists():
        print(f"ERROR: Config not found at {CONFIG_PATH}")
        sys.exit(1)

    await test_raw_client()

    if "--with-tool" in sys.argv:
        await test_web_fetch_tool()

    print("All tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
