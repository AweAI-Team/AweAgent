# Web Tools

Web search, fetch + summarize, and raw fetch — with pluggable backends.

## Directory Structure

```
aweagent/core/tool/search/
├── backends/
│   ├── search/                  # search backends
│   │   ├── __init__.py          # search_backend_registry + get_search_backend()
│   │   └── serpapi.py           # SerpAPIBackend
│   └── reader/                  # reader backends
│       ├── __init__.py          # reader_backend_registry + get_reader_backend()
│       └── jina.py              # JinaReaderBackend
├── constraints.py               # SearchConstraints (anti-hack URL/result filtering)
├── prompts.py                   # LLM prompt templates for web_fetch
├── web_search_tool.py           # WebSearchTool
├── web_fetch_tool.py            # WebFetchTool (fetch + LLM summarize)
├── web_fetch_raw_tool.py        # WebFetchRawTool (fetch raw content, no LLM)
└── README.md
```

## Tools

| Tool | Name | Description |
|------|------|-------------|
| `WebSearchTool` | `web_search` | Web search with anti-hack constraint filtering |
| `WebFetchTool` | `web_fetch` | Fetch + LLM-process via a prompt (depends on `WebFetchRawTool`) |
| `WebFetchRawTool` | `web_fetch_raw` | Fetch raw content (no LLM) |

---

## WebSearchTool

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SERPAPI_API_KEY` | Yes | SerpAPI key for web search |
| `SEARCH_BACKEND` | No | Override backend auto-discovery (e.g. `serpapi`). Default: auto-discover first available |
| `ENGINE` | No | Search engine name passed to SerpAPI. Default: `google` |

### Usage

**1. Environment variables (recommended):**

```bash
export SERPAPI_API_KEY=your_key_here
export SEARCH_BACKEND=serpapi  # optional, auto-discovered if only one backend
```

```python
from aweagent.core.tool.search import WebSearchTool

tool = WebSearchTool()
result = await tool.execute({"query": "python asyncio tutorial"})
result = await tool.execute({"query": ["query1", "query2"], "num": 3})
```

**2. Explicit backend via constructor:**

```python
# By registry name
tool = WebSearchTool(backend="serpapi")

# By instance (useful for custom api_key or timeout)
from aweagent.core.tool.search.backends.search.serpapi import SerpAPIBackend
tool = WebSearchTool(backend=SerpAPIBackend(api_key="your_key_here", timeout=60))
```

Backend resolution order: explicit `backend` argument → `SEARCH_BACKEND` env var → auto-discover first available.

### Manual Testing

```bash
export SERPAPI_API_KEY=your_key_here
python human_test/tools/search/test_web_search_tool.py
python human_test/tools/search/test_constraints.py
```

---

## WebFetchRawTool

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `JINA_API_KEY` | No | Jina API key. Optional — works without key at 20 RPM, 500 RPM with free key (10M free tokens) |
| `READER_BACKEND` | No | Override backend auto-discovery (e.g. `jina`). Default: auto-discover first available |

### Usage

**1. Environment variables (recommended):**

```bash
export JINA_API_KEY=your_key_here   # optional, increases rate limit
export READER_BACKEND=jina           # optional, auto-discovered if only one backend
```

```python
from aweagent.core.tool.search import WebFetchRawTool

tool = WebFetchRawTool()
result = await tool.execute({"url": "https://example.com"})
```

**2. Explicit backend via constructor:**

```python
# By registry name
tool = WebFetchRawTool(backend="jina")

# By instance (useful for custom options)
from aweagent.core.tool.search.backends.reader.jina import JinaReaderBackend
tool = WebFetchRawTool(backend=JinaReaderBackend(
    api_key="your_key_here",
    target_selector="article",       # CSS selector to extract specific elements
    wait_for_selector=".content",    # wait for JS-rendered content
    return_format="markdown",        # markdown (default), html, text
))
```

Backend resolution order: explicit `backend` argument → `reader_fn` callable → `READER_BACKEND` env var → auto-discover first available.

### Jina Reader Features

- Supports web pages, PDFs, and JS-rendered SPAs
- AI-generated image captions (`with_generated_alt=True`)
- CSS selector targeting (`target_selector`, `wait_for_selector`, `remove_selector`)
- Multiple output formats: markdown, html, text, screenshot
- Rendering engines: `browser` (headless Chrome), `direct` (fast, no JS), `cf-browser-rendering`
- Server-side caching (3600s default, configurable via `no_cache`)

### Manual Testing

```bash
export JINA_API_KEY=your_key_here  # optional
python human_test/tools/search/test_web_fetch_raw_tool.py
```

---

## WebFetchTool

Combines `WebFetchRawTool` (fetch) + LLM (run prompt). Takes a `url` and a
`prompt`, fetches the page, and returns the LLM's response to the prompt
applied against the page content. Internally creates a `WebFetchRawTool`,
so reader backend configuration (env vars, auto-discovery) applies equally here.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `JINA_API_KEY` | No | Jina API key for reader backend (same as WebFetchRawTool) |
| `READER_BACKEND` | No | Override reader backend (same as WebFetchRawTool) |
| `WEB_FETCH_CONFIG_PATH` | For summarization | Path to LLM YAML config |
| `WEB_FETCH_MODEL` | For summarization | LLM model name (alternative to config) |
| `OPENAI_API_KEY` | For summarization | OpenAI API key (when using env-var config) |

If no LLM is configured, returns raw fetched content without summarization.

### Usage

```python
from aweagent.core.tool.search import WebFetchTool

# Default — auto-discovers reader backend + LLM from env vars
tool = WebFetchTool()

# Explicit reader backend via injected WebFetchRawTool
from aweagent.core.tool.search import WebFetchRawTool
reader = WebFetchRawTool(backend="jina")
tool = WebFetchTool(reader=reader)

# Invoke
result = await tool.execute({
    "url": "https://docs.python.org/3/library/asyncio.html",
    "prompt": "How do you create a task that runs concurrently?",
})
```

### Manual Testing

```bash
export JINA_API_KEY=your_key_here
export WEB_FETCH_CONFIG_PATH=configs/llm/web_fetch/azure.yaml
python human_test/tools/search/test_web_fetch_tool.py
```

---

## Plugin Systems

Both search and reader backends use the same pattern: Registry + entry-points auto-discovery.

### Search Backend (`aweagent.search_backend`)

Location: `backends/search/`

**Built-in:** `serpapi` — Google search via [SerpAPI](https://serpapi.com/)

**Adding a custom search backend:**

```python
class MySearchBackend:
    async def search(self, query, *, num=10, start=0, engine="google"):
        ...
        return [{"position": 1, "title": "...", "url": "...", "description": "...", "snippets": [...]}]
```

```toml
# pyproject.toml
[project.entry-points."aweagent.search_backend"]
my_backend = "my_package.search:MySearchBackend"
```

### Reader Backend (`aweagent.reader_backend`)

Location: `backends/reader/`

**Built-in:** `jina` — URL content extraction via [Jina Reader](https://jina.ai/reader/)

**Adding a custom reader backend:**

```python
class MyReaderBackend:
    async def read_link(self, url: str) -> str:
        ...
        return "extracted content as markdown"
```

```toml
# pyproject.toml
[project.entry-points."aweagent.reader_backend"]
my_backend = "my_package.reader:MyReaderBackend"
```
