"""IterResearchAgent — long-horizon research agent with Markovian context.

A thin :class:`Agent` that owns the tool catalog and supplies the custom
:class:`~aweagent.scaffold.iter_research.loop.IterResearchLoop` via
``create_loop``. The loop (not ``step``) drives generation, because the
Markovian state reconstruction is incompatible with the stock append-only
``AgentLoop``.

Tools are referenced two ways: by their AweAgent name internally (``web_search``,
``web_fetch``, ``python_interpreter`` — used for ``ctx.get_tool`` and for
``agent.tools`` / ``tool_options`` in recipes, consistent with the rest of the
framework) and by their IterResearch *external* name in the prompt's tool
schema and the model's ``<tool_call>`` (``google_search``, ``Visit``,
``PythonInterpreter`` — what the model was trained to emit). The catalog maps
between them and adapts arguments where the AweAgent tool's contract differs
(notably ``Visit``: an array of URLs fans out over single-URL ``web_fetch``
calls whose results are concatenated).

IterResearch-specific knobs (sampling, format retries, observation truncation)
live under ``config.extra.iter_research`` so no global schema field is needed —
the scaffold ships faithful defaults and stays configurable.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aweagent.core.agent.context import AgentContext
from aweagent.core.agent.protocol import Agent
from aweagent.core.agent.trajectory import Action
from aweagent.core.tool.protocol import Tool
from aweagent.core.tool.public import PythonInterpreterTool
from aweagent.core.tool.registry import tool_registry
from aweagent.core.tool.search import (
    SearchConstraints,
    WebFetchRawTool,
    WebFetchTool,
    WebSearchTool,
)

if TYPE_CHECKING:
    from aweagent.core.agent.loop import AgentLoop
    from aweagent.core.config.schema import AweAgentConfig

logger = logging.getLogger(__name__)


# ── Canonical IterResearch (v19) tool schemas ──────────────────────────────
# These are the exact name/description/parameters the model was trained on.
# They populate the prompt's ``{tools}`` block; the ``name`` is the external
# name the model emits in ``<tool_call>``.

_GOOGLE_SEARCH_SCHEMA: dict[str, Any] = {
    "name": "google_search",
    "description": (
        "Perform Google web searches then returns a string of the top search "
        "results. Accepts multiple queries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string", "description": "The search query."},
                "minItems": 1,
                "description": "The list of search queries.",
            }
        },
        "required": ["query"],
    },
}

_VISIT_SCHEMA: dict[str, Any] = {
    "name": "Visit",
    "description": "Visit webpage(s) or paper(s) and return the summary of the content.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": (
                    "The URL(s) of the webpage(s) or paper(s) to visit. Can be a "
                    "single URL or an array of URLs."
                ),
            },
            "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s) or paper(s).",
            },
            "parse_type": {
                "type": "string",
                "enum": ["html", "pdf"],
                "default": "html",
                "description": (
                    "Specify whether to visit a HTML webpage or a PDF paper. Must "
                    "be either 'html' or 'pdf'. Defaults to 'html' if not specified."
                ),
            },
        },
        "required": ["url", "goal"],
    },
}

_PYTHON_INTERPRETER_SCHEMA: dict[str, Any] = {
    "name": "PythonInterpreter",
    "description": (
        "Executes arbitrary Python code in a secure, sandboxed environment. This "
        "tool is designed for performing complex calculations, data manipulations, "
        "string processing, logical operations, and general programming tasks that "
        "require programmatic logic. It can define variables, use built-in "
        "functions, and print results to standard output. It operates with a "
        "limited set of pre-installed libraries and cannot access local files, "
        "external networks, or system resources."
    ),
    "parameters": {
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
    },
}


# ── Tool catalog: maps an AweAgent tool to its IterResearch external identity ──


@dataclass(frozen=True)
class _ToolSpec:
    """How an AweAgent tool is presented to, and called from, the model."""

    external: str  # name the model emits in <tool_call>
    internal: str  # AweAgent tool name (ctx.get_tool)
    schema: dict[str, Any]  # entry in the prompt's {tools} block
    adapt: Callable[[dict[str, Any]], list[dict[str, Any]]]  # external args -> execute params


def _identity_adapt(args: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(args)]


def _visit_adapt(args: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapt a ``Visit`` call to one or more single-URL ``web_fetch`` calls.

    ``Visit`` takes a URL array + a ``goal``; ``web_fetch`` takes one ``url`` and
    a ``prompt``. We fan out over every URL (results are concatenated by the
    loop) and map ``goal`` -> ``prompt``. ``parse_type`` has no ``web_fetch``
    equivalent and is dropped.
    """
    urls = args.get("url")
    goal = (
        args.get("goal")
        or args.get("prompt")
        or "Extract all information relevant to the question."
    )
    if urls is None:
        return []
    if isinstance(urls, str):
        urls = [urls]
    return [
        {"url": u, "prompt": goal}
        for u in urls
        if isinstance(u, str) and u.strip()
    ]


# Keyed by AweAgent (internal) tool name.
_CATALOG: dict[str, _ToolSpec] = {
    "web_search": _ToolSpec("google_search", "web_search", _GOOGLE_SEARCH_SCHEMA, _identity_adapt),
    "web_fetch": _ToolSpec("Visit", "web_fetch", _VISIT_SCHEMA, _visit_adapt),
    "python_interpreter": _ToolSpec(
        "PythonInterpreter", "python_interpreter", _PYTHON_INTERPRETER_SCHEMA, _identity_adapt
    ),
}

_DEFAULT_TOOLS = ["web_search", "web_fetch", "python_interpreter"]
_ITER_RESEARCH_TOOLSETS = {"default": _DEFAULT_TOOLS}

# Faithful reproduction defaults (override via config.extra.iter_research).
_DEFAULT_SAMPLING: dict[str, Any] = {"temperature": 0.6, "top_p": 0.95, "presence_penalty": 1.5}
_DEFAULT_MAX_FORMAT_RETRIES = 5
_DEFAULT_OBSERVATION_MAX_TOKENS = 32000
_DEFAULT_TOKENIZER_ENCODING = "o200k_base"


def _resolve_tool_names(config: AweAgentConfig) -> list[str]:
    agent_config = config.agent
    if agent_config.tools is not None:
        return list(agent_config.tools)
    toolset = agent_config.toolset or "default"
    if toolset not in _ITER_RESEARCH_TOOLSETS:
        available = ", ".join(sorted(_ITER_RESEARCH_TOOLSETS))
        raise ValueError(
            f"Unknown IterResearch toolset {toolset!r}. Available: {available}"
        )
    return list(_ITER_RESEARCH_TOOLSETS[toolset])


def _build_tool(
    name: str,
    options: dict[str, Any],
    search_constraints: SearchConstraints | None,
) -> Tool:
    if name == "web_search":
        return WebSearchTool(
            engine=options.get("engine"),
            constraints=search_constraints,
            max_attempts=options.get("max_attempts", 3),
            result_scheme=options.get("result_scheme"),
            backend=options.get("backend"),
        )
    if name == "web_fetch":
        reader = None
        reader_backend = options.get("reader_backend")
        if reader_backend:
            reader = WebFetchRawTool(
                constraints=search_constraints,
                max_content_tokens=options.get("max_content_tokens", 100000),
                max_attempts=options.get("reader_max_attempts", 3),
                backend=reader_backend,
            )
        return WebFetchTool(
            constraints=search_constraints,
            max_content_tokens=options.get("max_content_tokens", 25000),
            max_attempts=options.get("max_attempts", 3),
            system_prompt=options.get("system_prompt"),
            llm_config_path=options.get("llm_config_path"),
            llm_params=options.get("llm_params"),
            reader=reader,
        )
    if name == "python_interpreter":
        return PythonInterpreterTool(
            endpoints=options.get("endpoints"),
            timeout=options.get("timeout", 50),
            run_timeout=options.get("run_timeout", 50),
            max_attempts=options.get("max_attempts", 8),
        )
    # Extensibility: any other tool resolved from the global registry.
    tool_cls = tool_registry.get(name)
    return tool_cls(**options)


def _iter_research_settings(config: AweAgentConfig) -> dict[str, Any]:
    raw = config.extra.get("iter_research", {}) if isinstance(config.extra, dict) else {}
    sampling = {**_DEFAULT_SAMPLING, **(raw.get("sampling") or {})}
    return {
        "sampling": sampling,
        "max_format_retries": int(raw.get("max_format_retries", _DEFAULT_MAX_FORMAT_RETRIES)),
        "observation_max_tokens": int(
            raw.get("observation_max_tokens", _DEFAULT_OBSERVATION_MAX_TOKENS)
        ),
        "tokenizer_encoding": str(raw.get("tokenizer_encoding", _DEFAULT_TOKENIZER_ENCODING)),
    }


def _constraints_from_config(config: AweAgentConfig) -> SearchConstraints | None:
    if config.security.blocked_search_patterns:
        return SearchConstraints(blocked_patterns=config.security.blocked_search_patterns)
    return None


def _warn_if_token_capped(config: AweAgentConfig) -> None:
    params = getattr(config.llm, "params", {}) or {}
    capped = [key for key in ("max_tokens", "max_completion_tokens") if params.get(key)]
    if capped:
        logger.warning(
            "IterResearch needs uncapped generation (a full <report> plus a long "
            "<answer> every turn), but llm.params sets %s. The report/answer may be "
            "truncated mid-tag and fail format validation. Remove the cap from the "
            "LLM config used for the IterResearch agent.",
            ", ".join(capped),
        )


class IterResearchAgent(Agent):
    """Markovian long-horizon research agent (paper: arXiv 2511.07327)."""

    @classmethod
    def from_config(cls, config: AweAgentConfig) -> IterResearchAgent:
        _warn_if_token_capped(config)
        return cls(
            search_constraints=_constraints_from_config(config),
            tool_names=_resolve_tool_names(config),
            tool_options=config.agent.tool_options,
            **_iter_research_settings(config),
        )

    @classmethod
    def from_config_with_constraints(
        cls,
        config: AweAgentConfig,
        instance_constraints: SearchConstraints,
    ) -> IterResearchAgent:
        _warn_if_token_capped(config)
        constraints = _constraints_from_config(config)
        constraints = (
            constraints.merge(instance_constraints) if constraints else instance_constraints
        )
        return cls(
            search_constraints=constraints,
            tool_names=_resolve_tool_names(config),
            tool_options=config.agent.tool_options,
            **_iter_research_settings(config),
        )

    def __init__(
        self,
        search_constraints: SearchConstraints | None = None,
        tool_names: list[str] | None = None,
        tool_options: dict[str, dict[str, Any]] | None = None,
        sampling: dict[str, Any] | None = None,
        max_format_retries: int = _DEFAULT_MAX_FORMAT_RETRIES,
        observation_max_tokens: int = _DEFAULT_OBSERVATION_MAX_TOKENS,
        tokenizer_encoding: str = _DEFAULT_TOKENIZER_ENCODING,
    ) -> None:
        options = tool_options or {}
        self._tools: list[Tool] = []
        self.tool_specs: dict[str, _ToolSpec] = {}
        schemas: list[dict[str, Any]] = []
        for name in tool_names or list(_DEFAULT_TOOLS):
            tool = _build_tool(name, options.get(name, {}), search_constraints)
            self._tools.append(tool)
            spec = _CATALOG.get(name)
            if spec is None:
                # Custom (registry) tool: advertise it under its own AweAgent name.
                spec = _ToolSpec(
                    external=tool.name,
                    internal=tool.name,
                    schema={
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                    adapt=_identity_adapt,
                )
            self.tool_specs[spec.external] = spec
            schemas.append(spec.schema)

        # The {tools} block, exactly as the reference builds it (json.dumps list).
        self.tool_str = json.dumps(schemas)
        self.sampling = {**_DEFAULT_SAMPLING, **(sampling or {})}
        self.max_format_retries = max(int(max_format_retries), 1)
        self.observation_max_tokens = max(int(observation_max_tokens), 1)
        self.tokenizer_encoding = tokenizer_encoding

    def get_system_prompt(self, task_info: dict[str, Any]) -> str:
        # IterResearch carries all instruction in the per-turn user message that
        # the loop reconstructs; there is no separate system prompt.
        return ""

    def get_tools(self) -> list[Tool]:
        return list(self._tools)

    def create_loop(self, context: AgentContext) -> AgentLoop:
        from aweagent.scaffold.iter_research.loop import IterResearchLoop

        return IterResearchLoop(self, context)  # type: ignore[return-value]

    async def step(self, context: AgentContext) -> Action:
        raise NotImplementedError(
            "IterResearchAgent drives generation through its custom "
            "IterResearchLoop (see create_loop); the stock AgentLoop.step path is "
            "not used."
        )
