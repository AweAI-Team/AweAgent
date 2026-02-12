"""Configuration loader — YAML + env var + CLI override system.

Priority: CLI args > Environment variables > YAML config > Defaults

Environment variables use the AWE_AGENT__ prefix with double underscore for nesting:
    AWE_AGENT__LLM__MODEL=gpt-4o  →  config.llm.model = "gpt-4o"
    AWE_AGENT__LLM__THINKING=true →  config.llm.thinking = True
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from awe_agent.core.config.schema import AweAgentConfig

logger = logging.getLogger(__name__)

_ENV_PREFIX = "AWE_AGENT__"


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AweAgentConfig:
    """Load AweAgent configuration.

    Args:
        config_path: Path to YAML config file. If None, uses only defaults + env vars.
        overrides: Dict of overrides (e.g., from CLI args).

    Returns:
        Fully resolved AweAgentConfig.
    """
    raw: dict[str, Any] = {}

    # 1. Load YAML
    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            # Handle !include directives
            raw = _resolve_includes(raw, path.parent)
            logger.info("Loaded config from %s", path)
        else:
            logger.warning("Config file not found: %s", path)

    # 2. Apply environment variable overrides
    env_overrides = _parse_env_overrides()
    if env_overrides:
        raw = _deep_merge(raw, env_overrides)
        logger.debug("Applied %d env overrides", len(env_overrides))

    # 3. Apply CLI overrides
    if overrides:
        raw = _deep_merge(raw, overrides)

    # 4. Resolve ${VAR} references in string values
    raw = _resolve_env_vars(raw)

    # 5. Validate with Pydantic
    return AweAgentConfig(**raw)


def _parse_env_overrides() -> dict[str, Any]:
    """Parse AWE_AGENT__* environment variables into nested dict."""
    result: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        # AWE_AGENT__LLM__MODEL → ["llm", "model"]
        parts = key[len(_ENV_PREFIX):].lower().split("__")
        # Set nested value
        current = result
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        # Type coercion
        current[parts[-1]] = _coerce_value(value)
    return result


def _coerce_value(value: str) -> Any:
    """Coerce string value to appropriate Python type."""
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False
    if value.lower() in ("null", "none", ""):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    # Check for list syntax: [a, b, c]
    if value.startswith("[") and value.endswith("]"):
        items = value[1:-1].split(",")
        return [item.strip().strip("'\"") for item in items if item.strip()]
    return value


def _resolve_env_vars(data: Any) -> Any:
    """Replace ${VAR} and ${VAR:-default} in string values with env var values."""
    if isinstance(data, str):
        def _replace(m: re.Match) -> str:
            var_expr = m.group(1)
            if ":-" in var_expr:
                var_name, default = var_expr.split(":-", 1)
                return os.environ.get(var_name, default)
            return os.environ.get(var_expr, m.group(0))
        return re.sub(r"\$\{([^}]+)\}", _replace, data)
    elif isinstance(data, dict):
        return {k: _resolve_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_resolve_env_vars(item) for item in data]
    return data


def _resolve_includes(data: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Resolve !include directives by loading referenced YAML files."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str) and value.startswith("!include "):
            include_path = base_dir / value[len("!include "):]
            if include_path.exists():
                with open(include_path) as f:
                    result[key] = yaml.safe_load(f) or {}
            else:
                logger.warning("Include file not found: %s", include_path)
                result[key] = value
        elif isinstance(value, dict):
            result[key] = _resolve_includes(value, base_dir)
        else:
            result[key] = value
    return result


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dicts. Override values take precedence."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
