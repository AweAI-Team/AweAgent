"""Search backend registry — auto-discovers backends via entry-points.

Usage::

    from awe_agent.core.tool.search.backends.search import get_search_backend

    # By name
    backend = get_search_backend("serpapi")

    # Auto-discover (env var SEARCH_BACKEND or first available)
    backend = get_search_backend()
"""

from __future__ import annotations

import logging
import os
from typing import Any

from awe_agent.plugins.registry import Registry

logger = logging.getLogger(__name__)

search_backend_registry: Registry[type] = Registry("awe_agent.search_backend")

# Register built-in backends
try:
    from awe_agent.core.tool.search.backends.search.serpapi import SerpAPIBackend

    search_backend_registry.register("serpapi", SerpAPIBackend)
except ImportError:
    pass  # aiohttp not installed — skip


def get_search_backend(name: str | None = None, **kwargs: Any) -> Any:
    """Get a search backend instance by name.

    Resolution order:
        1. Explicit ``name`` argument.
        2. ``SEARCH_BACKEND`` environment variable.
        3. First available backend in the registry.

    Returns:
        A search backend instance, or ``None`` if no backend is available.
    """
    name = name or os.environ.get("SEARCH_BACKEND")

    if name:
        try:
            cls = search_backend_registry.get(name)
            return cls(**kwargs)  # type: ignore[call-arg]
        except KeyError:
            logger.warning(
                "Search backend %r not found. Available: %s",
                name,
                search_backend_registry.list_available(),
            )
            return None

    # Auto-discover: try each available backend
    available = search_backend_registry.list_available()
    if not available:
        logger.info("No search backends registered. Search tools will be unavailable.")
        return None

    cls = search_backend_registry.get(available[0])
    logger.debug("Auto-selected search backend: %s", available[0])
    return cls(**kwargs)  # type: ignore[call-arg]
