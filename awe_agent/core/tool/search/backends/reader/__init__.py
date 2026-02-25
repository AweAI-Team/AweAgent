"""Reader backend registry — auto-discovers reader backends via entry-points.

Usage::

    from awe_agent.core.tool.search.backends.reader import get_reader_backend

    # By name
    backend = get_reader_backend("jina")

    # Auto-discover (env var READER_BACKEND or first available)
    backend = get_reader_backend()
"""

from __future__ import annotations

import logging
import os
from typing import Any

from awe_agent.plugins.registry import Registry

logger = logging.getLogger(__name__)

reader_backend_registry: Registry[type] = Registry("awe_agent.reader_backend")

# Register built-in backends
try:
    from awe_agent.core.tool.search.backends.reader.jina import JinaReaderBackend

    reader_backend_registry.register("jina", JinaReaderBackend)
except ImportError:
    pass  # aiohttp not installed — skip


def get_reader_backend(name: str | None = None, **kwargs: Any) -> Any:
    """Get a reader backend instance by name.

    Resolution order:
        1. Explicit ``name`` argument.
        2. ``READER_BACKEND`` environment variable.
        3. First available backend in the registry.

    Returns:
        A reader backend instance, or ``None`` if no backend is available.
    """
    name = name or os.environ.get("READER_BACKEND")

    if name:
        try:
            cls = reader_backend_registry.get(name)
            return cls(**kwargs)  # type: ignore[call-arg]
        except KeyError:
            logger.warning(
                "Reader backend %r not found. Available: %s",
                name,
                reader_backend_registry.list_available(),
            )
            return None

    # Auto-discover: try each available backend
    available = reader_backend_registry.list_available()
    if not available:
        logger.info("No reader backends registered. Link reader tools will be unavailable.")
        return None

    cls = reader_backend_registry.get(available[0])
    logger.debug("Auto-selected reader backend: %s", available[0])
    return cls(**kwargs)  # type: ignore[call-arg]
