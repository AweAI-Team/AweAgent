"""Generic Registry for plugin discovery and management.

Supports three registration methods:
1. Direct: registry.register("name", cls)
2. Decorator: @registry.decorator("name")
3. Entry points: auto-discovered via pyproject.toml [project.entry-points]
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    """Generic registry for managing named components.

    Example:
        llm_registry = Registry[type]("awe_agent.llm_backend")
        llm_registry.register("openai", OpenAIBackend)

        # Or via decorator
        @llm_registry.decorator("custom")
        class CustomBackend: ...

        # Retrieval (auto-discovers entry_points on miss)
        backend_cls = llm_registry.get("openai")
    """

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self._items: dict[str, T] = {}
        self._entry_points_loaded = False

    def register(self, name: str, item: T) -> None:
        """Register an item by name."""
        if name in self._items:
            logger.warning("Overriding '%s' in registry [%s]", name, self.namespace)
        self._items[name] = item

    def decorator(self, name: str):
        """Register via decorator."""

        def wrapper(cls: T) -> T:
            self.register(name, cls)
            return cls

        return wrapper

    def get(self, name: str) -> T:
        """Get an item by name. Auto-discovers entry_points on first miss."""
        if name not in self._items and not self._entry_points_loaded:
            self._discover_entry_points()
        if name not in self._items:
            available = ", ".join(sorted(self._items.keys())) or "(none)"
            raise KeyError(
                f"[{self.namespace}] '{name}' not found. Available: {available}"
            )
        return self._items[name]

    def list_available(self) -> list[str]:
        """List all registered names (triggers entry_points discovery)."""
        if not self._entry_points_loaded:
            self._discover_entry_points()
        return sorted(self._items.keys())

    def _discover_entry_points(self) -> None:
        """Load plugins from setuptools entry_points."""
        self._entry_points_loaded = True
        try:
            eps = entry_points(group=self.namespace)
        except TypeError:
            eps = entry_points().get(self.namespace, [])  # type: ignore[assignment]

        for ep in eps:
            if ep.name not in self._items:
                try:
                    self._items[ep.name] = ep.load()
                    logger.debug(
                        "Discovered plugin '%s' from entry_point [%s]",
                        ep.name,
                        self.namespace,
                    )
                except Exception:
                    logger.warning(
                        "Failed to load entry_point '%s' from [%s]",
                        ep.name,
                        self.namespace,
                        exc_info=True,
                    )

    def __contains__(self, name: str) -> bool:
        if name not in self._items and not self._entry_points_loaded:
            self._discover_entry_points()
        return name in self._items

    def __repr__(self) -> str:
        return f"Registry(namespace={self.namespace!r}, items={list(self._items.keys())})"
