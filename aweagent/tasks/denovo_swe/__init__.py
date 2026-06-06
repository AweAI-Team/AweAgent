"""DeNovoSWE task — build repositories from scratch with source-cleaned images."""

from __future__ import annotations

# NOTE: ``DeNovoSWETask`` / ``DeNovoSWEEvaluator`` are loaded lazily via
# module-level ``__getattr__`` (PEP 562).  Eagerly importing them at module
# load time triggers a circular import: when ``beyond_swe.task`` initialises
# it loads the scaffold's prompt aggregator, which loads
# ``denovo_swe.prompt.user`` — which in turn triggers this ``__init__.py``,
# which would then try to import ``DeNovoSWETask`` — whose parent class
# ``BeyondSWETask`` is still mid-initialisation.

__all__ = ["DeNovoSWEEvaluator", "DeNovoSWETask"]


def __getattr__(name: str):
    if name == "DeNovoSWETask":
        from aweagent.tasks.denovo_swe.task import DeNovoSWETask
        return DeNovoSWETask
    if name == "DeNovoSWEEvaluator":
        from aweagent.tasks.denovo_swe.evaluator import DeNovoSWEEvaluator
        return DeNovoSWEEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
