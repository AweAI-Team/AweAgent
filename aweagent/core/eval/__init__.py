"""Evaluation framework — isolated evaluation with pluggable evaluators.

Provides:
- :class:`Evaluator` ABC (via ``core.task.protocol``).
- :class:`IsolatedEvaluator` — generic script-based evaluation in fresh containers.
- :class:`PatchTestEvaluator` — template base for patch-and-test evaluators.
- :data:`evaluator_registry` — auto-discovering registry for evaluator backends.

Task-specific evaluators (``BeyondSWEEvaluator``) are registered via
``pyproject.toml`` entry-points and discovered lazily on first registry access.
They can also be imported directly::

    from aweagent.tasks.beyond_swe import BeyondSWEEvaluator
"""

from aweagent.core.eval.base import PatchTestEvaluator
from aweagent.core.eval.isolation import IsolatedEvaluator
from aweagent.plugins.registry import Registry

# Global evaluator registry — backends register via code or entry_points.
# Task-specific evaluators are discovered lazily through entry_points
# (see pyproject.toml [project.entry-points."aweagent.evaluator"]).
evaluator_registry: Registry[type] = Registry("aweagent.evaluator")

# Built-in core evaluators (always available)
evaluator_registry.register("isolated", IsolatedEvaluator)
evaluator_registry.register("patch_test", PatchTestEvaluator)

__all__ = [
    "IsolatedEvaluator",
    "PatchTestEvaluator",
    "evaluator_registry",
]
