"""Task & Evaluation framework.

Usage:
    from aweagent.core.task import Task, Evaluator, TaskRunner, Instance, EvalResult
"""

from aweagent.core.task.protocol import Evaluator, Task
from aweagent.core.task.runner import TaskRunner, runtime_registry
from aweagent.core.task.types import EvalResult, Instance, TaskResult

__all__ = [
    "EvalResult",
    "Evaluator",
    "Instance",
    "Task",
    "TaskResult",
    "TaskRunner",
    "runtime_registry",
]
