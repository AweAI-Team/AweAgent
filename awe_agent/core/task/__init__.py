"""Task & Evaluation framework.

Usage:
    from awe_agent.core.task import Task, Evaluator, TaskRunner, Instance, EvalResult
"""

from awe_agent.core.task.protocol import Evaluator, Task
from awe_agent.core.task.runner import TaskRunner, runtime_registry
from awe_agent.core.task.tb2_batch_runner import TB2BatchRunner
from awe_agent.core.task.types import EvalResult, Instance, TaskResult

__all__ = [
    "EvalResult",
    "Evaluator",
    "Instance",
    "Task",
    "TaskResult",
    "TaskRunner",
    "TB2BatchRunner",
    "runtime_registry",
]
