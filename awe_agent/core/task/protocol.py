"""Task & Evaluator Protocols — interfaces for defining tasks and evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from awe_agent.core.runtime.protocol import Runtime
from awe_agent.core.task.types import EvalResult, Instance
from awe_agent.core.tool.search.constraints import SearchConstraints


class Task(ABC):
    """Abstract task definition.

    A Task knows how to:
    1. Load instances (problems to solve)
    2. Generate prompts for each instance
    3. Provide the correct docker image
    4. Provide setup commands for the environment
    5. Provide a default evaluator (override in subclasses)
    """

    @abstractmethod
    def get_instances(self, instance_ids: list[str] | None = None) -> list[Instance]:
        """Load task instances, optionally filtered by IDs."""
        ...

    @abstractmethod
    def get_prompt(self, instance: Instance) -> str:
        """Generate the task prompt for an instance."""
        ...

    def get_image(self, instance: Instance) -> str:
        """Get the docker image for an instance. Override for per-instance images."""
        return instance.image

    def get_setup_commands(self, instance: Instance) -> list[str]:
        """Get setup commands to run before the agent starts."""
        commands = list(instance.setup_commands)
        if instance.base_commit:
            commands.insert(0, f"cd {instance.workdir} && git checkout {instance.base_commit}")
        return commands

    def get_task_info(self, instance: Instance) -> dict[str, Any]:
        """Get task info dict to pass to the agent context."""
        return {
            "instance_id": instance.id,
            "dataset_id": instance.dataset_id,
            "repo": instance.repo,
            "base_commit": instance.base_commit,
            "workdir": instance.workdir,
            "language": instance.language,
        }

    def get_search_constraints(self, instance: Instance) -> SearchConstraints | None:
        """Build search constraints for this instance.

        By default, generates constraints from the repo name (if available).
        Override in subclasses for custom constraint logic.
        """
        if instance.repo:
            return SearchConstraints.from_repo(instance.repo)
        return None

    def default_evaluator(self, timeout: int | None = None) -> Evaluator | None:
        """Return the default evaluator for this task type.

        Subclasses should override this to provide a task-specific evaluator.
        The ``TaskRunner`` uses this as a fallback when no evaluator is
        explicitly configured.

        Args:
            timeout: Evaluation timeout in seconds.  Passed through to the
                evaluator constructor so that ``config.eval.timeout`` is
                respected.

        Returns ``None`` to indicate no default evaluator is available.
        """
        return None


class Evaluator(ABC):
    """Abstract evaluator.

    Evaluates an agent's submission (patch) against the ground truth.
    Key design: evaluation runs in an ISOLATED runtime to prevent leakage.
    """

    @abstractmethod
    async def evaluate(
        self,
        instance: Instance,
        patch: str,
        runtime: Runtime,
    ) -> EvalResult:
        """Evaluate a patch in an isolated environment.

        Args:
            instance: The task instance.
            patch: The agent's submission (git diff).
            runtime: An isolated runtime for evaluation (NOT the agent's runtime).

        Returns:
            EvalResult with accepted/score/details.
        """
        ...
