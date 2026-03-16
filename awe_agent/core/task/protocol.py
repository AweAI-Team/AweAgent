"""Task & Evaluator Protocols — interfaces for defining tasks and evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from awe_agent.core.runtime.protocol import Runtime
from awe_agent.core.task.types import EvalResult, Instance
from awe_agent.core.tool.search.constraints import SearchConstraints

if TYPE_CHECKING:
    from awe_agent.core.runtime.protocol import RuntimeSession


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
            commands.insert(0, f"git checkout {instance.base_commit}")
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

    def get_llm_overrides(self, instance: Instance) -> dict[str, Any]:
        """Return per-instance LLM parameter overrides.

        These are merged into ``LLMConfig.params`` for this instance,
        allowing task types to customize e.g. ``max_completion_tokens``.

        Default: no overrides (empty dict).
        """
        return {}

    async def prepare_session(
        self,
        instance: Instance,
        session: RuntimeSession,
    ) -> None:
        """Task-specific session preparation after setup commands, before prompt.

        Called by the runner after ``get_setup_commands()`` and ``PreAgentSetup``
        have run, but before ``get_prompt()`` is called.  Use this to upload
        files, run commands, and populate ``instance.metadata`` with data
        that the prompt template needs (e.g. ``installed_packages``).

        Default implementation is a no-op.
        """

    def requires_git_snapshot(self) -> bool:
        """Whether a git snapshot should be taken before the agent runs.

        Returns ``True`` by default (SWE-bench style tasks where the
        patch is extracted via ``git diff``).  Override to ``False`` for
        tasks that do not use a git repository in the workdir
        (e.g. Terminal Bench).
        """
        return True

    def get_search_constraints(self, instance: Instance) -> SearchConstraints | None:
        """Build search constraints for this instance.

        By default, generates constraints from the repo name (if available).
        Override in subclasses for custom constraint logic.
        """
        if instance.repo:
            return SearchConstraints.from_repo(instance.repo)
        return None

    def get_resource_limits(self, instance: Instance) -> dict[str, str] | None:
        """Return per-instance resource limits for the runtime.

        Expected dict keys: ``"cpu"`` and ``"memory"`` (e.g. ``"2048Mi"``).
        Return ``None`` (default) when all instances share the same limits
        defined in the global runtime config.
        """
        return None

    def get_docker_environment(self, instance: Instance) -> dict[str, str] | None:
        """Return per-instance Docker environment variables.

        Return ``None`` (default) when no extra env vars are needed.
        """
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

    @property
    def requires_same_session(self) -> bool:
        """Whether evaluation must run in the agent's session.

        When ``True``, the runner evaluates inside the same container
        the agent used (e.g. Terminal Bench, where the agent modifies
        container state directly and there is no patch to apply).

        Default is ``False`` (isolated evaluation).
        """
        return False

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
