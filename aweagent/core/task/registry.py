"""Task registry — global registry for task discovery.

Mirrors ``agent_registry`` (``aweagent/scaffold/registry.py``): built-in tasks
are code-registered here so they work without ``pip install -e .``, and any
downstream package can add tasks via the ``aweagent.task`` entry-point group.
"""

from aweagent.plugins.registry import Registry

# Global task registry. Tasks register here and are discovered via entry_points.
task_registry: Registry[type] = Registry("aweagent.task")

# Built-in tasks (always available, even without pip install -e .)
from aweagent.tasks.beyond_swe.task import BeyondSWETask
from aweagent.tasks.scale_swe.task import ScaleSWETask
from aweagent.tasks.swe_bench_pro.task import SWEBenchProTask

task_registry.register("beyond_swe", BeyondSWETask)
task_registry.register("scale_swe", ScaleSWETask)
task_registry.register("swe_bench_pro", SWEBenchProTask)

# Lazy-register tasks with extra/optional dependencies (mirror terminus_2).
try:
    from aweagent.tasks.terminal_bench_v2.task import TerminalBenchV2Task

    task_registry.register("terminal_bench_v2", TerminalBenchV2Task)
except ImportError:
    pass

try:
    from aweagent.tasks.nl2repo.task import NL2RepoTask

    task_registry.register("nl2repo", NL2RepoTask)
except ImportError:
    pass

try:
    from aweagent.tasks.browsecomp.task import BrowseCompTask

    task_registry.register("browsecomp", BrowseCompTask)
except ImportError:
    pass

try:
    from aweagent.tasks.denovo_swe.task import DeNovoSWETask

    task_registry.register("denovo_swe", DeNovoSWETask)
except ImportError:
    pass
