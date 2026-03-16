"""Runtime Protocol — the interface all runtime backends must satisfy."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import AsyncIterator

from awe_agent.core.runtime.config import RuntimeConfig
from awe_agent.core.runtime.types import ExecutionResult

logger = logging.getLogger(__name__)

# ─── .gitignore rules for filtering patch output ────────────────────────────
# Shared across all runtime backends so that get_patch() excludes build
# artefacts regardless of whether the session runs on Docker or Portal.

_DEFAULT_GITIGNORE = [
    "*.jpg", "*.png", "*.jpeg", "*.o", "*.out", "*.obj", "*.so", "build", "Build",
]

_LANGUAGE_GITIGNORES: dict[str, list[str]] = {
    "c": ["bin/", "lib/", "*.dylib"],
    "cpp": ["bin/", "lib/", "*.dylib"],
    "java": ["target/", "out/", "*.class", "*.jar", ".gradle/"],
    "js": [
        "node_modules/", "dist/", ".next/", "coverage/", ".env",
        "npm-debug.log*", "yarn-debug.log*", "yarn-error.log*",
    ],
    "ts": [
        "node_modules/", "build/", "dist/", ".next/", "coverage/", ".env",
        "npm-debug.log*", "yarn-debug.log*", "yarn-error.log*",
        "*.js", "*.js.map", "*.d.ts", ".tsbuildinfo",
    ],
    "go": ["pkg/", "vendor/", "bin/", "*.test"],
    "rust": ["target/", "Cargo.lock", "*.rs.bk"],
    "python": [],
    "csharp": [
        "bin/", "obj/", "*.suo", "*.user", "*.userosscache",
        "*.sln.docstates", "*.vs/", "*.cache/", "*.pdb", "*.dll", "*.exe",
    ],
    "kotlin": ["build/", "out/", "*.class", "*.jar", ".gradle/", "buildSrc/build/", "*.kt.bak"],
    "php": ["vendor/", "composer.lock", "composer.phar", "*.log", "*.cache", "*.tmp", "*.swp", ".env", "phpunit.xml"],
    "ruby": ["*.gem", "Gemfile.lock", "vendor/", "log/", "tmp/", "*.bundle", "*.so", "*.o", "*.a", "mkmf.log"],
    "scala": ["target/", "project/target/", "project/project/", "*.class", "*.jar", ".sbt/", ".scala/", "*.log"],
    "swift": [".build/", "Packages/", "*.xcworkspace/", "*.xcuserstate", "*.xcprofdata", "DerivedData/", "*.swp", "*.swo", "*.log", "Pods/", "Podfile.lock"],
}

_LANGUAGE_ALIAS: dict[str, list[str]] = {
    "java": ["java"], "cpp": ["cpp", "c++"], "c": ["c"],
    "js": ["js", "javascript"], "ts": ["ts", "typescript"],
    "go": ["go", "golang"], "rust": ["rust"], "python": ["python"],
    "csharp": ["csharp", "c#", "cs"], "kotlin": ["kotlin"],
    "php": ["php"], "ruby": ["ruby"], "scala": ["scala"], "swift": ["swift"],
}

_GITIGNORE_START = "# === AWEAGENT AUTO-GENERATED START ==="
_GITIGNORE_END = "# === AWEAGENT AUTO-GENERATED END ==="


def _normalize_language(language: str) -> str:
    """Normalize language aliases (e.g., 'javascript' → 'js')."""
    language = language.lower().strip()
    for canonical, aliases in _LANGUAGE_ALIAS.items():
        if language in aliases:
            return canonical
    return language


class RuntimeSession(ABC):
    """Abstract session representing a single running environment.

    Provides command execution, file operations, and git helpers.
    All operations are async for high concurrency.
    """

    @abstractmethod
    async def execute(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute a shell command."""
        ...

    @abstractmethod
    async def upload_file(self, remote_path: str, content: bytes) -> None:
        """Upload a file to the container."""
        ...

    @abstractmethod
    async def download_file(self, remote_path: str) -> bytes:
        """Download a file from the container."""
        ...

    @abstractmethod
    async def list_files(self, path: str, recursive: bool = False) -> list[str]:
        """List files in a directory."""
        ...

    # ─── .gitignore management ────────────────────────────────────────

    async def _update_gitignore(self, cwd: str, language: str) -> None:
        """Inject language-specific .gitignore rules before git operations.

        Manages a clearly delimited block so the rules are idempotent:
        subsequent calls replace the block instead of duplicating it.
        Uses abstract ``download_file`` / ``upload_file`` so every backend
        inherits this behaviour automatically.
        """
        lang = _normalize_language(language)
        rules = _DEFAULT_GITIGNORE + _LANGUAGE_GITIGNORES.get(lang, [])
        block = "\n".join([_GITIGNORE_START] + rules + [_GITIGNORE_END])

        gitignore_path = f"{cwd}/.gitignore"

        # Read existing content (file may not exist)
        content = ""
        try:
            raw = await self.download_file(gitignore_path)
            content = raw.decode("utf-8", errors="replace")
        except (FileNotFoundError, Exception):
            pass

        # Idempotent update: replace existing block or append
        if _GITIGNORE_START in content and _GITIGNORE_END in content:
            start_idx = content.find(_GITIGNORE_START)
            end_idx = content.find(_GITIGNORE_END) + len(_GITIGNORE_END)
            new_content = content[:start_idx] + block + content[end_idx:]
        else:
            if content and not content.endswith("\n"):
                content += "\n"
            new_content = content + ("\n" if content else "") + block

        if new_content != content:
            await self.upload_file(gitignore_path, new_content.encode())

    # ─── Git helpers ──────────────────────────────────────────────────

    async def get_patch(
        self,
        cwd: str,
        base_commit: str | None = None,
        language: str = "python",
    ) -> str:
        """Get git diff as patch, including untracked new files.

        Updates ``.gitignore`` with language-specific rules before staging
        so that build artefacts are excluded from the patch.  Uses
        ``git add -A && git diff --cached`` to capture both modified
        tracked files and newly created files.
        """
        await self._update_gitignore(cwd, language)

        if base_commit:
            result = await self.execute(
                f"git add -A && git diff --cached {base_commit}", cwd=cwd,
            )
        else:
            result = await self.execute(
                "git add -A && git diff --cached", cwd=cwd,
            )
        return result.stdout

    async def apply_patch(self, cwd: str, patch: str) -> ExecutionResult:
        """Apply a patch. Tries 6 strategies for robustness."""
        await self.upload_file("/tmp/_awe_agent.patch", patch.encode())

        strategies = [
            ("git apply --verbose /tmp/_awe_agent.patch", False),
            ("git apply --verbose --ignore-space-change --ignore-whitespace /tmp/_awe_agent.patch", False),
            ("patch --batch --fuzz=5 -p1 -i /tmp/_awe_agent.patch", False),
            ("git apply --verbose --reject /tmp/_awe_agent.patch", True),
            ("git apply --verbose --reject --ignore-space-change --ignore-whitespace /tmp/_awe_agent.patch", True),
            ("git apply --verbose --reject --ignore-space-change --ignore-whitespace --allow-empty /tmp/_awe_agent.patch", True),
        ]

        last_result = None
        for cmd, is_reject in strategies:
            result = await self.execute(cmd, cwd=cwd)
            if result.success:
                return result
            if is_reject and result.exit_code == 1:
                # --reject partial success (rejected hunks written to .rej)
                return ExecutionResult(stdout=result.stdout, stderr=result.stderr, exit_code=0)
            last_result = result

        return last_result or ExecutionResult(stderr="All patch strategies failed", exit_code=1)

    @abstractmethod
    async def close(self) -> None:
        """Release all resources associated with this session."""
        ...

    async def __aenter__(self) -> RuntimeSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()


class Runtime(ABC):
    """Abstract runtime factory. Creates sessions for different backends."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    @abstractmethod
    async def create_session(
        self,
        image: str | None = None,
        **kwargs: object,
    ) -> RuntimeSession:
        """Create a new runtime session (container)."""
        ...

    @asynccontextmanager
    async def session(
        self,
        image: str | None = None,
        **kwargs: object,
    ) -> AsyncIterator[RuntimeSession]:
        """Context manager for runtime sessions. Ensures cleanup.

        Enforces ``config.timeout`` as a hard session TTL at the protocol
        level.  Individual backends (e.g. Docker) may enforce tighter
        per-operation timeouts internally; this acts as a final safety net.
        """
        sess = await self.create_session(image, **kwargs)
        try:
            timeout = self.config.timeout
            if timeout and timeout > 0:
                async with asyncio.timeout(timeout):
                    yield sess
            else:
                yield sess
        except TimeoutError:
            logger.error(
                "Session TTL expired after %ds — force-closing session",
                self.config.timeout,
            )
            raise
        finally:
            await sess.close()
