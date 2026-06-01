"""BrowseComp task mapping for DeepSearch."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from awe_agent.core.llm.config import LLMConfig
from awe_agent.core.task.protocol import Evaluator, Task
from awe_agent.core.task.types import Instance
from awe_agent.core.tool.search.constraints import SearchConstraints


_BROWSECOMP_ANTI_HACK_BLOCKED_PATTERNS: dict[str, list[str]] = {
    "url": [
        r".*://([^/]+\.)?huggingface\.co(/|$).*",
        r".*://([^/]+\.)?openreview\.net(/|$).*",
        r".*://([^/]+\.)?github\.com(/|$).*",
        r".*://([^/]+\.)?raw\.githubusercontent\.com(/|$).*",
        r".*huggingface\.co/datasets/.*",
        r".*huggingface\.co/api/datasets/.*",
        r".*datasets-server\.huggingface\.co/.*",
        r".*openreview\.net/attachment.*",
        r".*openreview\.net/forum\?id=UJFCyrYM1V.*",
        r".*researchgate\.net/publication/404627419_.*",
        r".*arxiv\.org/(abs|pdf)/(2601\.06021|2603\.20432).*",
    ],
    "title": [
        r".*BrowseComp.*",
        r".*BrowseComp-Plus.*",
        r".*bcp-(traj|plan).*",
        r".*benchmark-bcplus.*",
        r".*webgym_tasks.*",
        r".*Improving and Evaluating Open Deep Research Agents.*",
        r".*Coding Agents are Effective Long-Context Processors.*",
    ],
    "description": [
        r".*BrowseComp.*",
        r".*BrowseComp-Plus.*",
        r".*bcp-(traj|plan).*",
        r".*reference_answer.*",
        r".*ground truth.*",
        r".*Correct Answer:.*",
        r".*Answer sketch:.*",
        r".*timchen0618/.*",
        r".*rl-rag/browsecomp.*",
        r".*Nithish2410/benchmark-bcplus.*",
        r".*microsoft/webgym_tasks.*",
    ],
    "snippets": [
        r".*BrowseComp.*",
        r".*BrowseComp-Plus.*",
        r".*bcp-(traj|plan).*",
        r".*reference_answer.*",
        r".*ground truth.*",
        r".*Correct Answer:.*",
        r".*Answer sketch:.*",
        r".*timchen0618/.*",
        r".*rl-rag/browsecomp.*",
        r".*Nithish2410/benchmark-bcplus.*",
        r".*microsoft/webgym_tasks.*",
    ],
}


def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


class BrowseCompTask(Task):
    """Task implementation for BrowseComp-style text QA datasets."""

    def __init__(
        self,
        dataset_id: str = "browsecomp",
        data_file: str | None = None,
        grader_llm_config: LLMConfig | None = None,
    ) -> None:
        if not data_file:
            raise ValueError("data_file is required for browsecomp tasks")
        self.dataset_id = dataset_id
        self.data_file = Path(data_file)
        self._grader_llm_config = grader_llm_config
        self._instances: list[Instance] | None = None

    def get_instances(self, instance_ids: list[str] | None = None) -> list[Instance]:
        if self._instances is None:
            self._instances = self._load_instances()
        if not instance_ids:
            return list(self._instances)
        wanted = {str(instance_id) for instance_id in instance_ids}
        return [instance for instance in self._instances if instance.id in wanted]

    def get_prompt(self, instance: Instance) -> str:
        return str(instance.metadata["question"])

    def requires_git_snapshot(self) -> bool:
        return False

    def requires_patch_extraction(self) -> bool:
        return False

    def requires_agent_result_evaluation(self) -> bool:
        return True

    def get_search_constraints(self, instance: Instance) -> SearchConstraints:
        # BrowseComp answers are now mirrored in public benchmark traces.
        # Block those leak sources by default while keeping this task-specific.
        return SearchConstraints(
            blocked_patterns={
                field: list(patterns)
                for field, patterns in _BROWSECOMP_ANTI_HACK_BLOCKED_PATTERNS.items()
            }
        )

    def get_task_info(self, instance: Instance) -> dict[str, Any]:
        return {
            "instance_id": instance.id,
            "dataset_id": instance.dataset_id,
            "workdir": instance.workdir,
            "language": instance.language,
            "task_type": "browsecomp",
        }

    def default_evaluator(self, timeout: int | None = None) -> Evaluator:
        from awe_agent.tasks.browsecomp.evaluator import BrowseCompEvaluator

        kwargs: dict[str, Any] = {"grader_llm_config": self._grader_llm_config}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return BrowseCompEvaluator(**kwargs)

    def _load_instances(self) -> list[Instance]:
        records = _load_records(self.data_file)
        instances = [
            self._record_to_instance(record, index)
            for index, record in enumerate(records)
        ]
        if not instances:
            raise ValueError(f"No BrowseComp instances found in {self.data_file}")
        return instances

    def _record_to_instance(self, record: dict[str, Any], index: int) -> Instance:
        normalized = _normalize_record(record, index)
        question = normalized["question"]
        answer = normalized["answer"]
        if not question or not answer:
            raise ValueError(
                f"BrowseComp record {index} must contain question/problem and answer"
            )

        metadata = {
            "task_type": "browsecomp",
            "question": question,
            "answer": answer,
            "topic": normalized.get("topic", ""),
        }
        if normalized.get("canary"):
            metadata["canary"] = normalized["canary"]

        return Instance(
            id=normalized["id"],
            dataset_id=self.dataset_id,
            workdir="/testbed",
            language="text",
            metadata=metadata,
            problem_statement=question,
        )


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"BrowseComp data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json(path)
    if suffix == ".jsonl":
        return _load_jsonl(path)
    if suffix == ".csv":
        return _load_csv(path)
    if suffix in {".parquet", ".pq"}:
        return _load_parquet(path)
    raise ValueError(
        f"Unsupported BrowseComp data file suffix {suffix!r}; "
        "expected .json, .jsonl, .csv, or .parquet"
    )


def _load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [_ensure_record(item) for item in data]
    if isinstance(data, dict):
        for key in ("data", "examples", "instances"):
            value = data.get(key)
            if isinstance(value, list):
                return [_ensure_record(item) for item in value]
        return [_ensure_record(item) for item in data.values()]
    raise ValueError(f"Unsupported JSON BrowseComp data shape in {path}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(_ensure_record(json.loads(line)))
    return records


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Reading BrowseComp parquet files requires pandas/pyarrow"
        ) from exc

    return [_ensure_record(row) for row in pd.read_parquet(path).to_dict("records")]


def _ensure_record(value: Any) -> dict[str, Any]:
    value = _to_builtin(value)
    if not isinstance(value, dict):
        raise ValueError(f"Expected BrowseComp record dict, got {type(value).__name__}")
    return value


def _normalize_record(record: dict[str, Any], index: int) -> dict[str, str]:
    record = _to_builtin(record)
    extra = _mapping(record.get("extra_info"))
    reward_model = _mapping(record.get("reward_model"))
    canary = _scalar(record.get("canary") or extra.get("canary"))

    question = _scalar(
        extra.get("question")
        or record.get("question")
        or record.get("problem")
        or _prompt_question(record.get("prompt"))
    )
    answer = _scalar(
        extra.get("answer")
        or reward_model.get("ground_truth")
        or record.get("answer")
        or record.get("correct_answer")
    )

    if canary:
        question = _maybe_decrypt(question, canary)
        answer = _maybe_decrypt(answer, canary)

    instance_id = _scalar(extra.get("id") or record.get("id") or f"browsecomp_{index}")
    topic = _scalar(
        extra.get("problem_topic")
        or extra.get("topic")
        or record.get("problem_topic")
        or record.get("topic")
    )
    return {
        "id": str(instance_id),
        "question": question,
        "answer": answer,
        "topic": topic,
        "canary": canary,
    }


def _maybe_decrypt(value: str, canary: str) -> str:
    if not value:
        return value
    try:
        return decrypt(value, canary)
    except Exception:
        return value


def _prompt_question(prompt: Any) -> str:
    prompt = _to_builtin(prompt)
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                return _scalar(message.get("content"))
        if prompt and isinstance(prompt[0], dict):
            return _scalar(prompt[0].get("content"))
    return ""


def _mapping(value: Any) -> dict[str, Any]:
    value = _to_builtin(value)
    if isinstance(value, dict):
        return value
    return {}


def _scalar(value: Any) -> str:
    value = _to_builtin(value)
    if value is None:
        return ""
    if isinstance(value, list | tuple):
        return _scalar(value[0]) if value else ""
    if isinstance(value, dict):
        return ""
    return str(value)


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: _to_builtin(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [_to_builtin(item) for item in value]
    return value
