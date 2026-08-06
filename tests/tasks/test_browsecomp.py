# ruff: noqa: E501
from __future__ import annotations

import base64
import json
import logging
import subprocess
import sys
from typing import Any

import pytest

from aweagent.core.agent.context import AgentContext
from aweagent.core.agent.loop import AgentResult
from aweagent.core.agent.protocol import Agent
from aweagent.core.agent.trajectory import Action, Trajectory
from aweagent.core.config.schema import AweAgentConfig
from aweagent.core.llm.client import llm_registry
from aweagent.core.llm.config import LLMConfig
from aweagent.core.task.pipeline import build_task
from aweagent.core.llm.types import LLMResponse
from aweagent.core.runtime.config import RuntimeConfig
from aweagent.core.task.protocol import Task
from aweagent.core.task.runner import (
    TaskRunner,
    _build_trajectory_record,
    runtime_registry,
    select_instances,
)
from aweagent.core.task.types import Instance, TaskResult
from aweagent.core.tool.protocol import Tool
from aweagent.tasks.browsecomp.evaluator import (
    GRADER_TEMPLATE,
    BrowseCompEvaluator,
    _parse_judge_correct,
)
from aweagent.tasks.browsecomp.task import (
    BrowseCompTask,
    _is_plausible_text,
    _maybe_decrypt,
    derive_key,
)

REFERENCE_GRADER_TEMPLATE = r"""
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def _encrypt(plaintext: str, password: str) -> str:
    data = plaintext.encode()
    key = derive_key(password, len(data))
    return base64.b64encode(bytes(a ^ b for a, b in zip(data, key))).decode()


def test_browsecomp_loads_decrypted_json(tmp_path):
    data_file = tmp_path / "browsecomp.json"
    data_file.write_text(json.dumps([
        {
            "id": 1,
            "problem": "Question?",
            "answer": "Answer",
            "problem_topic": "Art",
        }
    ]))

    task = BrowseCompTask(data_file=str(data_file))
    instances = task.get_instances()

    assert len(instances) == 1
    assert instances[0].id == "1"
    assert instances[0].metadata["question"] == "Question?"
    assert instances[0].metadata["answer"] == "Answer"
    assert instances[0].metadata["topic"] == "Art"
    assert task.get_prompt(instances[0]) == "Question?"


def test_browsecomp_loads_encrypted_csv(tmp_path):
    canary = "secret"
    data_file = tmp_path / "browsecomp.csv"
    data_file.write_text(
        "id,problem,answer,canary,problem_topic\n"
        f"row-1,{_encrypt('Encrypted question?', canary)},"
        f"{_encrypt('Encrypted answer', canary)},{canary},History\n"
    )

    instance = BrowseCompTask(data_file=str(data_file)).get_instances()[0]

    assert instance.id == "row-1"
    assert instance.metadata["question"] == "Encrypted question?"
    assert instance.metadata["answer"] == "Encrypted answer"


def test_browsecomp_loads_sft_json(tmp_path):
    data_file = tmp_path / "browsecomp_sft.json"
    data_file.write_text(json.dumps([
        {
            "prompt": [{"role": "user", "content": "SFT question?"}],
            "reward_model": {"ground_truth": ["SFT answer"]},
            "extra_info": {"id": "test_1", "split": "test"},
        }
    ]))

    instance = BrowseCompTask(data_file=str(data_file)).get_instances()[0]

    assert instance.id == "test_1"
    assert instance.metadata["question"] == "SFT question?"
    assert instance.metadata["answer"] == "SFT answer"


def test_browsecomp_loads_sft_parquet(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    data_file = tmp_path / "browsecomp.parquet"
    pd.DataFrame([
        {
            "prompt": [{"role": "user", "content": "Parquet question?"}],
            "reward_model": {"ground_truth": ["Parquet answer"]},
            "extra_info": {
                "answer": ["Parquet answer"],
                "id": "train_0",
                "question": "Parquet question?",
            },
        }
    ]).to_parquet(data_file)

    instance = BrowseCompTask(data_file=str(data_file)).get_instances()[0]

    assert instance.id == "train_0"
    assert instance.metadata["question"] == "Parquet question?"
    assert instance.metadata["answer"] == "Parquet answer"


def test_browsecomp_blocks_known_benchmark_leak_sources(tmp_path):
    data_file = tmp_path / "browsecomp.json"
    data_file.write_text(json.dumps([
        {"id": "bc", "problem": "Question?", "answer": "Answer"}
    ]))
    task = BrowseCompTask(data_file=str(data_file))
    constraints = task.get_search_constraints(task.get_instances()[0])

    assert constraints.is_url_blocked(
        "https://huggingface.co/datasets/rl-rag/browsecomp-high-effort-gpt-oss-120b"
    )
    assert constraints.is_url_blocked("https://huggingface.co/Qwen/Qwen3-235B")
    assert constraints.is_url_blocked(
        "https://datasets-server.huggingface.co/rows?dataset=timchen0618%2Fbcp-traj-ext-formatted-v1"
    )
    assert constraints.is_url_blocked(
        "https://openreview.net/forum?id=ordinary-paper"
    )
    assert constraints.is_url_blocked("https://github.com/example/repo")
    assert constraints.is_url_blocked(
        "https://raw.githubusercontent.com/openai/grade-school-math/master/data/train.jsonl"
    )
    assert constraints.is_url_blocked("https://arxiv.org/html/2504.12516v1")
    assert constraints.is_url_blocked("https://arxiv.org/pdf/2603.20278")
    assert constraints.is_url_blocked(
        "https://infosecwriteups.com/minrlm-a-token-efficient-recursive-language-model-implementation-and-benchmark-bdc6840a3b00"
    )
    assert constraints.is_url_blocked(
        "https://softwaredoug.com/blog/2026/03/23/we-can-learn-from-this-late-interaction-win"
    )
    assert constraints.is_url_blocked(
        "https://www.linkedin.com/posts/example_i-came-across-an-evaluation-test-called-browsecomp"
    )
    assert constraints.is_url_blocked("https://openreward.ai/GeneralReasoning/ToolMind-Web-QA")
    assert constraints.is_url_blocked("https://www.modelscope.cn/datasets/nanbeige/ToolMind-Web-QA")
    assert constraints.is_url_blocked("http://www.hfdailybriefer.com/")
    assert constraints.is_url_blocked("https://img1.wsimg.com/blobby/go/example/OpenResearcher.pdf")
    assert constraints.is_url_blocked(
        "https://www.researchgate.net/publication/404627419_Beyond_Semantic_Similarity_Rethinking_Retrieval_for_Agentic_Search_via_Direct_Corpus_Interaction"
    )
    assert constraints.is_url_blocked(
        "https://www.researchgate.net/publication/ordinary_paper"
    )
    assert constraints.is_url_blocked(
        "https://www.salesforce.com/blog/poisoning-the-well-search-agents/"
    )
    assert constraints.is_url_blocked("https://arxiv.org/pdf/2603.20432.pdf")
    assert constraints.is_url_blocked(
        "https://openreview.net/attachment?id=N0d6tG377V&name=supplementary_material"
    )
    assert not constraints.is_url_blocked("https://www.britannica.com/topic/example")

    filtered, count = constraints.filter_search_results([
        {
            "title": "rl-rag/browsecomp-high-effort-gpt-oss-120b",
            "url": "https://huggingface.co/datasets/rl-rag/browsecomp-high-effort-gpt-oss-120b",
            "description": "reference_answer is visible",
            "snippets": "",
        },
        {
            "title": "Open primary source",
            "url": "https://www.britannica.com/topic/example",
            "description": "ordinary result",
            "snippets": "ordinary result",
        },
        {
            "title": "Puzzle supplement",
            "url": "https://example.com/supplement",
            "description": "Correct Answer: Steve Falat",
            "snippets": "",
        },
    ])

    assert count == 2
    assert filtered == [
        {
            "title": "Open primary source",
            "url": "https://www.britannica.com/topic/example",
            "description": "ordinary result",
            "snippets": "ordinary result",
        }
    ]


def test_browsecomp_blocks_only_configured_sites(tmp_path, monkeypatch):
    monkeypatch.setenv("BROWSECOMP_BLOCK_SITES", "huggingface")
    data_file = tmp_path / "browsecomp.json"
    data_file.write_text(json.dumps([
        {"id": "bc", "problem": "Question?", "answer": "Answer"}
    ]))
    task = BrowseCompTask(data_file=str(data_file))
    constraints = task.get_search_constraints(task.get_instances()[0])

    assert constraints.is_url_blocked(
        "https://huggingface.co/datasets/rl-rag/browsecomp-high-effort-gpt-oss-120b"
    )
    assert constraints.is_url_blocked("https://huggingface.co/Qwen/Qwen3-235B")
    assert constraints.is_url_blocked(
        "https://datasets-server.huggingface.co/rows?dataset=timchen0618%2Fbcp-traj-ext-formatted-v1"
    )
    assert not constraints.is_url_blocked(
        "https://openreview.net/forum?id=ordinary-paper"
    )
    assert not constraints.is_url_blocked("https://github.com/example/repo")
    assert not constraints.is_url_blocked(
        "https://raw.githubusercontent.com/openai/grade-school-math/master/data/train.jsonl"
    )
    assert not constraints.is_url_blocked("https://arxiv.org/html/2504.12516v1")
    assert not constraints.is_url_blocked(
        "https://infosecwriteups.com/minrlm-a-token-efficient-recursive-language-model-implementation-and-benchmark-bdc6840a3b00"
    )
    assert not constraints.is_url_blocked(
        "https://softwaredoug.com/blog/2026/03/23/we-can-learn-from-this-late-interaction-win"
    )
    assert not constraints.is_url_blocked(
        "https://www.linkedin.com/posts/example_i-came-across-an-evaluation-test-called-browsecomp"
    )
    assert not constraints.is_url_blocked("https://openreward.ai/GeneralReasoning/ToolMind-Web-QA")
    assert not constraints.is_url_blocked("https://www.modelscope.cn/datasets/nanbeige/ToolMind-Web-QA")
    assert not constraints.is_url_blocked("http://www.hfdailybriefer.com/")
    assert not constraints.is_url_blocked("https://img1.wsimg.com/blobby/go/example/OpenResearcher.pdf")
    assert not constraints.is_url_blocked(
        "https://www.researchgate.net/publication/ordinary_paper"
    )
    assert not constraints.is_url_blocked(
        "https://www.salesforce.com/blog/poisoning-the-well-search-agents/"
    )

    filtered, count = constraints.filter_search_results([
        {
            "title": "Open source",
            "url": "https://github.com/example/repo",
            "description": "ordinary result",
            "snippets": "ordinary result",
        },
        {
            "title": "Open primary source",
            "url": "https://www.britannica.com/topic/example",
            "description": "Correct Answer: Steve Falat",
            "snippets": "",
        },
    ])

    assert count == 1
    assert filtered == [
        {
            "title": "Open source",
            "url": "https://github.com/example/repo",
            "description": "ordinary result",
            "snippets": "ordinary result",
        }
    ]


def test_browsecomp_blocks_huggingface_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("BROWSECOMP_BLOCK_SITES", "HF")
    data_file = tmp_path / "browsecomp.json"
    data_file.write_text(json.dumps([
        {"id": "bc", "problem": "Question?", "answer": "Answer"}
    ]))
    task = BrowseCompTask(data_file=str(data_file))
    constraints = task.get_search_constraints(task.get_instances()[0])

    assert constraints.is_url_blocked("https://huggingface.co/Qwen/Qwen3-235B")
    assert not constraints.is_url_blocked("https://github.com/example/repo")


def test_browsecomp_rejects_unknown_block_site(tmp_path, monkeypatch):
    monkeypatch.setenv("BROWSECOMP_BLOCK_SITES", "gitub")
    data_file = tmp_path / "browsecomp.json"
    data_file.write_text(json.dumps([
        {"id": "bc", "problem": "Question?", "answer": "Answer"}
    ]))
    task = BrowseCompTask(data_file=str(data_file))

    with pytest.raises(ValueError, match="Unknown BROWSECOMP_BLOCK_SITES"):
        task.get_search_constraints(task.get_instances()[0])


def test_grader_template_keeps_reference_rubric_and_requests_json():
    assert REFERENCE_GRADER_TEMPLATE in GRADER_TEMPLATE
    assert "Return only one JSON object" in GRADER_TEMPLATE
    assert '"correct": "yes or no"' in GRADER_TEMPLATE
    assert "Answer equivalence policy" in GRADER_TEMPLATE
    assert "capitalization, accents/diacritics, punctuation" in GRADER_TEMPLATE
    assert "honorifics and titles such as Dr., Professor" in GRADER_TEMPLATE
    assert "missing middle names or middle initials" in GRADER_TEMPLATE
    assert "brand plus the product name is correct" in GRADER_TEMPLATE
    assert "lists multiple candidates without committing" in GRADER_TEMPLATE


class _FakeLLMClient:
    response_text = "correct: yes"
    configs: list[LLMConfig] = []
    last_messages: list[Any] = []

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.configs.append(config)

    async def chat(self, messages, tools=None, **kwargs):
        self.__class__.last_messages = messages
        return LLMResponse(content=self.response_text)


@pytest.mark.asyncio
async def test_browsecomp_evaluator_accepts_yes(monkeypatch):
    monkeypatch.setattr(
        "aweagent.tasks.browsecomp.evaluator.LLMClient",
        _FakeLLMClient,
    )
    _FakeLLMClient.response_text = "reasoning: ok\ncorrect: yes\nconfidence: 100"
    instance = Instance(
        id="bc_1",
        dataset_id="browsecomp",
        metadata={
            "question": "Question?",
            "answer": "Answer",
            "_agent_final_answer": "Answer",
        },
    )

    result = await BrowseCompEvaluator(LLMConfig(model="judge")).evaluate(
        instance,
        "",
        runtime=object(),
    )

    assert result.accepted
    assert result.score == 1.0
    assert result.details["judge_correct"] == "yes"
    assert result.details["judge_parse_error"] is False
    assert result.details["judge_parse_method"] == "regex"
    assert "[response]: Answer" in _FakeLLMClient.last_messages[0].content


@pytest.mark.asyncio
async def test_browsecomp_evaluator_accepts_json_yes(monkeypatch):
    monkeypatch.setattr(
        "aweagent.tasks.browsecomp.evaluator.LLMClient",
        _FakeLLMClient,
    )
    _FakeLLMClient.response_text = json.dumps({
        "extracted_final_answer": "Answer",
        "reasoning": "same",
        "correct": "yes",
        "confidence": 100,
    })
    instance = Instance(
        id="bc_1",
        dataset_id="browsecomp",
        metadata={
            "question": "Question?",
            "answer": "Answer",
            "_agent_final_answer": "Answer",
        },
    )

    result = await BrowseCompEvaluator().evaluate(instance, "", runtime=object())

    assert result.accepted
    assert result.details["judge_correct"] == "yes"
    assert result.details["judge_parse_method"] == "json"


@pytest.mark.asyncio
async def test_browsecomp_evaluator_accepts_fenced_json_yes(monkeypatch):
    monkeypatch.setattr(
        "aweagent.tasks.browsecomp.evaluator.LLMClient",
        _FakeLLMClient,
    )
    _FakeLLMClient.response_text = (
        "```json\n"
        '{"extracted_final_answer": "Answer", "reasoning": "same", '
        '"correct": "yes", "confidence": 100}'
        "\n```"
    )
    instance = Instance(
        id="bc_1",
        dataset_id="browsecomp",
        metadata={
            "question": "Question?",
            "answer": "Answer",
            "_agent_final_answer": "Answer",
        },
    )

    result = await BrowseCompEvaluator().evaluate(instance, "", runtime=object())

    assert result.accepted
    assert result.details["judge_correct"] == "yes"
    assert result.details["judge_parse_method"] == "json"


@pytest.mark.asyncio
async def test_browsecomp_evaluator_accepts_markdown_correct_label(monkeypatch):
    monkeypatch.setattr(
        "aweagent.tasks.browsecomp.evaluator.LLMClient",
        _FakeLLMClient,
    )
    _FakeLLMClient.response_text = "**correct**: yes"
    instance = Instance(
        id="bc_1",
        dataset_id="browsecomp",
        metadata={
            "question": "Question?",
            "answer": "Answer",
            "_agent_final_answer": "Answer",
        },
    )

    result = await BrowseCompEvaluator().evaluate(instance, "", runtime=object())

    assert result.accepted
    assert result.details["judge_correct"] == "yes"
    assert result.details["judge_parse_method"] == "regex"


@pytest.mark.asyncio
async def test_browsecomp_evaluator_rejects_no_or_unparsed(monkeypatch):
    monkeypatch.setattr(
        "aweagent.tasks.browsecomp.evaluator.LLMClient",
        _FakeLLMClient,
    )
    instance = Instance(
        id="bc_1",
        dataset_id="browsecomp",
        metadata={
            "question": "Question?",
            "answer": "Answer",
            "_agent_final_answer": "Wrong",
        },
    )

    _FakeLLMClient.response_text = '"correct": "no"'
    no_result = await BrowseCompEvaluator().evaluate(instance, "", runtime=object())
    assert not no_result.accepted
    assert no_result.score == 0.0
    assert no_result.details["judge_correct"] == "no"
    assert no_result.details["judge_parse_error"] is False

    _FakeLLMClient.response_text = "not parseable"
    bad_result = await BrowseCompEvaluator().evaluate(instance, "", runtime=object())
    assert not bad_result.accepted
    assert bad_result.details["judge_correct"] == "no"
    assert bad_result.details["judge_parse_error"] is True
    assert bad_result.details["judge_parse_error_reason"] == "missing_correct_yes_no"


def test_grader_correct_accepts_bool_and_punctuated():
    assert _parse_judge_correct('{"correct": true}')[0] == "yes"
    assert _parse_judge_correct('{"correct": false}')[0] == "no"
    assert _parse_judge_correct('{"correct": "Yes."}')[0] == "yes"
    assert _parse_judge_correct('{"correct": "yes, within margin of error"}')[0] == "yes"


def test_grader_correct_falls_back_to_regex_when_json_field_unusable():
    verdict, meta = _parse_judge_correct('{"correct": "maybe"}\ncorrect: yes')
    assert verdict == "yes"
    assert meta["judge_parse_method"] == "regex"
    assert meta["judge_parse_error"] is False


def test_grader_correct_unparseable_defaults_to_no():
    verdict, meta = _parse_judge_correct('{"correct": "maybe"}')
    assert verdict == "no"
    assert meta["judge_parse_error"] is True
    assert meta["judge_parse_error_reason"] == "missing_correct_yes_no"


@pytest.mark.asyncio
async def test_browsecomp_evaluator_flags_grader_error(monkeypatch):
    class _RaisingLLMClient:
        def __init__(self, config: LLMConfig) -> None:
            pass

        async def chat(self, messages, tools=None, **kwargs):
            raise RuntimeError("grader boom")

    monkeypatch.setattr(
        "aweagent.tasks.browsecomp.evaluator.LLMClient",
        _RaisingLLMClient,
    )
    instance = Instance(
        id="bc_1",
        dataset_id="browsecomp",
        metadata={"question": "Q?", "answer": "A", "_agent_final_answer": "A"},
    )

    result = await BrowseCompEvaluator().evaluate(instance, "", runtime=object())

    # Grader failure scores 0 like any unaccepted answer, but is flagged — and
    # still counts in the denominator (no special-casing).
    assert not result.accepted
    assert result.score == 0.0
    assert result.details["grader_error"] is True
    assert "grader boom" in result.details["error"]
    assert "judge_correct" not in result.details


def test_maybe_decrypt_keeps_plaintext_when_canary_present():
    # A plaintext answer is not valid canary ciphertext, so it must survive
    # unchanged even though a canary column is present.
    assert _maybe_decrypt("Real Answer", "some-canary") == "Real Answer"


def test_is_plausible_text_rejects_control_characters():
    assert _is_plausible_text("Normal answer\nwith tabs\t") is True
    assert _is_plausible_text("garbled\x00\x07output") is False


def test_browsecomp_cli_uses_judge_llm_when_configured(tmp_path):
    data_file = tmp_path / "browsecomp.json"
    data_file.write_text(json.dumps([
        {"id": "bc", "problem": "Question?", "answer": "Answer"}
    ]))
    config = AweAgentConfig(
        llm={"model": "search-model"},
        eval={"judge_llm": {"model": "judge-model"}},
        task={"type": "browsecomp", "data_file": str(data_file)},
    )

    task = build_task(config)
    evaluator = task.default_evaluator()

    assert evaluator._grader_llm_config.model == "judge-model"


def test_browsecomp_cli_falls_back_to_search_llm(tmp_path):
    data_file = tmp_path / "browsecomp.json"
    data_file.write_text(json.dumps([
        {"id": "bc", "problem": "Question?", "answer": "Answer"}
    ]))
    config = AweAgentConfig(
        llm={"model": "search-model"},
        task={"type": "browsecomp", "data_file": str(data_file)},
    )

    task = build_task(config)
    evaluator = task.default_evaluator()

    assert evaluator._grader_llm_config.model == "search-model"


class _NoSessionRuntime:
    """A runtime whose session must never be opened — BrowseComp opens none.

    Isolated evaluation still constructs a runtime (the BrowseComp evaluator
    ignores it), so the guard is on ``session()``, which only the agent path
    would call.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    def session(self, image=None):
        raise AssertionError("BrowseComp must not open a runtime session")


class _FinalAnswerLoop:
    async def run(self, task_prompt: str) -> AgentResult:
        return AgentResult(
            trajectory=Trajectory(),
            messages=[],
            finish_reason="finish",
            metadata={"final_answer": "Answer"},
        )


class _FinalAnswerAgent(Agent):
    @classmethod
    def from_config(cls, config):
        return cls()

    async def step(self, context: AgentContext) -> Action:
        raise AssertionError("loop should be provided by create_loop")

    def get_system_prompt(self, task_info: dict[str, Any]) -> str:
        return ""

    def get_tools(self) -> list[Tool]:
        return []

    def create_loop(self, context: AgentContext) -> _FinalAnswerLoop:
        return _FinalAnswerLoop()


class _ListTask(Task):
    def __init__(self, ids: list[str]) -> None:
        self._instances = [
            Instance(id=instance_id, dataset_id="list", metadata={"task_type": "list"})
            for instance_id in ids
        ]

    def get_instances(self, instance_ids: list[str] | None = None) -> list[Instance]:
        if not instance_ids:
            return list(self._instances)
        by_id = {instance.id: instance for instance in self._instances}
        return [by_id[instance_id] for instance_id in instance_ids if instance_id in by_id]

    def get_prompt(self, instance: Instance) -> str:
        return f"Prompt for {instance.id}"

    def requires_runtime(self) -> bool:
        return False

    def requires_patch_extraction(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_runner_passes_final_answer_to_agent_result_evaluator(
    tmp_path,
    monkeypatch,
):
    class _FakeOpenAIBackend:
        def __init__(self, config: LLMConfig) -> None:
            self.config = config

        async def chat(self, messages, tools=None, **kwargs):
            return LLMResponse(content="")

    llm_registry.register("openai", _FakeOpenAIBackend)
    runtime_registry.register("browsecomp_no_session", _NoSessionRuntime)
    monkeypatch.setattr(
        "aweagent.tasks.browsecomp.evaluator.LLMClient",
        _FakeLLMClient,
    )
    _FakeLLMClient.response_text = "correct: yes"
    data_file = tmp_path / "browsecomp.json"
    data_file.write_text(json.dumps([
        {"id": "bc", "problem": "Question?", "answer": "Answer"}
    ]))
    task = BrowseCompTask(
        data_file=str(data_file),
        grader_llm_config=LLMConfig(model="judge"),
    )
    runner = TaskRunner(
        task=task,
        agent_factory=lambda search_constraints=None: _FinalAnswerAgent(),
        llm_config=LLMConfig(model="search"),
        runtime_config=RuntimeConfig(backend="browsecomp_no_session"),
        max_retries=1,
        max_concurrent=1,
        save_trajectories=False,
    )

    result = await runner._run_instance(task.get_instances()[0])

    assert result.eval_result is not None
    assert result.eval_result.accepted
    assert "[response]: Answer" in _FakeLLMClient.last_messages[0].content


@pytest.mark.asyncio
async def test_runner_limits_instances_after_id_filtering(tmp_path):
    task = _ListTask(["train_0", "train_1", "train_2"])
    runner = TaskRunner(
        task=task,
        agent_factory=lambda search_constraints=None: _FinalAnswerAgent(),
        llm_config=LLMConfig(model="search"),
        runtime_config=RuntimeConfig(),
        max_instances=1,
        max_retries=1,
        max_concurrent=1,
        output_path=tmp_path,
        save_trajectories=False,
    )

    results = await runner.run_all(["train_2", "train_0"])

    assert [result.instance_id for result in results] == ["train_2"]


def test_select_instances_start_end_are_inclusive():
    instances = _ListTask(["train_0", "train_1", "train_2", "train_3"]).get_instances()

    assert [inst.id for inst in select_instances(instances, start_index=2)] == [
        "train_2",
        "train_3",
    ]
    assert [inst.id for inst in select_instances(instances, end_index=1)] == [
        "train_0",
        "train_1",
    ]
    assert [
        inst.id
        for inst in select_instances(instances, start_index=1, end_index=2)
    ] == ["train_1", "train_2"]


@pytest.mark.asyncio
async def test_runner_applies_id_filter_then_range_then_max(tmp_path):
    task = _ListTask(["train_0", "train_1", "train_2", "train_3"])
    runner = TaskRunner(
        task=task,
        agent_factory=lambda search_constraints=None: _FinalAnswerAgent(),
        llm_config=LLMConfig(model="search"),
        runtime_config=RuntimeConfig(),
        start_index=1,
        end_index=2,
        max_instances=1,
        max_retries=1,
        max_concurrent=1,
        output_path=tmp_path,
        save_trajectories=False,
    )

    results = await runner.run_all(["train_3", "train_1", "train_2", "train_0"])

    assert [result.instance_id for result in results] == ["train_1"]


def test_select_instances_warns_when_end_or_max_exceeds_available(caplog):
    instances = _ListTask(["train_0", "train_1", "train_2"]).get_instances()
    caplog.set_level(logging.WARNING)

    selected = select_instances(instances, end_index=99, max_instances=99)

    assert [inst.id for inst in selected] == ["train_0", "train_1", "train_2"]
    messages = "\n".join(record.message for record in caplog.records)
    assert "end_index=99 exceeds last loaded index 2" in messages
    assert "max_instances=99 exceeds 3 selected instance(s)" in messages


def test_select_instances_warns_when_start_exceeds_available(caplog):
    instances = _ListTask(["train_0", "train_1", "train_2"]).get_instances()
    caplog.set_level(logging.WARNING)

    selected = select_instances(instances, start_index=99)

    assert selected == []
    assert "start_index=99 is beyond 3 loaded instance(s)" in caplog.text


def test_cli_dry_run_respects_max_instances(tmp_path):
    data_file = tmp_path / "browsecomp.json"
    data_file.write_text(json.dumps([
        {"id": "bc0", "problem": "Question 0?", "answer": "Answer 0"},
        {"id": "bc1", "problem": "Question 1?", "answer": "Answer 1"},
        {"id": "bc2", "problem": "Question 2?", "answer": "Answer 2"},
    ]))
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"""
task:
  type: browsecomp
  data_file: {data_file}
eval:
  enabled: false
"""
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aweagent.cli",
            "run",
            "--config",
            str(config_file),
            "--max-instances",
            "2",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Dry run — 2 instances loaded" in result.stdout
    assert "bc0" in result.stdout
    assert "bc1" in result.stdout
    assert "bc2" not in result.stdout


def test_cli_dry_run_respects_instance_range(tmp_path):
    data_file = tmp_path / "browsecomp.json"
    data_file.write_text(json.dumps([
        {"id": "bc0", "problem": "Question 0?", "answer": "Answer 0"},
        {"id": "bc1", "problem": "Question 1?", "answer": "Answer 1"},
        {"id": "bc2", "problem": "Question 2?", "answer": "Answer 2"},
    ]))
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"""
task:
  type: browsecomp
  data_file: {data_file}
eval:
  enabled: false
"""
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aweagent.cli",
            "run",
            "--config",
            str(config_file),
            "--start-index",
            "1",
            "--end-index",
            "2",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Dry run — 2 instances loaded" in result.stdout
    assert "bc0" not in result.stdout
    assert "bc1" in result.stdout
    assert "bc2" in result.stdout


def test_cli_dry_run_warns_when_max_instances_exceeds_selected(tmp_path):
    data_file = tmp_path / "browsecomp.json"
    data_file.write_text(json.dumps([
        {"id": "bc0", "problem": "Question 0?", "answer": "Answer 0"},
        {"id": "bc1", "problem": "Question 1?", "answer": "Answer 1"},
    ]))
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"""
task:
  type: browsecomp
  data_file: {data_file}
eval:
  enabled: false
"""
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aweagent.cli",
            "run",
            "--config",
            str(config_file),
            "--max-instances",
            "99",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Dry run — 2 instances loaded" in result.stdout
    assert "max_instances=99 exceeds 2 selected instance(s)" in result.stderr


def test_trajectory_record_includes_answer_provenance():
    provenance = {
        "agent_submitted_final_answer": False,
        "forced_final_answer": True,
        "forced_final_answer_stage": "current_history",
        "forced_final_answer_reason": "max_steps",
    }
    agent_result = AgentResult(
        trajectory=Trajectory(),
        messages=[],
        finish_reason="finish",
        metadata={"final_answer": "Answer", "answer_provenance": provenance},
    )
    record = _build_trajectory_record(TaskResult(
        instance_id="bc",
        agent_result=agent_result,
    ))

    assert record is not None
    assert record["final_answer"] == "Answer"
    assert record["answer_provenance"] == provenance
