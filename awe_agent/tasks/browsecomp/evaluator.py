# ruff: noqa: E501
"""BrowseComp evaluator for DeepSearch text answers."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any

from awe_agent.core.llm.client import LLMClient
from awe_agent.core.llm.config import LLMConfig
from awe_agent.core.llm.types import Message
from awe_agent.core.task.protocol import Evaluator
from awe_agent.core.task.types import EvalResult, Instance

if TYPE_CHECKING:
    from awe_agent.core.runtime.protocol import Runtime

logger = logging.getLogger(__name__)

_CORRECT_RE = re.compile(
    r"""
    (?:^|[\s,{])
    [*_`]*["']?[*_`]*
    correct
    [*_`]*["']?[*_`]*
    \s*[:=]\s*
    ["']?(yes|no)["']?
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.I)

GRADER_TEMPLATE = r"""
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.

Return only one JSON object. Do not wrap it in markdown or add any extra text.
Use exactly this JSON shape:
{{
  "extracted_final_answer": "string or null",
  "reasoning": "string",
  "correct": "yes or no",
  "confidence": 100
}}
The "correct" value must be exactly "yes" or "no".
""".strip()


def _strip_json_fence(text: str) -> str:
    match = _JSON_FENCE_RE.match(text.strip())
    return match.group(1).strip() if match else text.strip()


def _normalize_correct(value: Any) -> str | None:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if not isinstance(value, str):
        return None
    match = re.match(r"\s*(yes|no)\b", value, re.IGNORECASE)
    return match.group(1).lower() if match else None


def _load_json_dict(text: str) -> dict[str, Any] | None:
    stripped = _strip_json_fence(text)
    candidates = [stripped]
    if stripped != text.strip():
        candidates.append(text.strip())

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _parse_judge_correct(grading_response: str) -> tuple[str, dict[str, Any]]:
    data = _load_json_dict(grading_response)
    if data is not None:
        judge_correct = _normalize_correct(data.get("correct"))
        if judge_correct:
            return judge_correct, {
                "judge_parse_error": False,
                "judge_parse_method": "json",
            }
        # JSON parsed but `correct` was missing or not a yes/no value; fall
        # through to the regex net before giving up.

    match = _CORRECT_RE.search(grading_response)
    if match:
        return match.group(1).lower(), {
            "judge_parse_error": False,
            "judge_parse_method": "regex",
        }

    return "no", {
        "judge_parse_error": True,
        "judge_parse_error_reason": "missing_correct_yes_no",
        "judge_parse_method": "none",
    }


class BrowseCompEvaluator(Evaluator):
    """Judge BrowseComp answers with the official grader prompt."""

    def __init__(
        self,
        grader_llm_config: LLMConfig | None = None,
        timeout: int = 3600,
    ) -> None:
        self._grader_llm_config = (grader_llm_config or LLMConfig()).model_copy(
            update={"timeout": timeout}
        )

    async def evaluate(
        self,
        instance: Instance,
        patch: str,  # unused; present for Evaluator protocol compatibility
        runtime: Runtime,
    ) -> EvalResult:
        start = time.monotonic()
        question = str(instance.metadata.get("question", ""))
        correct_answer = str(instance.metadata.get("answer", ""))
        response = instance.metadata.get("_agent_final_answer")

        if response is None:
            return EvalResult(
                accepted=False,
                score=0.0,
                details={
                    "error": "missing_agent_final_answer",
                    "question": question,
                    "correct_answer": correct_answer,
                    "finish_reason": instance.metadata.get("_agent_finish_reason"),
                },
                duration=time.monotonic() - start,
            )

        response_text = str(response)
        grader_prompt = GRADER_TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            response=response_text,
        )

        try:
            judge = LLMClient(self._grader_llm_config)
            judge_response = await judge.chat(
                messages=[Message(role="user", content=grader_prompt)],
            )
            grading_response = judge_response.content or ""
        except Exception as exc:
            logger.exception("BrowseComp grading failed for %s", instance.id)
            # The grader call failed (timeout, rate limit, ...). Score it 0 like
            # any unaccepted answer, but flag grader_error so a run with many
            # such failures is visible to whoever inspects the results.
            return EvalResult(
                accepted=False,
                score=0.0,
                details={
                    "error": str(exc),
                    "grader_error": True,
                    "question": question,
                    "correct_answer": correct_answer,
                    "prediction": response_text,
                },
                duration=time.monotonic() - start,
            )

        judge_correct, parse_details = _parse_judge_correct(grading_response)
        accepted = judge_correct == "yes"
        return EvalResult(
            accepted=accepted,
            score=1.0 if accepted else 0.0,
            details={
                "question": question,
                "correct_answer": correct_answer,
                "prediction": response_text,
                "grading_response": grading_response,
                "judge_correct": judge_correct,
                **parse_details,
            },
            duration=time.monotonic() - start,
        )
