import json
import re
from typing import Literal, NamedTuple

from inspect_ai import Task, task
from inspect_ai.model import get_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate

from inspect_charxiv.constants import (
    DESCRIPTIVE_GRADING_QMAP,
    GRADING_PREFIX,
    REASONING_GRADING_INST,
)
from inspect_charxiv.dataset import FieldOfStudy, load_charxiv_dataset


@task
def charxiv(
    subset: Literal["descriptive", "reasoning"] | None = None,
    category: FieldOfStudy | list[FieldOfStudy] | None = None,
    apply_corrections: bool = True,
) -> Task:
    return Task(
        dataset=load_charxiv_dataset(
            subset=subset, category=category, apply_corrections=apply_corrections
        ),
        solver=[generate()],
        scorer=charxiv_scorer(),
    )


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class GradeResult(NamedTuple):
    correct: bool
    extracted: str | None
    parse_error: bool


def _parse_grade(completion: str) -> GradeResult:
    """Leniently extract the grader's verdict from its completion.

    Tolerates code fences, prose around the JSON object, either key spelling
    used by the rubrics, and score values expressed as int, str, or bool. A
    completion with no parseable JSON object scores incorrect rather than
    raising, with parse_error set so such samples can be counted in the log.
    """
    match = _JSON_OBJECT.search(completion)
    if match is None:
        return GradeResult(correct=False, extracted=None, parse_error=True)
    try:
        score_object = json.loads(match.group())
    except json.JSONDecodeError:
        return GradeResult(correct=False, extracted=None, parse_error=True)
    raw_score = score_object.get("score", score_object.get("score_T1"))
    extracted = score_object.get(
        "extracted_answer",
        score_object.get("extract_answer", score_object.get("extract_answer_T1")),
    )
    return GradeResult(
        correct=str(raw_score).strip().lower() in {"1", "1.0", "true"},
        extracted=str(extracted) if extracted is not None else None,
        parse_error=False,
    )


@scorer(metrics=[accuracy(), stderr()])
def charxiv_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        # Resolve the grader inside score() so that model roles passed to
        # eval() from Python (not just --model-role on the CLI) are honoured.
        grader_model = get_model(role="grader", default="openai/gpt-4o")
        answer = state.output.message.text
        if state.metadata["is_descriptive"]:
            score_prompt = GRADING_PREFIX + (
                DESCRIPTIVE_GRADING_QMAP[state.metadata["question_id"]]
                .replace("<|ground_truth|>", target.text)
                .replace("<|response|>", answer)
            )
        else:
            score_prompt = GRADING_PREFIX + (
                REASONING_GRADING_INST[state.metadata["question_id"]]
                .replace("<|question|>", state.metadata["question_text"])
                .replace("<|ground_truth|>", target.text)
                .replace("<|response|>", answer)
            )

        output = await grader_model.generate(input=score_prompt)

        grade = _parse_grade(output.completion)
        return Score(
            value=CORRECT if grade.correct else INCORRECT,
            answer=answer,
            explanation=output.completion,
            metadata={"grader_parse_error": grade.parse_error},
        )

    return score
