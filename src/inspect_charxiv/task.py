import json
from typing import Literal

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


@scorer(metrics=[accuracy(), stderr()])
def charxiv_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        # Resolve the grader inside score() so that model roles passed to
        # eval() from Python (not just --model-role on the CLI) are honoured.
        grader_model = get_model(role="grader", default="openai/gpt-4o")
        answer = state.output.message.text
        result: str
        if state.metadata["is_descriptive"]:
            result = GRADING_PREFIX + (
                DESCRIPTIVE_GRADING_QMAP[state.metadata["question_id"]]
                .replace("<|ground_truth|>", target.text)
                .replace("<|response|>", answer)
            )
        else:
            result = GRADING_PREFIX + (
                REASONING_GRADING_INST[state.metadata["question_id"]]
                .replace("<|question|>", state.metadata["question_text"])
                .replace("<|ground_truth|>", target.text)
                .replace("<|response|>", answer)
            )
        score_prompt = result

        result = await grader_model.generate(input=score_prompt)

        score_object = json.loads(result.completion.strip())
        return Score(
            value=CORRECT if score_object.get("score") == 1 else INCORRECT,
            answer=answer,
            explanation=result.completion,
        )

    return score
