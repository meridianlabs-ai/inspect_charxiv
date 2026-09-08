import logging
from pathlib import Path
from typing import Any

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage, ChatMessageUser, ContentImage, ContentText

from inspect_charxiv.constants import (
    DESCRIPTIVE_RESP_INST,
    INSPECT_EVALS_CACHE_PATH,
    MANUAL_GRADING_CORRECTED_TARGETS,
    REASONING_RESP_INST,
)

NUMBER_IN_GENERAL_QUESTION: int = 4
IMAGE_BASE_DIR = INSPECT_EVALS_CACHE_PATH / "charxiv_images"

logger = logging.getLogger(__name__)


def convert_descriptive_question(
    question_index: int,
    input_sample: dict[str, Any],
    apply_corrections: bool = True,
) -> Sample:
    subplot_pos = convert_to_subplot_pos(
        input_sample["subplot_row"],
        input_sample["subplot_col"],
        input_sample["subplot_loc"],
    )
    message: list[ChatMessage] = [
        ChatMessageUser(
            content=[
                ContentImage(image=convert_image(input_sample)),
                ContentText(
                    text=subplot_pos
                    + DESCRIPTIVE_RESP_INST[
                        input_sample[f"descriptive_q{question_index}"]
                    ]
                ),
            ]
        )
    ]
    qid = (
        input_sample["figure_path"].removeprefix("images/").removesuffix(".jpg")
        + f".{question_index}"
    )
    return Sample(
        input=message,
        target=correct_target(
            question_id=qid, target=input_sample[f"descriptive_a{question_index}"]
        )
        if apply_corrections
        else str(input_sample[f"descriptive_a{question_index}"]),
        id=qid,
        metadata={
            "is_descriptive": True,
            "question_id": input_sample[f"descriptive_q{question_index}"],
            "field_of_study": input_sample["category"],
            "flagged_for_correction": qid in MANUAL_GRADING_CORRECTED_TARGETS,
            "correction_applied": apply_corrections
            and qid in MANUAL_GRADING_CORRECTED_TARGETS,
        },
    )


def convert_reasoning_question(
    input_sample: dict[str, Any], apply_corrections: bool = True
) -> Sample:
    instructions: str
    if input_sample.get("reasoning_a_type") == NUMBER_IN_GENERAL_QUESTION:
        instructions = REASONING_RESP_INST[input_sample["reasoning_a_type"]].format(
            input_sample["reasoning_q"],
            number_in_general_question_instructions(input_sample["reasoning_a"]),
        )
    else:
        instructions = REASONING_RESP_INST[input_sample["reasoning_a_type"]].format(
            input_sample["reasoning_q"]
        )

    message: list[ChatMessage] = [
        ChatMessageUser(
            content=[
                ContentImage(image=convert_image(input_sample)),
                ContentText(text=instructions),
            ]
        )
    ]
    qid = (
        input_sample["figure_path"].removeprefix("images/").removesuffix(".jpg") + ".5"
    )
    return Sample(
        input=message,
        target=correct_target(question_id=qid, target=input_sample["reasoning_a"])
        if apply_corrections
        else str(input_sample["reasoning_a"]),
        id=qid,
        metadata={
            "is_descriptive": False,
            "question_id": input_sample["reasoning_a_type"],
            "question_text": input_sample["reasoning_q"],
            "field_of_study": input_sample["category"],
            "flagged_for_correction": qid in MANUAL_GRADING_CORRECTED_TARGETS,
            "correction_applied": apply_corrections
            and qid in MANUAL_GRADING_CORRECTED_TARGETS,
        },
    )


# helper function that generates the appropriate prompt prefix explaining which subplot the model should be looking at
def convert_to_subplot_pos(
    subplot_row: str | int | None,
    subplot_col: str | int | None,
    subplot_loc: str | None,
) -> str:
    result = ""
    if subplot_row == 0:
        result += "For the current plot, "
    elif subplot_loc is None:
        result += (
            "For the subplot at row "
            + str(subplot_row)
            + " and column "
            + str(subplot_col)
            + ", "
        )
    else:
        result += "For " + str(subplot_loc) + ", "
    return result


def convert_image(input_sample: dict[str, Any]) -> str:
    """Cache the chart's original JPEG bytes to disk and return the path.

    input_sample["image"]["bytes"] is the complete original .jpg file as stored
    on Hugging Face, already compressed once by the CharXiv authors. Writing it
    verbatim preserves that single compression; decoding and re-saving would add
    a second lossy pass that blurs the tick labels and axis text the questions
    ask about. Caching by figure_path lets the 5 questions per chart reuse one
    file, and keeps the images out of the repository (avoiding licensing issues).
    """
    image = IMAGE_BASE_DIR / input_sample["figure_path"]
    if not image.exists():
        image.parent.mkdir(exist_ok=True, parents=True)
        # Write to a temp path and atomically rename so an interrupted write
        # can't leave a truncated file that exists() then trusts forever.
        tmp = image.with_suffix(image.suffix + ".tmp")
        tmp.write_bytes(input_sample["image"]["bytes"])
        tmp.replace(image)
    return str(image)


def number_in_general_question_instructions(answer: str | int | None) -> str:
    if answer is None:
        return ""
    elif str(answer).find(".") == -1:
        return "* Your final answer must be an exact integer."
    else:
        decimal_places = len(str(answer).split(".")[1])
        return f"* Your final answer must be a number with {decimal_places} decimal places."


def correct_target(question_id: str, target: str | int | None) -> str:
    if (target is None) or (target == ""):
        raise ValueError("Target is None or empty")
    if question_id in MANUAL_GRADING_CORRECTED_TARGETS:
        return str(target) + " -OR- " + MANUAL_GRADING_CORRECTED_TARGETS[question_id]
    else:
        return str(target)
