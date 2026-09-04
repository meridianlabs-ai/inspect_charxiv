import logging
from io import BytesIO
from pathlib import Path

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage, ChatMessageUser, ContentImage, ContentText
from PIL import Image

from inspect_charxiv.constants import (
    DESCRIPTIVE_RESP_INST,
    INSPECT_EVALS_CACHE_PATH,
    MANUAL_GRADING_CORRECTED_TARGETS,
    REASONING_RESP_INST,
)

NUMBER_IN_GENERAL_QUESTION: int = 4

logger = logging.getLogger(__name__)


def convert_descriptive_question(
    question_index: int,
    input_sample: dict[str, str | int | None],
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
                        input_sample.get(f"descriptive_q{question_index}")
                    ]
                ),
            ]
        )
    ]
    qid = (
        input_sample.get("figure_path").removeprefix("images/").removesuffix(".jpg")
        + f".{question_index}"
    )
    return Sample(
        input=message,
        target=correct_target(
            question_id=qid, target=input_sample.get(f"descriptive_a{question_index}")
        )
        if apply_corrections
        else input_sample.get(f"descriptive_a{question_index}"),
        id=qid,
        metadata={
            "is_descriptive": True,
            "question_id": input_sample.get(f"descriptive_q{question_index}"),
            "field_of_study": input_sample.get("category"),
            "target_corrected": qid in MANUAL_GRADING_CORRECTED_TARGETS,
        },
    )


def convert_reasoning_question(
    input_sample: dict[str, str | int | None], apply_corrections: bool = True
) -> Sample:
    instructions: str
    if input_sample.get("reasoning_a_type") == NUMBER_IN_GENERAL_QUESTION:
        instructions = REASONING_RESP_INST[input_sample.get("reasoning_a_type")].format(
            input_sample.get("reasoning_q"),
            number_in_general_question_instructions(input_sample.get("reasoning_a")),
        )
    else:
        instructions = REASONING_RESP_INST[input_sample.get("reasoning_a_type")].format(
            input_sample.get("reasoning_q")
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
        input_sample.get("figure_path").removeprefix("images/").removesuffix(".jpg")
        + ".5"
    )
    return Sample(
        input=message,
        target=correct_target(question_id=qid, target=input_sample.get("reasoning_a"))
        if apply_corrections
        else input_sample.get("reasoning_a"),
        id=qid,
        metadata={
            "is_descriptive": False,
            "question_id": input_sample.get("reasoning_a_type"),
            "question_text": input_sample.get("reasoning_q"),
            "field_of_study": input_sample.get("category"),
            "target_corrected": qid in MANUAL_GRADING_CORRECTED_TARGETS,
        },
    )


# helper function that generates the appropriate prompt prefix explaining which subplot the model should be looking at
def convert_to_subplot_pos(
    subplot_row: int | None, subplot_col: int | None, subplot_loc: str | None
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


def convert_image(input_sample: dict[str, str | int | None]) -> str:
    IMAGE_BASE_DIR = INSPECT_EVALS_CACHE_PATH / "charxiv_images"
    image = Path(IMAGE_BASE_DIR / input_sample["figure_path"])
    image_bytes = input_sample["image"]["bytes"]

    if not image.exists():
        logger.debug(f"Extracting {image.name}")
        image.parent.mkdir(exist_ok=True, parents=True)
        img = Image.open(BytesIO(image_bytes))
        img.save(image, format="JPEG")

    return str(image)


def number_in_general_question_instructions(answer: float) -> str:
    if float(answer) % 1 == 0:
        return "* Your final answer must be an exact integer."

    decimal_places = len(str(answer).split(".")[1])

    return f"* Your final answer must be a number with {decimal_places} decimal places."


def correct_target(question_id: str, target: str) -> str:
    if question_id in MANUAL_GRADING_CORRECTED_TARGETS:
        return target + " -OR- " + MANUAL_GRADING_CORRECTED_TARGETS[question_id]
    else:
        return target
