import logging
from pathlib import Path
from io import BytesIO
from PIL import Image
from typing import Dict

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage, ChatMessageUser, ContentImage, ContentText

from inspect_charxiv.constants import DESCRIPTIVE_RESP_INST, MANUAL_GRADING_CORRECTED_TARGETS, REASONING_RESP_INST, INSPECT_EVALS_CACHE_PATH


logger = logging.getLogger(__name__)

def convert_descriptive_question(question_index: int, input_sample: Dict[str, str | int | None], correct_targets: bool = True) -> Sample:
    subplot_pos = convert_to_subplot_pos(input_sample['subplot_row'], input_sample['subplot_col'], input_sample['subplot_loc'])
    message: list[ChatMessage] = [
        ChatMessageUser(
            content =[
                ContentImage(image= convert_image(input_sample)),
                ContentText(text= subplot_pos + DESCRIPTIVE_RESP_INST[input_sample.get(f'descriptive_q{question_index}')])
            ]
        )
    ]
    qid= input_sample.get('figure_path').removeprefix('images/').removesuffix('.jpg') + f".{question_index}"
    return Sample(
        input= message, 
        target= correct_target(question_id=qid, target=input_sample.get(f'descriptive_a{question_index}')) if correct_targets else input_sample.get(f'descriptive_a{question_index}'),
        id= qid,
        metadata= {
            "is_descriptive": True, 
            "question_id": input_sample.get(f'descriptive_q{question_index}'),
            "field_of_study": input_sample.get('category'),
            "target_corrected": qid in MANUAL_GRADING_CORRECTED_TARGETS,
        }
    )

def convert_reasoning_question(input_sample: Dict[str, str | int | None], correct_targets: bool = True) -> Sample:
    instructions: str
    if (input_sample.get('reasoning_a_type') == 4):
        instructions= REASONING_RESP_INST[input_sample.get('reasoning_a_type')].format(input_sample.get('reasoning_q'), number_in_general_question_instructions(input_sample.get('reasoning_a')))
    else:
        instructions= REASONING_RESP_INST[input_sample.get('reasoning_a_type')].format(input_sample.get('reasoning_q'))

    message: list[ChatMessage] = [
        ChatMessageUser(
            content= [
                ContentImage(image= convert_image(input_sample)),
                ContentText(text= instructions)
            ]
        )
    ]
    qid = input_sample.get('figure_path').removeprefix('images/').removesuffix('.jpg') + ".5"
    return Sample(
        input= message, 
        target= correct_target(question_id=qid, target=input_sample.get('reasoning_a')) if correct_targets else input_sample.get('reasoning_a'),
        id= qid,
        metadata= {
            "is_descriptive": False, 
            "question_id": input_sample.get('reasoning_a_type'),
            "question_text": input_sample.get('reasoning_q'),
            "field_of_study": input_sample.get('category'),
            "target_corrected": qid in MANUAL_GRADING_CORRECTED_TARGETS
        }
    )

# helper function that generates the appropriate prompt prefix explaining which subplot the model should be looking at
def convert_to_subplot_pos(subplot_row: int | None, subplot_col: int | None, subplot_loc: str | None) -> str:
    result = ''
    if(subplot_row == 0):
        result += "For the current plot, "
    elif(subplot_loc == None):
        result += "For the subplot at row " + str(subplot_row) + " and column " + str(subplot_col) + ", "
    else:
        result += "For " + str(subplot_loc) + ", "
    return result

# This needs to be blown up. We can use mathvista but it must be understood that when they get record[image] it's not the same as our record[image]. Their's refers to the path where ours are the actual bytes. If we can take the figure_path that they use, determine whether it's in our file location and if it's not: put it there using the decoded bytes from the dataset, then we can ensure that image actually corresponds to the classpath that we think it will.
# TODO decide on implementation of the image conversion.
# OPTION 1 is to take the image from huggingface and convert the image to a base64 string.
# OPTION 2 is to store the image files in the repository, then convert the path used in the experiment to a path for finding the images in inspect.
def convert_image(input_sample: Dict[str, str | int | None]) -> str:
    IMAGE_BASE_DIR = INSPECT_EVALS_CACHE_PATH / "charxiv_images"
    image = Path(IMAGE_BASE_DIR / input_sample['figure_path'])
    # TODO if the image doesn't exist, save it to our cache using the Bytes found in the input_sample['image']

    image_bytes = input_sample['image']['bytes']

    if not image.exists():
        logger.debug(f"Extracting {image.name}")
        image.parent.mkdir(exist_ok=True, parents=True)
        img = Image.open(BytesIO(image_bytes))
        img.save(image, format='JPEG')

    return str(image)

# nothing actually puts the image in here with this solution

def number_in_general_question_instructions(answer: float) -> str:
    if(float(answer) % 1 == 0):
        return"* Your final answer must be an exact integer."
    
    decimal_places= len(str(answer).split('.')[1])

    return f"* Your final answer must be a number with {decimal_places} decimal places."

def correct_target(question_id: str, target: str) -> str:
    if question_id in MANUAL_GRADING_CORRECTED_TARGETS:
        return target + " -OR- " + MANUAL_GRADING_CORRECTED_TARGETS[question_id]
    else:
        return target