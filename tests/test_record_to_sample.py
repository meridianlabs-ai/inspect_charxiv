"""Tests for record_to_sample using a real record from the CharXiv dataset.

The record is streamed from Hugging Face at the same pinned revision used by
``load_charxiv_dataset``, so only the first record is downloaded rather than
the full validation split. Image bytes are written to a temporary directory
rather than the real inspect_evals cache.
"""

from pathlib import Path
from typing import Any, Literal

import pytest
from datasets import Image, load_dataset
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText

import inspect_charxiv.utils
from inspect_charxiv.constants import (
    DESCRIPTIVE_RESP_INST,
    MANUAL_GRADING_CORRECTED_TARGETS,
)
from inspect_charxiv.dataset import _make_record_to_sample

# Same pinned revision as load_charxiv_dataset so the record is deterministic.
CHARXIV_REVISION = "f441eb632fc62f6f777830a0f47619e6e86459b0"

pytestmark = pytest.mark.dataset_download


@pytest.fixture(scope="module")
def record() -> dict[str, Any]:
    dataset = load_dataset(
        "princeton-nlp/CharXiv",
        split="validation",
        streaming=True,
        revision=CHARXIV_REVISION,
    )
    # hf_dataset materializes records with Dataset.to_list(), which yields the
    # raw {"bytes", "path"} storage for image columns; disable decoding so this
    # record has the same shape as the ones record_to_sample sees in production.
    dataset = dataset.cast_column("image", Image(decode=False))
    return dict(next(iter(dataset)))


@pytest.fixture(autouse=True)
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # convert_image writes image bytes under IMAGE_BASE_DIR (the real user
    # cache dir); redirect it so tests leave no trace outside tmp_path.
    monkeypatch.setattr(inspect_charxiv.utils, "IMAGE_BASE_DIR", tmp_path)
    return tmp_path


def to_samples(
    record: dict[str, Any],
    subset: Literal["descriptive", "reasoning"] | None = None,
    apply_corrections: bool = True,
) -> list[Sample]:
    record_to_sample = _make_record_to_sample(
        subset=subset, apply_corrections=apply_corrections
    )
    samples = record_to_sample(record)
    assert isinstance(samples, list)
    return samples


def sample_content(sample: Sample) -> tuple[ContentImage, ContentText]:
    """Unpack a sample's single user message into its image and text parts."""
    assert isinstance(sample.input, list)
    assert len(sample.input) == 1
    message = sample.input[0]
    assert isinstance(message, ChatMessageUser)
    assert isinstance(message.content, list)
    image, text = message.content
    assert isinstance(image, ContentImage)
    assert isinstance(text, ContentText)
    return image, text


@pytest.mark.parametrize(
    ("subset", "expected_descriptive"),
    [
        (None, [True, True, True, True, False]),
        ("descriptive", [True, True, True, True]),
        ("reasoning", [False]),
    ],
)
def test_subset_selection(
    record: dict[str, Any],
    subset: Literal["descriptive", "reasoning"] | None,
    expected_descriptive: list[bool],
) -> None:
    samples = to_samples(record, subset=subset)
    assert [
        (sample.metadata or {})["is_descriptive"] for sample in samples
    ] == expected_descriptive


def test_sample_ids(record: dict[str, Any]) -> None:
    figure = record["figure_path"].removeprefix("images/").removesuffix(".jpg")
    samples = to_samples(record)
    assert [sample.id for sample in samples] == [f"{figure}.{i}" for i in range(1, 6)]


def test_descriptive_samples(record: dict[str, Any]) -> None:
    samples = to_samples(record, subset="descriptive")
    for question_index, sample in enumerate(samples, start=1):
        _, text = sample_content(sample)
        # prompt is the subplot prefix followed by the question's instruction
        instruction = DESCRIPTIVE_RESP_INST[record[f"descriptive_q{question_index}"]]
        assert text.text.endswith(instruction)
        assert text.text != instruction
        assert isinstance(sample.target, str) and sample.target
        metadata = sample.metadata or {}
        assert metadata["field_of_study"] == record["category"]
        assert metadata["question_id"] == record[f"descriptive_q{question_index}"]


def test_reasoning_sample(record: dict[str, Any]) -> None:
    [sample] = to_samples(record, subset="reasoning")
    _, text = sample_content(sample)
    assert record["reasoning_q"] in text.text
    assert isinstance(sample.target, str) and sample.target
    metadata = sample.metadata or {}
    assert metadata["field_of_study"] == record["category"]
    assert metadata["question_id"] == record["reasoning_a_type"]
    assert metadata["question_text"] == record["reasoning_q"]


def test_image_cached_verbatim(record: dict[str, Any], cache_dir: Path) -> None:
    [sample] = to_samples(record, subset="reasoning")
    image, _ = sample_content(sample)
    image_path = Path(image.image)
    assert image_path == cache_dir / record["figure_path"]
    assert image_path.read_bytes() == record["image"]["bytes"]


def test_corrections_only_affect_flagged_targets(record: dict[str, Any]) -> None:
    corrected = to_samples(record, apply_corrections=True)
    plain = to_samples(record, apply_corrections=False)
    for corrected_sample, plain_sample in zip(corrected, plain, strict=True):
        qid = corrected_sample.id
        assert isinstance(qid, str)
        assert qid == plain_sample.id
        assert isinstance(plain_sample.target, str)

        flagged = qid in MANUAL_GRADING_CORRECTED_TARGETS
        for sample in (corrected_sample, plain_sample):
            assert (sample.metadata or {})["flagged_for_correction"] is flagged
        assert (plain_sample.metadata or {})["correction_applied"] is False
        assert (corrected_sample.metadata or {})["correction_applied"] is flagged

        if flagged:
            expected = (
                f"{plain_sample.target} -OR- {MANUAL_GRADING_CORRECTED_TARGETS[qid]}"
            )
            assert corrected_sample.target == expected
        else:
            assert corrected_sample.target == plain_sample.target
