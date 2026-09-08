from typing import Literal

from inspect_ai.dataset import Dataset, RecordToSample, Sample, hf_dataset

from inspect_charxiv.utils import (
    convert_descriptive_question,
    convert_reasoning_question,
)

FieldOfStudy = Literal[
    "econ", "math", "physics", "q-bio", "cs", "eess", "q-fin", "stat"
]


# Ordinarily the hf_dataset function would be sufficient, but CharXiv uses a five-question-per-row batching method that is incompatible with inspect-ai's singleton sampling. This function produces an exploded single-question-per-row dataset instead.
def load_charxiv_dataset(
    subset: Literal["descriptive", "reasoning"] | None = None,
    category: FieldOfStudy | list[FieldOfStudy] | None = None,
    apply_corrections: bool = True,
) -> Dataset:
    """Load the CharXiv dataset from Hugging Face and return it as a Dataset object. If subset is None, include the entire dataset. If subset is "descriptive", include the descriptive questions only. If subset is "reasoning", include only the reasoning questions. If categories is not None, filter the dataset to only include records with a field_of_study in categories."""
    dataset = hf_dataset(
        path="princeton-nlp/CharXiv",
        split="validation",
        sample_fields=_make_record_to_sample(
            subset=subset, apply_corrections=apply_corrections
        ),
        revision="f441eb632fc62f6f777830a0f47619e6e86459b0",  # The latest commit to the CharXiv repo as of Aug 11, 2026.
    )
    if category is not None:
        categories: set[FieldOfStudy] = (
            {category} if isinstance(category, str) else set(category)
        )
        dataset = dataset.filter(lambda sample: _in_categories(sample, categories))
    return dataset


# record_to_sample factory to retain task parameters for record_to_sample function getting passed to hf_dataset.
def _make_record_to_sample(
    subset: Literal["descriptive", "reasoning"] | None = None,
    apply_corrections: bool = True,
) -> RecordToSample:
    """Return a sample_fields callable that converts a record to a list of Samples derived from the descriptive questions, reasoning questions, or both depending on the value of subset. If subset is None, return both descriptive and reasoning questions. If subset is "descriptive", return only descriptive questions. If subset is "reasoning", return only reasoning questions."""

    def record_to_sample(record: dict[str, str | int | None]) -> Sample | list[Sample]:
        samples: list[Sample] = []
        if subset in (None, "descriptive"):
            samples += [
                convert_descriptive_question(i, record, apply_corrections)
                for i in range(1, 5)
            ]
        if subset in (None, "reasoning"):
            samples.append(convert_reasoning_question(record, apply_corrections))
        return samples

    return record_to_sample


# helper function to be passed into categories filter when loading the huggingface dataset
def _in_categories(sample: Sample, categories: set[FieldOfStudy]) -> bool:
    field = (sample.metadata or {}).get("field_of_study")
    return isinstance(field, str) and field in categories
