"""End-to-end smoke test: run a single CharXiv sample through the task.

Both the solver model and the grader are inspect's ``mockllm`` provider, so no
API keys are needed. The test does download the CharXiv validation split from
Hugging Face (about 67 MB) on first run.
"""

import itertools
from pathlib import Path

import pytest
from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from inspect_charxiv import charxiv

GRADER_RESPONSE = '{"extracted_answer": "mock answer", "score": 1}'


@pytest.mark.dataset_download
def test_charxiv_single_sample(tmp_path: Path) -> None:
    grader = get_model(
        "mockllm/model",
        custom_outputs=itertools.repeat(
            ModelOutput.from_content("mockllm/model", GRADER_RESPONSE)
        ),
        memoize=False,
    )

    [log] = eval(
        charxiv(),
        model="mockllm/model",
        model_roles={"grader": grader},
        limit=1,
        log_dir=str(tmp_path),
        display="none",
    )

    assert log.status == "success", log.error
    assert log.samples is not None
    assert len(log.samples) == 1

    sample = log.samples[0]
    assert sample.metadata["is_descriptive"] is True
    assert sample.scores is not None
    assert sample.scores["charxiv_scorer"].value == "C"

    assert log.results is not None
    assert log.results.scores[0].metrics["accuracy"].value == 1.0
