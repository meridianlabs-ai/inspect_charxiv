# CharXiv: Realistic Chart Understanding in Multimodal LLMs

[CharXiv](https://arxiv.org/abs/2406.18521) This is an inspect-native implementation of the CharXiv eval used to evaluate the extent that Multimodal LLMs can identify and interpret data from charts. Each problem has an associated chart image which is required to solve the problem. The dataset is made up of 5,000 questions from 1,000 charts across 8 different fields of study. Each chart uses real-world data, and questions pertain to either evaluating the extent that the model can identify the chart elements (descriptive) or the extent that a model can interpret the information present in the chart (reasoning). There are 19 types of descriptive questions with 4 of the 19 being asked for each of the 1,000 charts. Reasoning questions are unique, but come with 4 distinct answer types: text-in-chart, text-in-general, number-in-chart, and number-in-general.

Contributed by [@jchengjr](https://github.com/jchengjr)

## Usage

### Installation

Latest development version:

```bash
pip install git+https://github.com/meridianlabs-ai/inspect_charxiv.git
```

If you are using Inspect CharXiv in its repository, start by installing the necessary dependencies with:

```bash
uv sync
```

### Running evaluations

Now you can start evaluating models. If you are not using `uv` to manage dependencies in your own project, you can use the same commands with `uv run` dropped.

```bash
uv run inspect eval inspect_charxiv/charxiv --model openai/gpt-5-nano
```

You can also import tasks as normal Python objects and run them from python:

```python
from inspect_charxiv import charxiv
from inspect_ai import eval
eval(charxiv(subset="reasoning"))
```

After running evaluations, you can view their logs using the `inspect view` command:

```bash
uv run inspect view
```

For VS Code, you can also download [Inspect AI extension for viewing logs](https://inspect.ai-safety-institute.org.uk/log-viewer.html).

If you don't want to specify the `--model` each time you run an evaluation, create a `.env` configuration file in your working directory that defines the `INSPECT_EVAL_MODEL` environment variable along with your API key. For example:

```bash
INSPECT_EVAL_MODEL=anthropic/claude-opus-4-1-20250805
ANTHROPIC_API_KEY=<anthropic-api-key>
```

## Options

You can control a variety of options from the command line. For example:

```bash
uv run inspect eval inspect_charxiv/charxiv --limit 10 --sample-shuffle
uv run inspect eval inspect_charxiv/charxiv --max-connections 10
uv run inspect eval inspect_charxiv/charxiv --model-role grader=openai/gpt-4o
```

See `uv run inspect eval --help` for all available options.

## Parameters

### `charxiv`

- `subset` (`Literal["descriptive", "reasoning"] | None`): Which subset of the data to use based off the types of question in the sample. If no subset is specified, will run with both descriptive and reasoning questions. (default: `None`)

- `category` (`Literal["econ", "math", "physics", "q-bio", "cs", "eess", "q-fin", "stat"] | list[Literal["econ", "math", "physics", "q-bio", "cs", "eess", "q-fin", "stat"]] | None`): Which category or categories to include in the eval. Categories are based off the 8 fields of study for the data represented in the charts used for samples. If no categories are specified, run will include all categories. (default: `None`)

- `apply_corrections` (`bool`): The current CharXiv dataset on HuggingFace contains some typographical errors in the targets that make certain samples impossible to score correctly. If `apply_corrections` is `True`, these unwinnable targets will be appended with a corrected version and an indicator showing the grader model that either answer is acceptable. (default: `True`)

## Dataset

There are two types of questions in CharXiv: descriptive and reasoning. If no subset is specified, the eval will run using both types of questions.

Here is an example from the dataset of a sample with a descriptive question:

```text
**Question:** For the current plot, what is the total number of explicitly labeled ticks across all axes?

**Image:**

![Image for example question](example.png)

**Target:** "14"
```

Here is an example from the dataset of a sample with a reasoning question:

```text
**Question:** Based on plot (a), should we say that the spin dimer correlations depend on r linearly, logarithmically, or exponentially?

**Image:**

![Image for example question](example.png)

**Target:** "exponentially"
```

### License and attribution

"Code in this repository is MIT-licensed. Question/answer content from the CharXiv dataset — including the corrected targets in constants.py — is CC BY-SA 4.0, © Zirui Wang et al. Prompt templates and grading rubrics are adapted from the CharXiv repository under Apache-2.0. Chart images remain under the copyrights of the original arXiv paper authors and are fetched from the official Hugging Face distribution at runtime."


## Scoring

Inspect charxiv uses LLM as a judge for grading. The model used for this can be specified using `--model-role` `grader=`[model-provider]/[model-name]. The default model is "openai/gpt-4o". This grader model is asked to compare the sample's target to the model's response and given grading instructions to inform this comparison. If the grader model evaluates them as equivalent, the question is given a score of 'C' for correct. Otherwise, the question is given a score of 'I' for incorrect. The grading instructions given to the grader model depend on the question type. For descriptive questions, each of the 19 possible questions have grading instructions specific to what the sample's question was. For reasoning questions, there are 4 different instruction templates that will be employed based on a contingency matrix of whether the question is asking for a number or text, and whether or not the answer can be found in the chart or not.

## Evaluation Report

### Implementation Deviations

Each sample contains a single question-answer pairing with multiple questions referencing the same chart. This is a deviation from the original CharXiv experiment which used batched sorting by images. Singleton sorting was chosen to better leverage the tools available in the inspect framework.

The inspect implementation of CharXiv uses singleton sorting in the grading as well where the original batched the questions by grading instruction templates in order to conserve tokens. Singleton sorting was chosen to ensure grade determinism and independence at the expense of additional tokens.

Due to the swap to singleton sorting in grading, the instructions provided to the grader model were reformatted to make more sense for the desired output.

Some of the JSON keys used in the grading instruction examples were inconsistent with the originally specified output formatting and were therefore changed to avoid a possible instrument failure.

Opted to fix a typographical error in the response instructions for reasoning response questions with answer-in-chart as it risked creating instrument failures: "exlicitly" -> "explicitly".

Opted to fix a typographical error in the grading instructions as it risked creating instrument failures: "interger" -> "integer".

Grading instruction altered to accomodate the possibility of a multiple answer target in the case of a corrected target.

## Citation

'''bibtex
@article{wang2024charxiv,
  title={CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs},
  author={Wang, Zirui and Xia, Mengzhou and He, Luxi and Chen, Howard and Liu, Yitao and Zhu, Richard and Liang, Kaiqu and Wu, Xindi and Liu, Haotian and Malladi, Sadhika and Chevalier, Alexis and Arora, Sanjeev and Chen, Danqi},
  journal={arXiv preprint arXiv:2406.18521},
  year={2024}
}
'''

## Changelog

### [2-A] - 2026-09-08

- Fixed categories to no longer consider "physics" within the category "cs" due to substring matching.

- Altered grading instructions to reference JSON keys consistent with the desired output in the examples provided to the grader model.

- Fixed a bug where a target with a decimal place followed by a 0 would ask the model for an exact integer rather than a single decimal place.

### [1-A] - 2026-09-04

- Changed name of "apply_corrections" task parameter (previously was "correct_targets").