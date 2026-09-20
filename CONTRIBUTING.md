# Contributing

Use Python 3.12+ and `uv sync --locked`. Run `uv run python -m unittest -v`
before submitting changes. Dataset validation is a separate step:
`uv run python validate_dataset.py data/v1` after building or downloading sources.

Keep dataset versions and checkpoint results separate. New training data belongs
in a new output directory; never replace the frozen `evaluation/v1` evidence with
results from a different checkpoint. Describe changes to rules, option ordering,
splits, prompts, or label semantics in the pull request.

Include a small failing example for decision or data bugs. Do not include private
documents, API keys, model binaries, or source caches. Submit proposed datasets
with provenance and license information. The code license does not relicense
third-party training data or model weights.

Useful contributions include base-Qwen comparisons, missing-information cases,
option-order robustness, local inference verification, and calibration measured
on a held-out split. Avoid selecting the next model against the published test
set; once its failures guide training, use a fresh final holdout.
