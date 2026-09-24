# Contributing

Use Python 3.12+ and `uv sync --locked`. Run
`uv run python -m unittest -v` before submitting changes. Follow the
[dataset guide](docs/DATASET.md) for the separate data-build and validation steps.

The supported workflow is tev1-4B-experimental on Qwen3.5-4B using `data/new-v1`.
The v1, v2, and v2.1 builders are intermediate dependencies of that recipe.
Keep their splits, option ordering, label semantics, and reproducibility intact.
Write new experiments to fresh output directories and preserve the frozen
`runs/new-v1` evidence.

Include a failing example for decision or data bugs. Do not commit private
documents, API keys, model binaries, generated datasets, or source caches.
Contributed data needs provenance and license information. Once test errors
inform training changes, reserve a new final holdout.
