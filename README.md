# tev1-4B-experimental

We fine-tuned **Qwen3.5-4B on Together AI** to make decisions: give it context,
a question, and 2–24 options, and it returns one answer letter. We're calling
it **tev1-4B-experimental**. A separate blog post will cover the project, how we
fine-tuned it, and what we learned.

This repo contains the data recipe, training example, and saved results so you
can fine-tune your own decision model.

```text
State       Returns are allowed within 30 days. This purchase was 12 days ago.
Question    Is this return within the allowed window?
Options     A: Yes   B: No   C: Not enough information
Answer      A
```

## The 4B run

The latest recipe is called **new v1** in the files. It starts from base
`Qwen/Qwen3.5-4B` and combines **37,840 unique training examples** with
**4,568 validation examples**. The mixture covers language classification,
policy decisions, routing, and synthetic research classification. We use ordinary
LoRA supervised fine-tuning with Qwen's existing language-model output head.

- Checkpoint: `hassan/Qwen3.5-4B-v1-new-69617472-bdc3c2fc`
- Evaluated endpoint: `together/Tev1-4B-experimental`
- Saved results: **880/1,000** main decisions and **300/300** policy-transfer decisions.

These are reused development benchmarks, not untouched final tests. Endpoint
access depends on your Together account. See the [run record](runs/new-v1/README.md)
for evidence and limitations. The saved dataset has the counts above; uploaded-file
identity and the job's exact settings still need verification. The training example
uses the saved starting recipe: rank 8, one epoch, learning rate `5e-5`, and a
2,048-token sequence limit.

## Fine-tune it

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).
Run these commands from the repository root:

```bash
uv sync --locked
uv run python fetch_sources.py
uv run python build_all.py
```

`build_all.py` runs the four builders in dependency order.
The older builder names remain because each is a dependency of the final dataset.
They build local intermediate data; **new v1 is the dataset to train on**.
Builders refuse existing output directories. See the [dataset guide](docs/DATASET.md)
for validation, provenance, and how to adapt the recipe.

Preview the training settings, then launch when ready:

```bash
uv run python examples/train_together.py
export TOGETHER_API_KEY='your-key'
uv run python examples/train_together.py --launch
```

`--launch` uploads `data/new-v1/instruction/train.jsonl` and `dev.jsonl` and starts
a billed training job. Save the returned job ID and follow it in Together;
rerunning the command starts another job. For your own data, use `--data path/to/instruction`;
use `--settings path/to/settings.json` to override recipe settings.
See the [training guide](docs/TRAINING.md) for the input format and deployment step.

## Try your model

Once you have deployed your fine-tune on Together:

```bash
export TOGETHER_MODEL='your-deployed-endpoint'
uv run python examples/decide.py examples/charge-dispute.json
```

If you copied `.env.example` to `.env`, use `uv run --env-file .env` to load it.
`JEV_MODEL` is also supported as an alias for `TOGETHER_MODEL`. More input examples
cover return policies, yes/no comprehension, and sentiment in `examples/`.

The example disables thinking and constrains the answer to the supplied option
letters. It returns the selected letter, semantic key, and token logprobs.
Logprobs are model preferences, not calibrated confidence.

Evaluate your endpoint on labeled records:

```bash
uv run python scripts/evaluate.py --provider together \
  --model "$TOGETHER_MODEL" --records data/v2.1/records/test.jsonl \
  --output outputs/my-evaluation
```

Use a fresh holdout once evaluation results inform your training changes.

## License and contributions

Code and original documentation are [MIT licensed](LICENSE). Third-party datasets
and model weights have their own terms; see [source provenance](DATA_SOURCES.md).
Training data and model weights are not bundled. This is an independent,
Jev-inspired implementation and does not use Jev's answers as training labels.

Run `uv run python -m unittest -v` before contributing. See [CONTRIBUTING.md](CONTRIBUTING.md).
