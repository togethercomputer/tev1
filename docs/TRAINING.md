# Fine-tuning on Together

`examples/train_together.py` previews a starting recipe for base
`Qwen/Qwen3.5-4B`. These are the settings proposed alongside new v1, not a
verified export of the historical job's settings.

| Setting | Value |
|---|---|
| Method | LoRA SFT, all-linear modules |
| Rank / alpha / dropout | 8 / 16 / 0 |
| Epochs / batch size / gradient accumulation | 1 / 8 / 1 |
| Learning rate | 0.00005 |
| Scheduler | Cosine; minimum LR ratio 0; 0.5 cycles |
| Warmup / gradient norm / weight decay | 0.03 / 1 / 0 |
| Sequence length / packing | 2,048 / enabled |
| Checkpoints / evaluations | 3 / 6 |
| Early stopping / seed | Disabled / 42 |
| Train on inputs | False; completion-only loss |

Install the locked SDK with `uv sync --locked --extra train`. Preview with
`uv run --extra train python examples/train_together.py`; append `--launch` to
upload and start a billed job. The script waits for both uploaded files to finish
processing before submission. It uses the Python SDK directly to set LoRA rank.
See Together's [fine-tuning API reference](https://docs.together.ai/reference/post-fine-tunes).

To change settings, pass a JSON object via `--settings`, for example:

```json
{"suffix": "my-decision-model", "learning_rate": 0.00005, "n_epochs": 1}
```

To use your own data, pass `--data path/to/instruction`. That directory must
contain `train.jsonl` and `dev.jsonl`. Each line is a JSON object with a `prompt`
and `completion`; the prompt is already rendered using the model's chat template,
and the completion is the answer letter plus EOS. Do not apply a second chat
template. The [dataset guide](DATASET.md) describes how to produce this format.

Keep the system instruction, option schema, and non-thinking mode consistent
between training and inference. When changing base model or tokenizer, regenerate
the exports and verify token lengths against the selected sequence limit.

Save the returned job ID and inspect progress in Together. After training,
deploy the resulting model to an endpoint available to your account and set
`TOGETHER_MODEL` to that endpoint ID. The training script does not provision
inference hosting. Test with `examples/decide.py`; use `--dry-run` to inspect the
request before calling the endpoint. Inference and hosting have separate charges.

## Follow the job and deploy from the CLI

If using a `.env` file, load it explicitly for each command. Use the locked
`train` extra so the Python example and `tg` use the same SDK:

```bash
uv sync --locked --extra train
cp .env.example .env
# Edit .env to set TOGETHER_API_KEY, then preview or launch:
uv run --extra train --env-file .env python examples/train_together.py
uv run --extra train --env-file .env python examples/train_together.py --launch
```

Replace `YOUR_JOB_ID` below with the ID printed by that launch. Use the same ID
for status and output-model lookup:

```bash
uv run --extra train --env-file .env tg fine-tuning retrieve YOUR_JOB_ID
uv run --extra train --env-file .env tg fine-tuning retrieve YOUR_JOB_ID --json
```

Once training completes, copy `model_output_name` from the JSON. In the next
command, replace `MODEL_OUTPUT_NAME` and choose hardware available to your account
that supports your model. Creating the endpoint starts separately billed hosting:

```bash
uv run --extra train --env-file .env tg endpoints create MODEL_OUTPUT_NAME \
  --hardware 1x_nvidia_h100_80gb_sxm --display-name tev1-4B-experimental --wait
```

Set `TOGETHER_MODEL` in `.env` to the returned endpoint **name**, then query it:

```bash
uv run --env-file .env python examples/decide.py examples/charge-dispute.json
```

The blog's `JEV_MODEL` variable also works when `TOGETHER_MODEL` is empty.
The model returns one letter; the client maps it to a semantic key and prints
JSON with `label`, `key`, and `logprobs` (null if absent). It does not train the
model to emit that JSON object.

Stop your dedicated endpoint when finished, using its **ID**:

```bash
uv run --extra train --env-file .env tg endpoints stop YOUR_ENDPOINT_ID
```

Training cost and duration vary with settings and provider capacity. The original
job's price and runtime have not been verified in this repo; a training-only
estimate does not include dedicated hosting or inference.
