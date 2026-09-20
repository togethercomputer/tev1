# Original $4 run: Command-line recipe notes

The [article](../articles/how-to-train-your-own-jev-model-for-4-dollars.md)
uses the Together CLI to upload and inspect files and retrieve job status,
then `curl` to submit the fine-tuning request. The historical job was launched
through Together's interface. These commands reconstruct that configuration;
they have not been used to launch another paid training run.

## Why job submission uses curl

In [Together CLI 2.23.0](https://pypi.org/project/together/2.23.0/),
`together/lib/cli/api/fine_tuning/create.py` treats a LoRA rank equal to its
default value, 8, as unspecified. With `--lora`, it replaces that rank with the
model's maximum, even if the caller explicitly passes `--lora-r 8`.

On September 20, 2026, the read-only model-limits response for
`Qwen/Qwen3.5-2B` reported a maximum LoRA rank of 64. A shortened `tg fine-tuning
create --lora --lora-r 8` command would therefore change the recorded recipe.
The REST request sets rank 8 and alpha 16 directly. CLI 2.24.0 source was also
checked and retained this behavior. Recheck before switching the example to a
newer CLI.

## Parameters and omitted defaults

The article includes job identifiers and the explicit training choices needed
for this run. It omits the unchanged SFT/input-loss defaults (`sft`, `auto`),
packing (`true`), early-stopping patience/warmup (2/1), and the model's default
gradient accumulation (1). LoRA targets are `all-linear`. The cosine schedule is
explicit because the direct REST request does not pass through the CLI's
scheduler construction.

The historical run recorded `train_on_inputs="auto"`; its resolved loss mask
was not independently inspected. Do not describe it as a verified explicit
completion-only-loss run.

For the full settings rather than the shortened article version, see
[examples/train_together.py](../examples/train_together.py). It previews the
configuration by default, and only uploads and starts a billed job with
`--launch`. [release/training.json](../release/training.json) holds sanitized
job metadata, and [training-data-verification.json](../release/training-data-verification.json)
records the uploaded train/dev files' matches to the local exports.

Follow the [Files API ingestion checks](https://docs.together.ai/docs/fine-tuning/quickstart)
before submitting, and consult the [create-job API reference](https://docs.together.ai/reference/post-fine-tunes)
if using a different API or CLI version. The file IDs belong to your account;
the example placeholders must be replaced. Repeated submissions create separate
billed jobs.
