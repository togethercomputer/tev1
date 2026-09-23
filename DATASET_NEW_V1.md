# New v1 — combined v1 + v2.1 training set

A fresh-start experiment for the original `Qwen/Qwen3.5-4B` model. This is not a continuation from a fine-tuned v2.1/v3.1 checkpoint. No training job has been launched.

## Upload files

- `data/new-v1/instruction/train.jsonl`: **37,840 examples**, **16,929,529 tokens** per epoch.
- `data/new-v1/instruction/dev.jsonl`: **4,568 validation examples**, **2,218,673 tokens**.

The inputs contained 43,840 rows (v1 10,000 + v2.1 33,840). Exactly 6,000 were duplicate replay examples, removed by identical decision prompt with checked agreement of labels and exported completions. The resulting union preserves all unique examples once. Shuffling seed: 42. Validation is the union of the original v1 and v2.1 development sets; no duplicates there. No test, calibration, transfer, or v3 holdout examples enter training. No v3/v3.1 training augmentation is included.

| Training source | Rows |
|---|---:|
| MNLI | 5,000 |
| BoolQ | 3,000 |
| Banking77 | 3,000 |
| AG News | 1,500 |
| SST-5 | 2,000 |
| Original synthetic policies | 1,500 |
| v2 synthetic policies | 12,000 |
| v2 priority routing | 6,000 |
| v2.1 synthetic research | 3,840 |
| **Total** | **37,840** |

This is 14,500 public-data rows and 23,340 synthetic rows. It is a union, not a recreation of the sequential training schedule: the replay weighting and optimization trajectory differ. Existing label limitations, synthetic template correlations, licenses, and evaluation exposure remain unchanged. See DATA_SOURCES.md and DATASET_V21.md.

## Recommended starting parameters

These are proposed experiment settings, not an optimized recipe or a guarantee of improvement. Keep rank at 8 for this first fresh-start comparison; a rank-16 experiment can follow with other settings held fixed.

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen3.5-4B — original model, no fine-tuned parent |
| Method | SFT |
| Training type | LoRA |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| Trainable modules | all-linear |
| Epochs | 1 |
| Learning rate | 0.00005 |
| LR scheduler | cosine |
| Min LR ratio | 0 |
| Scheduler cycles | 0.5 |
| Warmup ratio | 0.03 |
| Batch size | 8 (model-supported fixed value) |
| Gradient accumulation steps | 1 |
| Max gradient norm | 1 |
| Weight decay | 0 |
| Sequence packing | On |
| Maximum sequence length | 2048 |
| Checkpoints | 3 |
| Evaluations | 6 |
| Early stopping | Off for this one-epoch comparison |
| Random seed | 42 |
| Train on inputs | False — completion-only loss |
| Suffix / run name | jev-new-v1 |

The LR is higher than the 1e-5 continuation proposal because this run starts from original weights. It is still a starting hypothesis, not the result of an LR sweep. Select checkpoints on development accuracy and regressions; do not select solely on token loss. Decide whether to add another epoch only after measuring this run.

The instruction files already contain the non-thinking Qwen chat template and answer letter plus EOS. Do not apply another chat template. All sequences are at most 1,526 tokens, so 2,048 covers them. Regex and logprobs are inference settings, not fine-tuning parameters.

Six evaluations process the validation set six times. At one epoch the planned training-plus-validation token volume is 30,241,567; actual provider accounting may differ. This is a token estimate, not a price quote.

## Verification and reproduction

`build_new_v1.py` is an offline merge, not an example generator. It refuses an existing output directory. It checks label conflicts, source export preservation, split isolation and exclusion of normalized states from all previous non-training splits. It independently verifies the prompt template, answer/EOS completion, and token lengths using the same pinned Qwen tokenizer used by the source datasets.

`data/new-v1/manifest.json` contains input/output hashes, counts and duplicate statistics. `membership.json` maps each retained prompt to its source versions/IDs. `validation.json` records successful checks. Training settings are saved separately in `data/new-v1/training-settings.json` as a human-readable configuration; it is not an API request body.
