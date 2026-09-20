# Dataset recipe

Dataset starter for fine-tuning Qwen3.5-2B into a single-token decision model.
This page describes data generation; see the root README for the trained model and inference tools.
It uses five public datasets plus deterministic, programmatically labeled policy cases.

## Built dataset

| Source | Train | Dev | Calibration | Test |
|---|---:|---:|---:|---:|
| MultiNLI | 2,500 | 250 | 250 | 250 |
| BoolQ | 2,000 | 200 | 200 | 200 |
| Banking77 | 2,000 | 200 | 200 | 200 |
| AG News | 1,000 | 100 | 100 | 100 |
| SST-5 | 1,000 | 100 | 100 | 100 |
| Synthetic policy | 1,500 | 150 | 150 | 150 |
| **Total** | **10,000** | **1,000** | **1,000** | **1,000** |

`policy_transfer` adds 300 examples from two rule structures absent from training.
It measures compositional transfer within the synthetic policy domain, not broad
out-of-domain capability. Public benchmark exposure during base-model pretraining
is unknown.

The first build contains 2,160,836 training tokens including prompt and end token.
The median training example is 207 tokens; the longest example across all splits
is 1,370 tokens. Nothing is truncated; the build fails above the 2,048-token budget.

## Files to use

- `data/v1/instruction/train.jsonl`: **recommended training export**. Exact Qwen
  non-thinking chat prefix in `prompt`, answer letter plus `<|im_end|>` in `completion`.
  Use Together's instruction format and completion-only loss (`train_on_inputs=False`).
  Do not apply a second chat template to this export.
- `data/v1/instruction/dev.jsonl`: corresponding development/validation export.
- `data/v1/sft/*.jsonl`: standard `messages` format, for servers with verified
  non-thinking template handling. The messages alone do not encode
  `enable_thinking=False`; this is why the pre-rendered export is provided.
- `data/v1/records/*.jsonl`: full records, semantic answers, source revisions,
  original row IDs, group IDs, transformations, and token counts. Keep these for
  audits/evaluation; do not upload as conversational SFT data.
- `data/v1/manifest.json`: source lock, file hashes, counts, label distributions,
  token lengths, and integrity checks.
- `data/v1/samples.json`: three training examples per source for inspection.
- `sources.lock.json`: immutable Hugging Face dataset/tokenizer revisions.
- `uv.lock`: pinned Python environment.

The evaluator-facing records and both training exports have identical row order
within each split. The model sees no source IDs, split names, or ground-truth metadata.
The tokenizer was downloaded; model weights were not.

## Reproduce

```bash
cd open-jev
uv sync --locked
uv run python fetch_sources.py
uv run python -m unittest -v
uv run python build_dataset.py --output data/v1-rebuild
uv run python validate_dataset.py data/v1-rebuild
```

Source revisions are resolved only if missing from `sources.lock.json`; existing
pins are reused. Source fetching uses the public Hugging Face Hub and project-local
`.cache/`. Once cached, add `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` before build
commands to build without network access. Builds refuse to overwrite output folders.

To check the delivered data without downloading anything:

```bash
uv run python validate_dataset.py data/v1
```

## Task conversion

- MultiNLI: premise in state, hypothesis in question; original classes mapped to
  supported / insufficient information / contradicted.
- BoolQ: passage plus yes/no question. No fabricated unknown labels.
- Banking77: 4, 6, or 8 candidate intents, with up to two lexically similar hard
  negatives and additional random negatives. About 15% replace the true class
  with a correct none-of-the-above; another 15% add an incorrect none distractor.
  This is **candidate-set classification, not a full 77-way benchmark**.
  None labels assume the dataset's intent taxonomy is mutually exclusive.
- AG News: four original topic classes.
- SST-5: five ordered sentiment levels; level order is preserved.
- Policy: four training rule trees over numerical thresholds and a boolean field,
  plus two unseen transfer trees. Each group contains a positive record, a
  one-fact-flipped negative, and the same record with that decisive fact omitted.
  Missing-data labels enumerate all completions rather than treating missing as false.
  These are artificial rules to follow as written, not real retailer policies.

Categorical option order is shuffled deterministically and correct labels are
remapped. Labels A–H are verified as single tokens both alone and at the actual
Qwen non-thinking assistant boundary. Score levels use their canonical order.
The instruction completion adds a termination token after the single answer token.

## Splits and evaluation

The seed is `20260920`. Development and calibration examples come from upstream
training splits; test examples come from official held-out splits (BoolQ validation;
MultiNLI matched + mismatched validation; the other sources' test splits).

We reserve all upstream held-out normalized states before selecting training data,
then retain at most one record per normalized state. This keeps repeated premises
and passages together. All synthetic siblings stay in one partition. Validation
rejects cross-split group, normalized-state, and exact-prompt overlaps.

Deduplication normalizes Unicode, case, punctuation, and whitespace. It does not
prove the absence of semantic paraphrases or pretraining contamination. Original
text and source labels are retained, including upstream annotation noise.

Sampling balances original classes within each source where possible. Resulting
calibration measures apply to this artificial mixture; they are **not production
calibration guarantees**. Fit on calibration only, select models on dev, and reserve
test + policy_transfer for final assessment. Fit deployment calibration on data
with the actual application's distribution.

Recommended comparisons: untuned vs fine-tuned accuracy by source; NLL/Brier/ECE;
answer changes under option reordering; errors above a chosen confidence threshold;
paired-policy accuracy (all three siblings); local quantized vs original precision.
For policy-level uncertainty estimates, resample groups rather than individual siblings.

This document describes the dataset recipe, not the training provenance of any specific checkpoint. See ../MODEL_CARD.md.

## Attribution and scope

See [DATA_SOURCES.md](../DATA_SOURCES.md) for source links and recorded license metadata. Code and
dataset licensing are distinct: the combined dataset is not assigned a blanket
open-source license. Generated files and source caches are excluded from Git.

The mixture is inspired by [Kev's original recipe](https://github.com/jaredpalmer/kev/blob/main/MODEL_CARD.md#training-data),
adapted to standard token-target SFT rather than Kev's pointer head.
[Fireworks' classifier tutorial](https://fireworks.ai/blog/Finetuning-LLMs-as-Classifiers)
motivates token labels. [Together's format documentation](https://docs.together.ai/docs/fine-tuning/data-preparation)
describes the exported instruction/chat schemas.
