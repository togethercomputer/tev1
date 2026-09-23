# Dataset v3 — classification-focused continuation of 4B v2.1

Prepared September 21, 2026. This is a new training recipe, not an evaluated or trained checkpoint. Continue from the current **4B v2.1** weights; do not describe this as a replacement base model.

## Upload files

- Training: `data/v3/instruction/train.jsonl` — **15,843 examples**, **7,378,734 tokens** per epoch.
- Validation: `data/v3/instruction/dev.jsonl` — **4,068 examples**, **2,109,972 tokens**.
- Reserved evaluation: `data/v3/records/holdout.jsonl` — **500 examples**, never used in mining or training. Do not upload it as validation.

The longest training example is 1,525 tokens; validation maximum is 1,519. Use a 2,048-token sequence limit. Instruction exports include the same non-thinking Qwen chat template and letter plus EOS completions used previously. Do not apply another chat template. Use completion-only loss (`train_on_inputs=false`). Regex/logprobs are inference settings and do not belong in training examples.

## What changed

| Component | Examples | Purpose |
|---|---:|---|
| Fresh public classification | 2,400 | 666 current-model mistakes plus 1,734 correct examples, with original source labels |
| 24-option banking variants | 813 | 650 wider candidate sets plus 163 paired cases with the gold intent removed and none correct |
| Controlled inference contrasts | 600 | 100 ledger groups, each with six claims distinguishing support, contradiction, and missing details |
| Controlled news contrasts | 24 | 12 authored business-versus-technology pairs |
| Group-preserving v2.1 replay | 12,006 | Preserve policy, routing, research, reading, intent, news, and sentiment behavior |
| **Total** | **15,843** | |

Replay includes 4,002 policy, 2,004 priority-routing, 2,000 research, 1,500 MNLI, 800 BoolQ, 700 banking, 500 news, and 500 sentiment examples. The high replay share is intentional for continuation: the classification additions should not erase the broader skills learned in v2.1. This smaller dataset is not intended to retrain an untuned base model from scratch.

Validation combines 500 fresh classification examples (100 per source) with all 3,568 prior v2.1 development examples to watch regressions. The 500-example holdout was partitioned and hash-frozen before model mining. It comprises unused records from upstream training partitions, after excluding all previous dataset states and every official upstream held-out state. It is fresh to this fine-tuning recipe, not necessarily absent from base-model pretraining and not a new-domain evaluation.

## Evidence and quality controls

See [classification error analysis](evaluation/v3-analysis/README.md). The current model scored 4,000 fresh training candidates using regex plus logprobs. Its predictions select examples; they never become gold labels. Error selection is capped at half of each source's fresh subset. Ten manually spot-reviewed questionable examples are quarantined, not relabeled. All remaining source labels have not been manually adjudicated.

Banking distractors incorporate semantic confusion pairs observed in the old evaluation, with lexical neighbors and other options. No old evaluation passage is copied into training. Group variants stay together. Correct labels for the 24-option versions derive from the upstream 77-class label; absent-gold labels follow the candidate-set construction. Some candidate names overlap semantically, so this construction is not a guarantee of unambiguous natural-language annotation.

The 600 inference contrasts are template-generated variations of 100 groups, not 600 independent human-authored reasoning problems. The 24 news examples are authored fictional contrasts. These controlled exercises target a specific distinction and are a small part of training; they do not demonstrate broad unseen-policy or research-paper generalization. No new research-paper examples or judge labels are imported.

`validate_v3.py` passed checks for file hashes, frozen holdout preservation, source gold preservation, candidate mappings, an independently checked inference oracle, group/state/prompt split separation, exclusion of all previous evaluation states, chat/instruction alignment, and token limits. Original public-source licensing and attribution in `DATA_SOURCES.md` still apply; no blanket relicensing is implied.

## Suggested next run

One continuation epoch is a reasonable first experiment, with peak learning rate **0.00001**, cosine decay, warmup ratio **0.03**, packing enabled, and maximum sequence length **2048**. Keep compatible LoRA rank/alpha/modules when continuing the existing adapter. These are conservative starting settings, not tuned optimal values. Save multiple checkpoints and select using development accuracy by source and regression slices, not aggregate validation loss alone.

Compare with 4B v2.1 under the same regex/logprobs inference settings. Evaluate the selected checkpoint and baseline on the frozen holdout only after model selection; do not turn holdout errors into another round of v3 training. The 100-example per-source holdout is modest, so report counts and uncertainty rather than claiming small percentage differences are definitive. The older 1,000/300 and paper benchmarks remain feedback-informed development comparisons.

## Reproduction and audit files

Scripts: `prepare_v3.py` freezes candidate/dev/holdout partitions from pinned cached sources; `mine_v3.py` scores only candidates (4,000 billed calls, requires httpx and TOGETHER_API_KEY); `v3_contrasts.py` defines authored examples; `build_v3.py` samples and exports; `validate_v3.py` verifies exports offline. Preparation refuses an existing v3 directory, mining refuses an existing result file, and the builder refuses an existing final manifest. Preserve this completed build when reproducing in another checkout.

`data/v3/manifest.json` records file hashes, source/tokenizer pins, build-code hashes, counts, mining statistics, and token totals. `holdout-lock.json` freezes the initial partition hashes; `mining.jsonl` stores predictions; `quarantine.json` records manual exclusions. These generated files follow the repo's existing ignored-data convention. The authored builder scripts, analysis, and recipe documentation are source-controlled candidates, left uncommitted for review.
