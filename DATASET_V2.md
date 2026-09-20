# Continuation dataset v2

This dataset targets the observed policy-transfer and missing-information weaknesses of the first Qwen3.5-2B fine-tune. It preserves the same system prompt, single-letter answer target, tokenizer revision, and non-thinking assistant prefix. It contains no teacher-model answers and makes no claim to outperform Jev before the next checkpoint is evaluated.

## Training mixture

| Component | Examples | Purpose |
|---|---:|---|
| New executable eligibility policies | 12,000 | Nested AND/OR/NOT, exceptions, thresholds, categorical membership, at-least/exactly-k rules, and uncertainty |
| New priority-routing policies | 6,000 | First matching rule, competing routes, default route, and uncertain route |
| Fresh public examples | 6,000 | Natural-language inference, yes/no reading, intent, topics, and sentiment |
| Replay from v1 train only | 6,000 | Reduce forgetting of the original tasks |
| **Total** | **30,000** | **24,000 new examples + 6,000 intentional replay** |

Fresh public counts: MNLI 2,500; BoolQ 1,000; Banking77 1,000; AG News 500; SST-5 1,000. Replay counts: MNLI 2,000; BoolQ 1,500; Banking77 1,000; AG News 750; SST-5 750. These public examples use the same immutable source revisions and original source labels as v1. Existing public dataset license conditions still apply; see DATA_SOURCES.md. This collection has no blanket open-source license.

## Build statistics

The training export contains 11,589,360 tokens including prompts and termination tokens; median 398, p95 690, maximum 995. All splits fit the 2,048-token budget without truncation. New synthetic training data includes 6,000 unknown answers, 5,000 incomplete-but-decidable answers, 5,370 examples with distractor entities, and 3,624 examples with untrusted comments. There are 2,243 canonical rule shapes in synthetic training.

## Supervision

Each new eligibility-policy group has six related cases: eligible, one-fact-flipped ineligible, decisive fact omitted, partially observed but provably eligible, partially observed but provably ineligible, and multiple missing facts yielding uncertainty. Its three answer classes have equal counts. This explicitly prevents “a field is missing, therefore unknown” from being a successful general rule.

Each routing group covers all three ordered routes and the default, plus two uncertain cases. At least one of the certain cases has missing fields. Missing facts may not affect the winning route when an earlier rule already settles the outcome.

Rules use four to six named predicates over independently varying typed fields. Numeric operators include strict and inclusive comparisons and equality/inequality, with values sampled on and near boundaries. Categorical predicates include inclusion/exclusion. Negation supports exceptions. Every label comes from enumerating all completions consistent with the observed facts; a second bitset implementation checks the result after reconstructing predicates from the concrete record.

Six training domains cover returns, event admission, subscription perks, parcels, workspace access, and equipment booking. Two other domains are reserved for transfer evaluation. Two rendering styles use symbolic or verbal operators. Some cases include other entities, irrelevant comments containing instructions, and explicit nulls instead of omitted fields. These are controlled synthetic cases, not a comprehensive prompt-injection benchmark or real business policies.

## Splits

| Split | Public | Policy | Routing | Total | Use |
|---|---:|---:|---:|---:|---|
| train | 12,000 including replay | 12,000 | 6,000 | 30,000 | Continue training |
| dev | 1,000 | 1,200 | 600 | 2,800 | Upload as validation; select checkpoint |
| calibration | 1,000 | 1,200 | 600 | 2,800 | Fit confidence calibration after model selection |
| test | 1,000 | 1,200 | 600 | 2,800 | Fresh final comparison |
| transfer | 0 | 1,200 | 600 | 1,800 | Reserved domains and rule shapes |

All synthetic siblings remain in one split. All v1 dev, calibration, test, and policy_transfer states are excluded from v2. Fresh public rows also exclude v1 train. Official upstream held-out states are excluded from new public training/dev/calibration. New public test examples come from unused official held-out rows.

Transfer splits are chosen by a fixed hash of canonical syntactic rule shape, before evaluating a model. Canonicalization ignores leaf names and order for commutative operators. This guarantees syntactic-shape separation, not separation of logically equivalent Boolean functions. Transfer also changes domain names, so it measures the combined shift. Exact and normalized-state deduplication does not detect all semantic paraphrases or prior pretraining exposure.

The already-inspected v1 test and policy_transfer sets are now historical development benchmarks. They can track regression, but cannot support an unbiased claim that a model chosen to fix their failures beats Jev. Keep v2 test and transfer out of training and checkpoint selection; repeated inspection would similarly turn them into development benchmarks.

## Upload and training

Use data/v2/instruction/train.jsonl for training and data/v2/instruction/dev.jsonl for validation. These contain exact Qwen non-thinking prompt/completion pairs. Completion is the answer letter followed by the end-of-message token. Select instruction format, completion-only loss (train_on_inputs=false), and do not add another chat template. Standard messages-format copies are available in data/v2/sft if the training system's non-thinking template behavior has been verified.

Start from the existing fine-tune's checkpoint using Together's continued-fine-tuning workflow. Suggested first experiment: one epoch, learning rate 0.00002, cosine schedule, warmup ratio 0.03, maximum sequence length 2048, and sequence packing enabled. These are conservative experiment choices, not established optimal settings. Continue the existing training type/adapter configuration rather than silently starting from base Qwen. Use validation and checkpoint evaluation to decide whether more training is beneficial.

Compare per-source dev accuracy and missing-information accuracy, not just aggregate validation loss. Keep the old checkpoint if policy gains cause unacceptable regressions on language tasks. This dataset is intentionally policy-heavy, so an aggregate score can hide regressions.

After selecting a checkpoint on dev, compare it with the old checkpoint and Jev on the exact same v2 test and transfer records. Report each task, missing-but-decidable vs ambiguous cases, all-six-cases-correct group accuracy, and option-order consistency. Confidence should be calibrated separately on representative data. Bootstrap synthetic comparisons by group, not individual sibling rows. A win on these synthetic tasks is not evidence of being better than Jev on arbitrary production tasks.

Together references: [data preparation](https://docs.together.ai/docs/fine-tuning/data-preparation), [continued fine-tuning](https://www.together.ai/blog/continued-fine-tuning).

## Reproduce and validate

From ~/dev/open-jev:

```bash
.venv/bin/python -m unittest -v
.venv/bin/python build_v2.py --output data/v2-rebuild
.venv/bin/python validate_v2.py data/v2-rebuild
```

The builder works offline from the pinned v1 cache, refuses to overwrite an existing output folder, verifies labels and partition separation before exporting, and fails if any example exceeds 2048 tokens. It downloads no model weights and makes no inference calls. The manifest records source pins, builder hash, export hashes, token statistics, and integrity results. Records include auditable rule trees, concrete predicate definitions, group IDs, and generation variants; these metadata are excluded from the training prompts.
