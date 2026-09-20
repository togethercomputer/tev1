# Dataset v2.1 — use for the second training run

This revision combines policy improvements with feedback from the task “Test Jev model cost and success.” It replaces the v2 upload files for the next training run; the v2 dataset itself remains unchanged.

The paper experiment measured 579/891 (65.0%) agreement for the fine-tuned Qwen model and 813/891 (91.2%) for Jev. References are Kimi K3/GPT 6 Astra consensus, not human-verified labels. The experiment supplied 24 topic options, A–X. The original training data only supplied up to eight options. That is an exposure gap, not proof that option count alone caused the errors. The observed confusions also concern the main contribution versus methods mentioned: VLM architecture vs efficiency, agent training vs reasoning RL, and video generation vs related image/3D tasks.

## What changed

- All 30,000 v2 training examples are retained unchanged.
- Added 3,840 controlled research-classification exercises, representing 480 scenario groups. Each group contains two contribution/background role variants; each is presented with 24 canonical-order options, 24 shuffled options, 12 candidates, and eight candidates. All variants stay in the same split.
- Questions and all 24 topic definitions match the task's rubric. The paper titles, paper summaries, model predictions, and judge rationales/labels are not used as training examples.
- Added 768 development exercises, making the upload validation file 3,568 examples.
- Added 768 separately grouped research-challenge exercises with distinct rendering templates. The existing v2 calibration, test, and transfer records are unchanged.
- All answer labels A–X are verified to be single tokens at the actual non-thinking Qwen assistant boundary.

Training now contains **33,840 examples** and **16,050,303 tokens per epoch**. The longest example across the new build is 1,526 tokens, below the 2,048-token budget. The added research component is about 11.3% of examples, preserving the policy, general-language, and replay mixture rather than letting this one evaluation dominate training.

The fictional summaries distinguish the work being contributed from borrowed implementation ingredients, baselines, and related work. This includes reinforcement learning used to improve video generation; a multimodal architecture whose efficiency is a side benefit; and agent training that happens to use reinforcement learning. Gold labels are assigned from authored contribution/background construction, not inferred from keyword matching or obtained from Jev.

## Important limits

The 3,840 added rows are **controlled template and option-order augmentations**, not 3,840 independent real research papers. They use 96 authored contribution descriptions across 24 topics, plus implementation/background variations. Contribution vocabulary is shared across splits; the separate research challenge checks new renderings and scenarios, not broad out-of-distribution paper understanding. Templates can provide shortcuts. Real-paper evaluation remains essential before claiming improvement.

The 891 previously evaluated papers remain excluded from training. Because their aggregate failure patterns informed this revision, treat that comparison as a feedback-informed development benchmark. A broader claim of outperforming Jev needs another independently labeled real-paper set that has not guided training decisions.

The general v2 policy oracle, licensing notes, source locks, and evaluation precautions remain in DATASET_V2.md. Existing public-source license conditions continue to apply. No blanket dataset license is assigned.

## Upload

Use `data/v2.1/instruction/train.jsonl` and `data/v2.1/instruction/dev.jsonl` in ~/dev/open-jev. Convenient copies are supplied as `train.jsonl` and `validation.jsonl` in the v2.1 output folder.

Continue SFT from the existing fine-tuned checkpoint. Keep the initial recommendation of one epoch and learning rate 0.00002, completion-only loss, maximum sequence length 2048, and packing enabled. The instruction files already include the exact non-thinking chat template. Do not add another template. These are starting settings to evaluate, not optimized hyperparameters.

Use dev to compare policy, public-language, and research scores separately, including performance by answer position and candidate count. Select the checkpoint before opening reserved evaluations. Never assume an aggregate validation-loss improvement guarantees that all task families improved.

## Reproduce

`build_v21.py` reads the frozen v2 records and the local paper experiment config at `/Users/hassan/dev/1kpapers/experiments/qwen35-2b-jev/config.json` for taxonomy and question text only. Its hash is recorded. It makes no model calls. Run it from ~/dev/open-jev with the pinned .venv Python, choosing a new output directory via --output. `validate_v21.py` checks base-record preservation, hashes, option mappings, group partitions, all 24 token labels, and new export alignment.
