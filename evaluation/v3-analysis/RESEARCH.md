# Research classification failures — latest 4B v2.1

Analyzed the completed run in the task **Test Jev model cost and success**, with model `hassan/Qwen3.5-4B-jev-2-1-4286f48b-c0ad7564`. The source is `qwen35-4b-jev-v21-c0ad7564/comparison.json` under the separate 1kpapers experiment. No paper benchmark was rerun here.

The model agrees with the two-model judge consensus on **767/891 (86.1%)**. Jev agrees on **813/891 (91.2%)**. Jev alone agrees on 77 cases; Qwen alone on 31. The 124 disagreements are not all proven model errors: the reference is model consensus, and several papers span multiple shelves.

| Reference topic → model prediction | Cases |
|---|---:|
| RL for reasoning → agent training | 8 |
| Agent skills → agent training | 6 |
| Search agents → agent training | 5 |
| Interpretability/analysis → reasoning methods | 4 |
| Search agents → agent benchmarks | 4 |
| VLM architecture → image generation | 3 |
| Research automation → agent training | 3 |
| Agent benchmarks → agent training | 3 |

The largest useful per-topic gaps include RL-for-reasoning (61/77 versus Jev 76/77), efficiency (26/35 versus 34/35), interpretability (17/25 versus 22/25), skills (26/33 versus 31/33), and VLMs (38/48 versus 44/48). We do not treat every ambiguous case as a clean distinction or train against judge rationales.

## Changes motivated by the errors

1. Distinguish a learning objective for reasoning from a curriculum or policy-learning loop for interacting agents.
2. Distinguish a reusable skill/memory artifact from general agent post-training, even when both use RL.
3. Follow the actual shelving rubric: a domain-specific benchmark belongs with its subject. Search and coding benchmarks should not automatically map to generic agent evaluation.
4. Distinguish analysis of reasoning from a proposed algorithm that improves reasoning.
5. Recognize efficiency/serving as the contribution when the underlying generator is unchanged, including video models.
6. Distinguish a general-purpose multimodal model from one capability demonstrated in its evaluation.
7. Distinguish scientific workflow assistance from a domain-specific scientific predictor.

New v3.1 scenarios use longer title/metadata/summary/limitation inputs, place contribution evidence before or after contextual material, and include a neighboring topic as background. All 24 topic definitions and the exact decision rubric are preserved. Each scenario appears in canonical and shuffled 24-option order. Eight focus topics receive more examples, while every other topic remains represented.

The new scenarios are authored fictional studies, with 48 training cores and 24 separately authored development cores. Multiple renderings and option permutations are correlated augmentations, not independent papers. This is closer to the benchmark's input format and topic boundaries, but not a representative sample of real publication prose. Generic titles and metadata also simplify the task. Real-paper evaluation remains necessary; no improved research agreement is claimed before training.

No benchmark title, full summary, or judge rationale enters the new training examples. The builder checks exact state/title exclusion and exact 16-word overlap between new scenario text and benchmark states. Rubric and option descriptions are intentionally shared. This check is not a proof of zero semantic overlap: benchmark failure themes deliberately inform the augmentation. The benchmark must remain labeled development evidence.

`paper-error-analysis.json` contains per-topic counts, the complete confusion table, and IDs/titles of disagreements. `data/v3.1/manifest.json` records the benchmark configuration and analysis hashes. The existing 500-example classification holdout is preserved byte-for-byte and has not been queried. There is no new untouched real-paper holdout in this revision.

## Before-training development baseline

The existing 4B v2.1 scored **168/192 (87.5%)** on the new authored research development exercises, using regex plus logprobs, with no HTTP or format errors. Seven of 96 canonical/shuffled pairs changed their semantic prediction. These 192 rows derive from just 24 core scenarios; they are correlated, and the score is not comparable to an independently sampled real-paper accuracy. No changes to the dataset were made in response to this baseline. Results are saved in `evaluation/v3-analysis/research-baseline/`. The reserved holdout was not queried.
