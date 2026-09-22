# Dataset v3.1 — use these upload files for the next 4B run

This revision preserves **all v3 training and validation records**, adding research-paper classification exercises informed by the latest 4B failure analysis. v3 remains unchanged. Continue from the existing 4B v2.1 checkpoint; this dataset has not yet produced an evaluated new model.

## Upload

- **Training:** `data/v3.1/instruction/train.jsonl` — **17,251 examples**, **9,614,572 tokens** per epoch.
- **Validation:** `data/v3.1/instruction/dev.jsonl` — **4,260 examples**, **2,411,558 tokens**.
- **Reserved holdout:** `data/v3.1/records/holdout.jsonl` — the same 500-example classification holdout as v3. Do not upload it as validation.

Maximum sequence length across the exports is **1,618 tokens**, below 2,048. The non-thinking Qwen chat template is already applied. Train with completion-only loss; do not add another template. Regex and logprobs remain inference settings, not training fields.

## Research representation

- Add **1,408 training exercises** derived from **48 authored research cores** (two per topic), with background and presentation variations and canonical/shuffled 24-option choices.
- Add **192 development exercises** derived from **24 separately authored cores**, one per topic.
- Preserve the **2,000 research replay examples** in v3, bringing total research training to **3,408 examples**, approximately **19.8% of training rows**.
- Give each of eight focus topics 80 new rows; the other sixteen topics receive 48 each. Focus: RL for reasoning, skills/memory, search, interpretability, efficiency, reasoning methods, research automation, and VLM architecture.
- Present longer prose with title, metadata, summary, and limitations. Vary the position of contribution evidence; include neighboring-topic background. Apply the real rubric's domain-specific benchmark rule.

See [research error analysis](evaluation/v3-analysis/RESEARCH.md) for the measured failure patterns and [v3](DATASET_V3.md) for the preserved general-classification augmentation, quarantine, replay, and split procedure.

These are **synthetic controlled exercises**, not 1,408 independently annotated real papers. Training and dev share the same taxonomy and construction style, although their core descriptions differ. Extra context and option-order variants are correlated. The extension is representative of the targeted format and distinctions, not demonstrated representative sampling of the real-paper distribution. Some reported benchmark disagreements are genuinely ambiguous, so we target clear distinctions rather than reproducing judge answers.

All benchmark papers and judge rationales remain out of the new training data. Checks cover exact benchmark states/titles and 16-word scenario overlap. The old research benchmark informed this revision and remains a development comparison. A separate real-paper holdout is still needed for a strong generalization claim.

## Training and selection

The conservative v3 starting proposal still applies: one continuation epoch, peak LR 0.00001, cosine decay, warmup 0.03, packing, sequence length 2048, completion-only loss, and compatible existing LoRA configuration. Rank or LR sweeps should be separate controlled experiments from the same starting weights; do not change data and rank simultaneously and attribute the result to rank.

Use development accuracy broken out into fresh public classification, new research exercises, existing research, policy, and routing. Check order sensitivity in the paired research renderings. Select a checkpoint before opening the reserved holdout. No claim of outperforming Jev is supported by this build alone.

## Reproduce and validate

`research_v31.py` contains all 72 authored cores and rendering rules. `build_v31.py` preserves v3, loads the benchmark rubric/options and aggregate confusion graph, checks overlap, and exports records/SFT/instruction formats. It refuses an existing output directory. `validate_v31.py` verifies all original v3 records, unchanged holdout bytes, hashes, split isolation, topic coverage, label mapping, distinct core descriptions, and export alignment. Source licenses and attribution for retained public data remain in `DATA_SOURCES.md`.

## Before-training development baseline

The existing 4B v2.1 scored **168/192 (87.5%)** on the new authored research development exercises, using regex plus logprobs, with no HTTP or format errors. Seven of 96 canonical/shuffled pairs changed their semantic prediction. These 192 rows derive from just 24 core scenarios; they are correlated, and the score is not comparable to an independently sampled real-paper accuracy. No changes to the dataset were made in response to this baseline. Results are saved in `evaluation/v3-analysis/research-baseline/`. The reserved holdout was not queried.
