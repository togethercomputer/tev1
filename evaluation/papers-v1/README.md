# Original checkpoint: Research-paper classification

Historical results imported from the completed paper benchmark on September 20,
2026. These are separate from the dataset-label evaluation in `../v1/`.

Checkpoint: `hassan/Qwen3.5-2B-jev-f321c13e`.
Measured deployment: `hassan/Qwen3.5-2B-jev-f321c13e-20b140d0`, Together, 1× H100.

## Results

| Model | Matches / 891 | Judge agreement |
|---|---:|---:|
| DeepSeek V4.1 Flash, low reasoning | 828 | 92.9% |
| Jev | 813 | 91.2% |
| GLM 5.3 Flash, low reasoning | 809 | 90.8% |
| Original fine-tuned Qwen3.5-2B | 579 | 65.0% |

The benchmark selected the 891 of 1,018 papers where Kimi K3 and GPT 6 Astra
agreed on one of 24 category labels; the 127 disagreements were excluded.
These labels are model-judge consensus, not human-verified ground truth.
The project's author confirmed that these papers and labels were absent from
the fine-tuning data. Exposure during base-model pretraining is unknown.

## Method and limits

The paper state, rubric, and topic definitions matched the comparator experiments.
Qwen used the training system instruction and native `state`/`question`/`options`
JSON wrapper. A–X preserved the original taxonomy order. Generation used
temperature 0, thinking disabled, max_tokens 8, and no constrained decoding.
The generic JSON-output preflight was excluded from scoring. No prompt tuning or
option-order selection was performed using these test scores.

All 891 Qwen requests completed, with zero failed attempts or rate limits.
At concurrency 15, median response latency was 472.5 ms and p95 was 587.7 ms,
including local network and provider overhead. Comparator concurrency differed,
so these figures do not establish a speed ranking. Dedicated H100 hosting was
billed separately; the API returned no billable per-request cost.

Training examples had at most eight options; this evaluation used 24. Option
count, subject matter, and category boundaries changed together, so the results
do not isolate the cause of the performance gap. No untuned-Qwen baseline was run.

## Artifacts

- [metrics.json](metrics.json) is the original aggregate export from the paper
  benchmark, retaining the measured endpoint identity and comparison counts.
- [PNG chart](../../articles/assets/research-paper-classification.png) and
  [editable SVG](../../articles/assets/research-paper-classification.svg) show
  these same original-checkpoint results.

This directory contains aggregate evidence, not a standalone reproduction of the
paper benchmark: paper texts, raw responses, and the original runner are not
bundled. The repo's `scripts/evaluate.py` commands in `../v1/` reproduce the
decision-dataset protocol, not this separate paper experiment.
