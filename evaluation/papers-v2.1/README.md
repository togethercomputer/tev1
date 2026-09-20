# Research-paper classification: 2B and 4B v2.1

Historical results from the completed “Test Jev model cost and success” task,
imported September 20, 2026. No new inference was run for the article update.

| Model | Matches / 891 | Agreement |
|---|---:|---:|
| DeepSeek V4.1 Flash, low reasoning | 828 | 92.9% |
| Jev | 813 | 91.2% |
| Qwen3.5-4B v2.1 | 767 | 86.1% |
| Qwen3.5-2B v2.1 | 701 | 78.7% |

Measured deployments:

- 2B: `hassan/Qwen3.5-2B-jev-v2-1-9fea2a3a-203bb784`
- 4B: `hassan/Qwen3.5-4B-jev-2-1-4286f48b-c0ad7564`

## Method and limitations

The benchmark uses the 891 of 1,018 papers where Kimi K3 and GPT 6 Astra agreed
on one of 24 category labels, excluding 127 disagreements. Agreement with these
model judges is not human-verified accuracy. The paper state, rubric, and topic
definitions matched earlier experiments. Qwen received its native decision
format with A–X in the original taxonomy order, temperature 0, thinking disabled,
max_tokens 8, and no constrained decoding. Each run completed all 891 requests
with no failed attempts or rate limits, at concurrency 15.

The [v2.1 dataset recipe](../../DATASET_V21.md) uses the taxonomy and aggregate
failure patterns to construct synthetic research exercises; it excludes the
benchmark's real paper titles, summaries, and judge answers from those exercises.
The actual v2.1 training uploads have not been independently verified against
that recipe. These are feedback-informed development results, not an untouched
test set. Do not claim proven training independence or general model parity.

Original 2B agreement was 579/891 (65.0%). The continued 2B reached 701/891
(78.7%), with 155 previously wrong answers corrected and 33 previously correct
answers lost. Original 4B agreement was 730/891 (81.9%); the latest 4B reached
767/891 (86.1%). In `4b-metrics.json`, the `paired` block compares with original
2B, not original 4B.

## Artifacts

- [2b-metrics.json](2b-metrics.json) and [4b-metrics.json](4b-metrics.json):
  original aggregate exports, including endpoint identities, counts, timing,
  and the training-overlap caveat.
- [Final chart](../../articles/assets/research-paper-classification-v21-no-glm.png)
  and [editable SVG](../../articles/assets/research-paper-classification-v21-no-glm.svg).
  The chart omits GLM at the author's request; the original metrics retain its
  809/891 result (90.8%).
- [Original 2B paper evaluation](../papers-v1/README.md).

Paper texts, raw responses, and the original runner are not bundled here, so
these aggregate artifacts are not a standalone reproduction of the benchmark.
Dedicated-endpoint cost cannot be inferred from zero per-token prices in model
metadata. Longer paper prompts and different serving conditions also prevent
using these runs as a controlled speed comparison.
