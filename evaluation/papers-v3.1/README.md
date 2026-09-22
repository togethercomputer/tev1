# 4B v3.1: combined evaluation

Model: `hassan/Qwen3.5-4B-v3.1-a9db1708-e46d1350`.

| Evaluation | N | 4B v2.1 | 4B v3.1 | Jev |
|---|---:|---:|---:|---:|
| News | 100 | 86.0% | 86.0% | 91.0% |
| Banking intent | 200 | 83.5% | 83.5% | 86.5% |
| Yes/no reading | 200 | 90.0% | 89.5% | 89.5% |
| Textual inference | 250 | 88.0% | 89.6% | 85.2% |
| Familiar policies | 150 | 100.0% | 100.0% | 97.3% |
| Sentiment | 100 | 49.0% | 47.0% | 51.0% |
| **Main test** | **1,000** | **85.2%** | **85.3%** | **85.3%** |
| **Policy transfer** | **300** | **88.0%** | **88.7%** | **99.3%** |
| **Research papers** | **891** | **86.1% (767)** | **84.3% (751)** | **91.2% (813)** |

Main and policy results are dataset-label accuracy. Paper scores are agreement with the previous two-model consensus, not human-verified accuracy; do not pool the three scores. Comparator values are historical. All 2,191 v3.1 regex requests succeeded and returned valid options. The 1,300-task run had already completed; the paper experiment adds 891 different tasks.

Paper changes: 25 previous disagreements fixed, 41 previous agreements regressed. Interpretation/analysis improved from 17/25 to 24/25. Agent benchmarks fell 33/41 to 28/41, coding 30/33 to 25/33, and image generation 54/57 to 48/57. The targeted recipe did not improve the overall paper result. Keep v2.1 as the stronger overall baseline; these results do not establish which training choice caused the regression.

## Decoding control

Prior paper comparators used unconstrained output. An additional 891-paper v3.1 run with unconstrained output scored 750/891 (84.2%), versus regex 751/891 (84.3%). Only one semantic answer changed. Regex does not explain the observed regression. Both new runs requested logprobs 5; the historical paper run did not request logprobs.

## Method

Same states, titles, summaries, rubric, system instruction, canonical A-X options, and consensus references as the source task **Test Jev model cost and success**. No judge rerun or relabeling. New paper requests: temperature 0, thinking off, max_tokens 8, logprobs 5, regex over all 24 letters, pooled HTTP, concurrency 15, no retries. The control omits only response_format. Configuration and reference hashes are recorded in report.json.

Paper regex latency: median 858 ms, p95 1,148 ms, concurrency 15. General test latency: median 467 ms, p95 645 ms, concurrency 8. These are wall-clock client latencies, not concurrency-one timings or controlled hardware comparisons. Dedicated endpoint cost was not measured.

The prior failures informed training, so these remain development comparisons. The reserved holdout was not queried. No new training or deployment changes were made.

Artifacts: report.json, results.jsonl, run.py; plain-control/ holds the matching decoding control. The 1,300-task details are in ../qwen4b-v31-benchmark/ for this task's outputs, and evaluation/v3.1-4b/ in the repository.
