# Continued Qwen3.5-2B v2.1 benchmark

Model: hassan/Qwen3.5-2B-jev-v2-1-9fea2a3a-203bb784
Started: 2026-09-20T21:12:50.369593+00:00

| Task | N | Initial 2B | Initial 4B | 2B v2.1 | Jev |
|---|---:|---:|---:|---:|---:|
| ag_news | 100 | 88.0% | 84.0% | 87.0% | 91.0% |
| banking77 | 200 | 77.5% | 83.0% | 79.0% | 86.5% |
| boolq | 200 | 81.0% | 85.0% | 83.0% | 89.5% |
| mnli | 250 | 77.2% | 86.8% | 85.2% | 85.2% |
| policy | 150 | 92.7% | 91.3% | 82.0% | 97.3% |
| sst5 | 100 | 49.0% | 45.0% | 50.0% | 51.0% |
| test | 1000 | 78.6% | 81.9% | 79.7% | 85.3% |
| policy_transfer | 300 | 52.3% | 75.3% | 52.0% | 99.3% |

## Policy details and verification

- test: 797/1000 correct; 0 HTTP errors; 1000 exact single-letter responses.
- Missing-information policies: 24/50 correct.
- Definite policies incorrectly answered unknown: 0/100.
- All three related policy cases correct: 24/50 groups.
- Compared with initial 2B: 51 newly correct, 40 newly wrong.
- Median latency 453.2 ms; p95 737.66 ms.
- policy_transfer: 156/300 correct; 0 HTTP errors; 300 exact single-letter responses.
- Missing-information policies: 6/100 correct.
- Definite policies incorrectly answered unknown: 0/200.
- All three related policy cases correct: 1/100 groups.
- Compared with initial 2B: 18 newly correct, 19 newly wrong.
- Median latency 458.83 ms; p95 595.95 ms.

## Interpretation

These are the same v1 decision and policy tests used in prior comparisons. Temperature 0, thinking disabled, max_tokens 8, logprobs enabled, concurrency 8, no retries. One dev probe is excluded. IDs, coverage, and scores are independently checked against the source files. Previous model scores are reused from earlier runs.

The v1 test failures informed the v2.1 training recipe. Exact v1 held-out records were excluded from the constructed training dataset, but these are now development benchmarks, not untouched final evaluations. In particular, “policy_transfer” means structures unseen in v1 training; the broader v2.1 training generator may cover those structures. Improvement here does not establish generalization to rule structures unseen by v2.1. Use reserved v2.1 test and transfer data for a fresh comparison.

No paper benchmark or load test was run. Latency includes network and provider overhead and may include competing endpoint traffic. It is not a controlled hardware-speed comparison. Banking77 uses candidate subsets. Jev scores use its native Choice API. These are small task subsets with correlated policy groups, not proof of general superiority.

The saved report.json contains original aggregate metrics. test.jsonl and policy_transfer.jsonl contain text-free per-record outcomes. Source passages, response bodies, and credentials are omitted. For the portable rerun command, see [the v1 evaluation guide](../v1/README.md#rerun-quality-evaluation). The unchanged v1 dataset manifest and comparator results remain in ../v1. These files record a historical run, not a new evaluation during documentation updates.

The separate [4B v2.1 evaluation](../v2.1-4b/README.md) and
[research-paper comparison](../papers-v2.1/README.md) provide the companion
results used in the article's latest two charts.
