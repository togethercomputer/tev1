# Qwen3.5-4B v2.1 decision evaluation

Measured deployment: `hassan/Qwen3.5-4B-jev-2-1-4286f48b-c0ad7564`.
Historical run started September 20, 2026 at 21:35 UTC.

| Task | N | 2B v2.1 | 4B v2.1 | Jev |
|---|---:|---:|---:|---:|
| Familiar policies | 150 | 82.0% | 100.0% | 97.3% |
| News classification | 100 | 87.0% | 86.0% | 91.0% |
| Yes/no questions | 200 | 83.0% | 90.0% | 89.5% |
| Banking intent, candidate subsets | 200 | 79.0% | 83.5% | 86.5% |
| Textual inference | 250 | 85.2% | 88.0% | 85.2% |
| Five-level sentiment | 100 | 50.0% | 49.0% | 51.0% |
| Main test | 1,000 | 79.7% | 85.2% | 85.3% |
| Original policy-transfer split | 300 | 52.0% | 88.0% | 99.3% |

All 1,300 4B requests succeeded with exact single-letter responses. The 4B
answered unknown correctly on 78/100 missing-information policy-transfer cases,
versus 6/100 for 2B v2.1 and 99/100 for Jev.

## Protocol and interpretation

These are the same v1 records and option orders as earlier runs. Qwen used
temperature 0, thinking disabled, max_tokens 8, logprobs enabled, concurrency 8,
and no retries. A development availability probe was excluded. Jev used its
native Choice API. The preceding, incorrect 4B endpoint identifier failed its
availability probe and never entered this benchmark.

Earlier failures informed the v2.1 dataset. These are development benchmarks,
not untouched final evaluations. Exact held-out records were excluded from the
constructed training dataset, but policy-transfer structures were unseen only
in the original v1 training recipe. A fresh holdout is needed to establish
generalization. No untuned-base-model comparison was performed.

The main-test median was 450.7 ms and p95 was 582.17 ms, including network and
provider overhead. Concurrent benchmarking may have affected timing. No 4B load
test was performed; original 2B QPS and serving-cost projections do not apply.

## Artifacts

- [report.json](report.json): original aggregate decision and policy results.
- [latency.json](latency.json): separate 30-request sequential latency sample,
  collected at 22:05 UTC with one request in flight, max_tokens 8, thinking off,
  temperature 0, and no logprobs. All 30 succeeded. After the first request,
  median latency was about 296 ms and p95 about 359 ms. The first request was
  about 390 ms; it was not a proven cold model start. Other endpoint traffic
  was not controlled. This sample is not a capacity benchmark.
- [2B results](../v2.1/README.md), [Jev results](../v1/jev/report.json), and
  [paper results](../papers-v2.1/README.md) supply the companion measurements.
- [Main chart](../../articles/assets/jev-vs-together-jev-v21-main.png) excludes
  policy transfer for readability; the full results remain in the table above.

These files were imported from the completed evaluations in “Explain TypeSafe
AI JEV.” This documentation update did not run new inference. The portable
runner and v1 data reconstruction instructions are in [the v1 guide](../v1/README.md).
