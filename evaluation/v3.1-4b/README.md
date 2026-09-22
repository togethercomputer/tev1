# 4B v3.1 decision benchmark

Model: `hassan/Qwen3.5-4B-v3.1-a9db1708-e46d1350`.

| Task | N | 4B v2.1 regex | 4B v3.1 regex |
|---|---:|---:|---:|
| News | 100 | 86.0% | 86.0% |
| Banking intent | 200 | 83.5% | 83.5% |
| Yes/no reading | 200 | 90.0% | 89.5% |
| Textual inference | 250 | 88.0% | 89.6% |
| Familiar policies | 150 | 100.0% | 100.0% |
| Sentiment | 100 | 49.0% | 47.0% |
| **Main** | **1,000** | **85.2% (852)** | **85.3% (853)** |
| **Policy transfer** | **300** | **88.0% (264)** | **88.7% (266)** |

Effectively flat on this development suite. Main-test paired changes: 15 previously wrong cases fixed, 14 previously correct cases regressed. Policy transfer: 5 fixed, 3 regressed. Missing-information cases: 77/100 versus 78/100. Entire three-case policy groups correct: 76/100 versus 75/100. No meaningful improvement established by these tiny net changes.

All 1,300 requests succeeded and returned a valid option. Prompt, option order, input hashes and regex/logprobs settings match the previous decoding comparison. Temperature 0, max_tokens 8, logprobs 5, thinking disabled, pooled HTTP, total concurrency 8, shuffled seed 42, no quality retries. Readiness probes used a dev example and were excluded. Research-paper tests and the reserved holdout were not queried.

Main-test median latency: 467 ms, p95 645 ms. Policy median: 462 ms, p95 653 ms. Timings are from this run, shortly after deployment. Previous baseline timings came from a different run with interleaved decoding modes; hardware, external traffic, and startup conditions were not controlled. Do not infer a model-speed regression from this comparison.

Historical Jev scores on the same records: main 853/1000 (85.3%), policy 298/300 (99.3%). Matching the main aggregate does not mean equivalent per-task behavior or broad Jev parity.

These are feedback-informed development benchmarks, not fresh holdouts. `report.json` contains aggregates, hashes, and paired counts; `results.jsonl` contains predictions, logprobs, timing and model identity. `run.py` is the runner used here. No training or deployment changes were performed.
