# New Qwen3.5-4B v1 comparison

Model: `hassan/Qwen3.5-4B-v1-new-69617472-bdc3c2fc`

| Benchmark | 4B v2.1 | New 4B v1 |
|---|---:|---:|
| Main decisions | 852/1000 (85.2%) | 880/1000 (88.0%) |
| Policy transfer | 264/300 (88.0%) | 300/300 (100.0%) |
| All decisions | 1116/1300 (85.8%) | 1180/1300 (90.8%) |
| Research papers | 767/891 (86.1%) | 773/891 (86.8%) |

All 2,191 scored requests succeeded with valid output and the requested model identity; no quality retries. One separate dev readiness probe also succeeded. IDs, gold labels, option mappings, input hashes, and scores were independently verified.

Main test: 45 previous misses fixed, 17 regressions. Policy transfer: 36 fixed, no regressions; missing-information subset improved from 78/100 to 100/100. Papers: 34 fixed, 28 regressions, net +6. The paper gain is small and does not by itself establish a reliable generalization improvement.

## Main test breakdown

| Task | v2.1 | New model |
|---|---:|---:|
| ag_news | 86/100 | 87/100 |
| banking77 | 167/200 | 181/200 |
| boolq | 180/200 | 181/200 |
| mnli | 220/250 | 229/250 |
| policy | 150/150 | 150/150 |
| sst5 | 49/100 | 52/100 |

## Protocol

Decision tests: same v1 test and policy_transfer inputs as verified v2.1 regex baseline, temperature 0, max_tokens 8, thinking off, logprobs 5, regex per option list, concurrency 8. Papers: identical frozen state, system, option order and judge-consensus reference; temperature 0, max_tokens 8, thinking off, no regex or logprobs, concurrency 15. The modes differ to match each historical baseline. Ground-truth labels are never sent to the endpoint.

Median latency: 487 ms main decisions; 476 ms policy transfer; 864 ms papers. These are client-observed timings, not a controlled hardware speed comparison.

These are reused development benchmarks. Paper scores measure agreement with two model judges, not human ground truth. No independent audit of the new model’s actual training upload was performed. Dedicated endpoint request billing is unavailable; no cost inferred from zero token prices. Raw responses, usage and reports are in the adjacent evaluation folders.
