# Decoding comparison: Qwen3.5-4B Jev v2.1

Model: `hassan/Qwen3.5-4B-jev-2-1-4286f48b-c0ad7564`.

| Mode | Main test (1,000) | Policy transfer (300) | Missing information (100, subset) | Main median latency |
|---|---:|---:|---:|---:|
| Prompt only | 852 / 1,000 (85.2%) | 264 / 300 (88.0%) | 78% | 305 ms |
| Regex | 852 / 1,000 (85.2%) | 264 / 300 (88.0%) | 78% | 306 ms |
| JSON Schema | 855 / 1,000 (85.5%) | 258 / 300 (86.0%) | 73% | 336 ms |

All 3,900 requests succeeded and matched their requested output format. Regex produced exactly the same selected answer as plain output on every example. JSON Schema changed 39 main-test answers (18 fixes, 15 regressions, 6 wrong-to-wrong) and 8 policy answers (1 fix, 7 regressions). This is no evidence of a meaningful accuracy improvement from JSON Schema. Recommend regex for single-choice decisions and construct any JSON envelope in application code.

Requests were randomly interleaved with seed 42 and total concurrency 8, temperature 0, thinking disabled, logprobs 5, and no retries. Plain and regex shared the exact same prompt and max_tokens 8. JSON Schema changed the output instruction, included the schema in the system prompt, and used max_tokens 64. JSON Schema's changes cannot be attributed solely to the decoding constraint. It generated about 7 output tokens versus 2 for plain/regex.

Latency includes HTTP/provider overhead but excludes local semaphore wait. Mixed workloads share the endpoint; outside traffic was not controlled. These are not concurrency-one or maximum-throughput measurements. Earlier failures informed v2.1 training, so these are existing development benchmarks, not a fresh final holdout. Research-paper evaluation was excluded.

For new regex evaluations, use `scripts/evaluate.py` from the repository root with the v1 test and policy_transfer records. `report.json` records aggregate/per-source/paired results and input SHA-256 hashes. `results.jsonl` records each outcome without source passages or credentials.
