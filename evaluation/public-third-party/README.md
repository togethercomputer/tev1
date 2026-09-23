# Public third-party Jev benchmarks: paired rerun

Models: {'qwen': ['hassan/Qwen3.5-4B-v1-new-69617472-bdc3c2fc'], 'jev': ['typesafe/jev-1.13-20260917']}

| Benchmark | N | New Qwen 4B | Jev | Difference |
|---|---:|---:|---:|---:|
| phishing | 2000 | 1013/2000 (50.6%) | 1259/2000 (62.9%) | -12.3 pp |
| tool_risk | 60 | 56/60 (93.3%) | 56/60 (93.3%) | +0.0 pp |
| ticket_routing | 27 | 26/27 (96.3%) | 27/27 (100.0%) | -3.7 pp |

Attempt accounting: {'qwen': {'attempts': 2087, 'failed_attempts': 0}, 'jev': {'attempts': 2088, 'failed_attempts': 1}}

## Sources

- [anisselbd/jev-phishing-bench](https://github.com/anisselbd/jev-phishing-bench/tree/1d56e8c64d029a9554a0874e2ef2901ed196e230)
- [themsquared/jev-benchmark](https://github.com/themsquared/jev-benchmark/tree/d699d44558e27c9071caef8d5cea51615d5a3131)
- [WallerChen/jev-measured](https://github.com/WallerChen/jev-measured/tree/4a12dfb3e59fe368760af44607745c97b65e3fa9)

## Protocol and interpretation

Primary decision only; exact source states, instructions, criteria, labels and option order. Qwen native letter wrapper with regex; Jev one native choice. No tuning, no few-shot examples, no truncation. Phishing original asked nine questions; this matched rerun asks only verdict. One scored pass; one failed Jev HTTP 520 retried once. Report accuracy, not probability calibration.

Qwen: temperature 0, max_tokens 8, thinking disabled, logprobs 5, per-task letter regex, 15 concurrency. Jev: typesafe/jev-1.13 via OpenRouter Decisions API, one native choice per request, concurrency 10. Same input evidence and decision descriptions, different native API wrappers. All source labels are used only locally. Preflight successes are included once. Successful requests reused by ID. One Jev HTTP 520 attempt was retried once; original failure retained in raw logs. No Qwen retries.

Both models were run afresh; historical published results are not substituted. Phishing uses the source verdict question only, while the original report used nine questions and probability audits. This run does not reproduce its signal classifiers or calibration study. Source risk and routing scores also used different response wrappers and repetition protocols.

PhishNChips contains 2,000 synthetic emails around malicious/legitimate URLs and known construction shortcuts. The source reports a simple URL heuristic at 91.6%; classify this as a dataset result, not a deployed security guarantee. Risk has only 60 hand-labelled examples with subjective/ambiguous classes; routing has 27 deliberately clear examples and a ceiling effect. No pooled headline score because suite sizes and difficulties differ. Wilson intervals and paired counts are in report.json; small score gaps are not proof of broad superiority.

No exact normalized state matches were found against locally prepared new-v1 training and validation records. This does not audit actual uploads, near-duplicates, or base-model pretraining. No prompt or label-order tuning was performed based on the outcomes.

Latency uses different concurrency/provider paths and is not a controlled speed comparison. Jev cost is API-reported; Qwen dedicated-hosting cost is unknown here. See raw JSONL usage. Both raw responses and complete frozen inputs are saved for reproducibility.

Phishing recall: Qwen 1.3%, Jev 42.7%. False positive rate: Qwen 0.0%, Jev 16.8%.

Total Jev reported cost: $0.047469.
