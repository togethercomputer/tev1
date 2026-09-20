# First checkpoint evaluation — September 20, 2026

Checkpoint: `hassan/Qwen3.5-2B-jev-f321c13e`.
Measured deployment: `hassan/Qwen3.5-2B-jev-f321c13e-20b140d0`, Together H100.
Comparator: `typesafe/jev-1.13`, resolved to `typesafe/jev-1.13-20260917`, OpenRouter.

## Results

| Split | N | Qwen correct | Jev correct |
|---|---:|---:|---:|
| Main test | 1,000 | 786 (78.6%) | 853 (85.3%) |
| Unseen policy structures | 300 | 157 (52.3%) | 298 (99.3%) |

On the main test, both were correct on 730 records, Jev alone on 123, and Qwen
alone on 56. On unseen policies, both were correct on 157 and Jev alone on 141;
Qwen alone was never correct. The remaining records were wrong for both.

The most important failure is missing information: Qwen selected unknown correctly
on 3/100 unseen-policy cases, Jev on 99/100. For entire three-case policy groups,
Qwen got 3/100 correct and Jev 98/100. All 1,300 Qwen outputs were valid single
letters without constrained decoding; all 1,300 requests succeeded for both models.

## Method

States, questions, options, semantic descriptions, and option order were identical.
Qwen received the training system instruction and state/question/options JSON,
with temperature 0, thinking disabled, max_tokens 8, and logprobs enabled. Jev
received one native Choice question per record, including yes/no and sentiment
cases, with criteria keyed by the same letters. Its instructions retained the
state-as-data rule. Neither received ground-truth labels or examples. This is
matched task evidence, not identical serialization across two different APIs.

Both quality runs used concurrency 8 and no retries. Main-test median latency
was 302.79 ms for Qwen and 272.28 ms for Jev, including network and provider
routing. Different serving stacks and run times prevent a hardware or architecture
speed conclusion. Jev reported $0.026197 for 1,300 calls, excluding a dev smoke
call. Qwen's dedicated hosting cost is not comparable to a per-call usage charge;
we do not claim a Qwen cost advantage.

The separate Qwen load test issued requests for up to six seconds or 512 requests
per concurrency stage, then drained them. It reused short test prompts, potentially
benefiting from caches, with max_tokens 1 and no logprobs. Concurrency 16 achieved
61.7 successful requests/sec with 0/388 errors. Concurrency 32 had 1/512 errors;
64 had 69/512. All were HTTP 503s; the failing serving component was not identified.
These are brief observations, not sustainable capacity estimates. Jev was not
load-tested.

## Serving-cost estimate

[serving-cost-estimate.json](serving-cost-estimate.json) derives a hosting-cost
equivalent from the fastest error-free load stage: 61.7 successful requests/sec,
388 requests, concurrency 16, and 6.288 seconds including the drain. That stage
processed 82,520 input tokens and 388 output tokens, averaging 212.6804 input
tokens plus one output token per successful request.

At the author's assumed **$5/hour for one H100**, sustaining that rate throughout
the billed hour would yield 222,120 decisions/hour, **$22.51 per million decisions**,
or **$0.105 per million combined input/output tokens**. Formula:

```text
decisions/hour = successful QPS × 3,600 × utilization
$/million decisions = hourly cost × 1,000,000 / decisions/hour
$/million total tokens = $/million decisions / mean total tokens per request
```

Utilization here means average traffic relative to the observed request rate,
not a measured GPU-utilization percentage. At half the traffic, both unit costs
double. $5/hour is a scenario assumption, not a current provider quote; the
historical deployment screen quoted $5.40/hour, giving costs 8% higher.

This extrapolation is not a measured sustained maximum or a per-token API price.
It does not establish independent input/output rates, apply to longer paper
inputs, or transfer to the continued checkpoint. It counts successful responses,
not correct decisions: the selected load stage got 299/388 correct. Training,
other infrastructure, and possible retries are excluded. See the JSON for the
source-report hash, full-precision calculations, and utilization scenarios.

## Evidence and reproducibility

- `qwen/report.json` and `jev/report.json`: original aggregate reports.
- Each model's `test.jsonl` and `policy_transfer.jsonl`: sorted, text-free outcomes
  retaining record IDs, gold/predicted semantic keys, groups, correctness, status,
  and latency. Source passages, provider response bodies, and credentials are omitted.
- `dataset-manifest.json` and `sources.lock.json`: frozen v1 build metadata and
  upstream pins. The manifest includes export hashes, split counts, and integrity
  checks. Uploaded train/dev bytes were verified against the v1 instruction export hashes;
  see `release/training-data-verification.json`.
- `python -m unittest -v` recomputes correctness, paired counts, per-source totals,
  missing-information performance, and dataset-manifest consistency from the saved
  outcomes. These checks require no network or model weights.

The records and reports were imported from the original endpoint and Jev comparison
runs in the project’s training task. These are historical results, not a fresh run
during release preparation. Rebuild the v1 dataset using `docs/DATASET.md`, then
match its file hashes before using it to reproduce these scores. Original training
and dataset generation use different seeds (42 and 20260920 respectively).

## Limits and next comparisons

Banking77 is candidate-set classification, not full 77-way prediction. The overall
score weights tasks according to this artificial mixture. Public-dataset exposure
in pretraining is unknown; deduplication does not exclude semantic overlap.
Policy-transfer examples share only two held-out rule families and correlated
three-case groups. They do not measure broad real-world policy competence.

No untuned Qwen baseline, verified local-weight inference, quantized evaluation,
or calibration fitting was performed. Winning-token probabilities are not
normalized, calibrated class probabilities. Unknown is a ground-truth class,
not a confidence threshold. New training guided by these failures needs a fresh
final holdout. Preserve this directory when developing the next dataset.

The separate 891-paper benchmark also evaluated this Qwen checkpoint against
model-judge consensus. Its 65.0% agreement score and comparator results are
archived in [evaluation/papers-v1](../papers-v1/README.md). They remain separate
from the dataset-label accuracy reported here.

## Rerun quality evaluation

The portable runner below makes billed requests to existing endpoints. It does
not deploy anything or run a load test. Set `TOGETHER_API_KEY` or
`OPENROUTER_API_KEY` in your shell. Start with `--limit 10` for a smoke test;
omit it for the full split. Output directories must not already exist.

```bash
python scripts/evaluate.py --provider together --model "$JEV_MODEL" \
  --records data/v1/records/test.jsonl --output outputs/qwen-test --limit 10
python scripts/evaluate.py --provider jev --model typesafe/jev-1.13 \
  --records data/v1/records/test.jsonl --output outputs/jev-test --limit 10
```

Repeat with `policy_transfer.jsonl` and fresh output paths for the transfer split.
The runner preserves the quality input formats and generation settings, logs the
input hash and model identity, and counts HTTP/format failures as incorrect. Its
standard-library HTTP client differs from the original pooled client: rerun
latencies must not be presented as a reproduction of the original load test.
