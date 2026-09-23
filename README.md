# open-jev

**Small models for decisions with explicit options.**

Give the model context, a question, and two to 24 choices. Get back one letter.
open-jev fine-tunes **Qwen3.5-2B and 4B** with ordinary LoRA supervised fine-tuning,
keeping its standard language-model output head.

```text
State       A return is allowed within 30 days. This purchase was 12 days ago.
Question    Is this return within the allowed window?
Options     A: Yes   B: No   C: Not enough information
Answer      A
```

This is an independent, Jev-inspired research project. It is not affiliated with
TypeSafe AI, Jev, or Kev, and is not trained on Jev's answers.

**Status:** latest evaluated checkpoints are **2B v2.1 and 4B v2.1**.
Public weights are not yet published. The measured Together endpoints require
account access; they are not public shared endpoints or Hugging Face model IDs:

- 2B: `hassan/Qwen3.5-2B-jev-v2-1-9fea2a3a-203bb784`
- 4B: `hassan/Qwen3.5-4B-jev-2-1-4286f48b-c0ad7564`

| Training run | Model | Cost |
|---|---|---:|
| Original, 10,000 examples | `hassan/Qwen3.5-2B-jev-f321c13e` | $4.000 |
| Continuation, 33,840 examples | `hassan/Qwen3.5-2B-jev-v2-1-9fea2a3a` | $9.671 |
| **Total 2B training** | | **$13.671** |

These 2B training charges are user-confirmed and exclude dedicated hosting,
evaluation, and the separate 4B training. The article's $4 headline refers to
the first 2B run.

[2B model card](MODEL_CARD.md) · [2B evaluation](evaluation/v2.1/README.md) ·
[4B evaluation](evaluation/v2.1-4b/README.md) · [Paper comparison](evaluation/papers-v2.1/README.md) ·
[Original evaluation](evaluation/v1/README.md) · [v2.1 dataset](DATASET_V21.md) ·
[Release guide](docs/RELEASE.md) ·
[Training walkthrough](articles/how-to-train-your-own-jev-model-for-4-dollars.md)

**Next training dataset:** [v3.1 classification and research continuation](DATASET_V31.md) supersedes the v3 upload recommendation: 17,251 training examples, 4,260 validation examples, and the same reserved 500-example holdout. It has not been trained as a new model.

**Fresh-start experiment:** [new v1 (v1 + v2.1 union)](DATASET_NEW_V1.md) provides 37,840 deduplicated training examples and 4,568 validation examples for the original Qwen3.5-4B. This is separate from the continuation datasets.

## Try a decision

Requires Python 3.12+ and access to a deployed Together endpoint. Inference is
billed by your provider; the script does not create or start an endpoint.

```bash
export TOGETHER_API_KEY='your-key'
export JEV_MODEL='your-deployed-endpoint'
python examples/decide.py examples/return-window.json
```

The example sends the training system instruction and `state/question/options`
JSON, disables thinking, and constrains decoding to the supplied letters using
`response_format: {"type": "regex", "pattern": "(A|B|C)"}` (built from each
request's options). All Together inference examples and the evaluation runner
request `logprobs: 5`. The example returns the selected letter, semantic key,
and raw token logprobs; the runner saves logprobs in each result and records its
decoding settings in the report. Missing provider logprobs are represented as null.

Convert a returned logprob to probability with `math.exp(logprob)`. Read the
answer-letter token, not the end token. Top-five alternatives may omit valid
options, especially for 24-way decisions; do not treat missing options as zero or
renormalize a partial list as a full distribution. Our endpoint test showed
regex constraints affect the reported probabilities. These are model preferences,
not calibrated correctness confidence. Regex restricts completed answers to the
listed letters; the client still validates responses and handles API failures.

The archived scores below describe their original, unconstrained runs. New
Together evaluations use regex plus logprobs. See the
[decoding comparison](evaluation/decoding/README.md) for the matched 4B rerun.

Public Hugging Face loading instructions will be added after the downloaded
checkpoint has passed a local smoke test. We do not yet claim verified local,
quantized, or Mac inference support for this fine-tune.

## What works—and what needs work

Both latest checkpoints produced **one valid answer letter on all 1,300 decision-evaluation
requests**, without constrained decoding. The 2B continuation improved several language
tasks, especially textual inference, but familiar-policy accuracy regressed.

Missing information remains a weakness: the latest 2B correctly
marked unknown on **6/100** original policy-transfer missing-information cases,
versus **3/100** for v1 and **99/100** for Jev. Output format is reliable on this
sample; decision accuracy and calibrated confidence are separate concerns. The
latest 4B recognized unknown in 78/100 of those cases.

| Task | Examples | Original 2B | 2B v2.1 | 4B v2.1 | Jev 1.13 |
|---|---:|---:|---:|---:|---:|
| Familiar policy rules | 150 | 92.7% | 82.0% | 100.0% | 97.3% |
| News classification | 100 | 88.0% | 87.0% | 86.0% | 91.0% |
| Yes/no questions | 200 | 81.0% | 83.0% | 90.0% | 89.5% |
| Banking intent, candidate subsets | 200 | 77.5% | 79.0% | 83.5% | 86.5% |
| Textual inference | 250 | 77.2% | 85.2% | 88.0% | 85.2% |
| Five-level sentiment | 100 | 49.0% | 50.0% | 49.0% | 51.0% |
| **Main test** | **1,000** | **78.6%** | **79.7%** | **85.2%** | **85.3%** |
| **Original policy-transfer split** | **300** | **52.3%** | **52.0%** | **88.0%** | **99.3%** |

Measured September 20, 2026 on the same v1 records and options, using each model's
native API format. These are accuracies against dataset labels. The v1 failures
informed continuation training, so these are now **development benchmarks**, not
untouched final tests. Policy-transfer structures were unseen in v1 training;
that claim does not apply to v2.1. Use the reserved v2.1 evaluations for fresh
comparisons. No untuned Qwen baseline has been run. See [the latest protocol and
evidence](evaluation/v2.1/README.md) and the [4B report](evaluation/v2.1-4b/README.md).

On the separate 891-paper, 24-category benchmark drawn from
[1kpapers.com](https://1kpapers.com), judge agreement was **78.7%
for 2B v2.1**, **86.1% for 4B v2.1**, and **91.2% for Jev**. These are agreements
with Kimi K3/GPT 6 Astra consensus, not human-verified accuracy. Earlier paper
failures informed the synthetic research exercises, and actual v2.1 training
overlap has not been independently verified. See the
[paper protocol and metrics](evaluation/papers-v2.1/README.md).

The **original checkpoint's** brief H100 load test reached 61.7 successful
requests/sec at concurrency 16 without errors; higher concurrency produced HTTP
503s. No load test was run for v2.1. Do not transfer those throughput figures to
the latest endpoint or treat them as sustained-capacity guarantees.

## Design choices

We train Qwen's existing next-token predictor to select labels A–X (A–H in v1). The visible
answer is one token; normal generation can also emit a termination token.

Compared with [Jared Palmer's Kev](https://github.com/jaredpalmer/kev), this uses
ordinary token-target SFT rather than a custom pointer head. Kev shares document
computation across isolated question branches and scores their options directly.
Our current format handles one question per example; separate requests may repeat
context processing. The benefit here is a conventional model interface, while
Kev's specialized design offers capabilities we have not implemented or benchmarked.

We acknowledge Kev's public dataset recipe as inspiration. This project has its
own implementation and documentation; neither architecture nor performance
parity is claimed.

## Reproduce the data and training history

```bash
uv sync --locked
uv run python fetch_sources.py
uv run python build_dataset.py --output data/v1
uv run python validate_dataset.py data/v1
uv run python -m unittest -v
```

Builds refuse to overwrite existing output. If `data/v1` already exists, validate
it or choose a fresh output directory. The first mixture contains 10,000 training
examples, 1,000 each for dev/calibration/test, and 300 policy-transfer examples.
Sources and tokenizer revisions are pinned. See the [full recipe](docs/DATASET.md)
and [source provenance](DATA_SOURCES.md).

The continuation uses [dataset v2.1](DATASET_V21.md): 33,840 training examples,
including the 30,000-example v2 mixture and 3,840 research-classification exercises.
It adds 12- and 24-choice tasks; the 1,300-record table above still uses the original
v1 decision tests. See [actual continuation settings](release/v2.1/training.json):
one epoch, learning rate `2e-5`, sequence length 4,096, and `train_on_inputs="auto"`.
These recorded settings differ from the dataset guide's recommended 2,048-token
limit and explicit completion-only loss. The [training example](examples/train_together.py)
recreates **v1 from base Qwen**; it does not reproduce the continuation run.

## Release layout and licensing

**GitHub:** code, docs, source pins, training metadata, and evaluation evidence.
**Hugging Face:** model weights, tokenizer/configuration, model card, and checksums.
Weight downloads and staging folders stay out of Git. See the
[release guide](docs/RELEASE.md) for the download and local packaging steps.

Original code and documentation are MIT-licensed under [LICENSE](LICENSE).
That license does not cover third-party datasets or Qwen weights. Qwen3.5-2B's
upstream license is Apache-2.0; the fine-tuned weight release license is pending
review. The mixed dataset has no blanket license. In particular, AG News and
SST-5 need further provenance review before a combined dataset release.
