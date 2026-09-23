# tev1-4B-experimental: new v1, Qwen3.5-4B

This is the retained fresh-start 4B experiment. The local dataset contains
37,840 training examples and 4,568 validation examples after deduplicating the
v1 + v2.1 union. It contains no v3/v3.1 augmentation.

- Historical checkpoint: `hassan/Qwen3.5-4B-v1-new-69617472-bdc3c2fc`
- Evaluated endpoint: `together/Tev1-4B-experimental`
- [Run identity and verification limits](run.json)
- [Fresh-build reproduction check](reproduction.json): all final file hashes match
- [Original dataset manifest](dataset-manifest.json) and [validation](dataset-validation.json)
- [Decision evaluation report](evaluation.json) and [per-example results](results.jsonl)

| Development benchmark | Correct | Valid answers |
|---|---:|---:|
| Main decisions | 880/1,000 (88.0%) | 1,000/1,000 |
| Policy transfer | 300/300 (100%) | 300/300 |

The [saved Tev1 comparison](comparison.json) reported identical predictions to the previous v1-new
endpoint across 2,191 decision and paper examples. This is behavioral evidence,
not proof of identical weights. The retained decision results use temperature 0,
max_tokens 8, thinking disabled, per-request regex, `logprobs=true`, and
`top_logprobs=5`. Dataset input hashes are in the evaluation report.

These benchmarks were reused during development. They are not untouched test
results and do not establish calibrated confidence, general accuracy, or an
untuned-base comparison. Historical paired-baseline fields in the report are
preserved as recorded; the older model's release is no longer maintained here.

The local dataset files and counts are verified. The uploaded-file hashes,
training job ID, and actual hyperparameters could not be retrieved because the
Together API returned HTTP 403 during cleanup. The [training example](../../examples/train_together.py)
therefore exposes the saved **proposed** recipe, not claimed exact historical
settings. Endpoint names also do not establish public access or a public weights
release. The separate blog post will expand on the training run.
