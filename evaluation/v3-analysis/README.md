# Classification error analysis for v3

Target: `hassan/Qwen3.5-4B-jev-2-1-4286f48b-c0ad7564` (4B v2.1; interpreted as the user's “Tev1”). Existing benchmark inputs informed the analysis but are not training examples. Jev predictions are comparison evidence, never training labels.

| Source | 4B correct | Jev correct | Jev-only correct | 4B-only correct |
|---|---:|---:|---:|---:|
| News | 86/100 | 91/100 | 8 | 3 |
| Banking candidate sets | 167/200 | 173/200 | 12 | 6 |
| Five-level sentiment | 49/100 | 51/100 | 17 | 15 |
| Textual inference | 220/250 | 213/250 | 9 | 16 |
| Yes/no reading | 180/200 | 179/200 | 7 | 8 |

The model does not trail Jev on every task. The largest absolute error count is sentiment, but Jev is only two examples better there. Neither model's score is proof that a disputed upstream label is correct.

## Findings and response

- News: 7 of 14 errors are business labeled technology by the model. Acquisition, financing, and earnings stories can be distracted by company/product vocabulary. Add distinct authored business-versus-technical-event pairs and fresh source-labeled news, while excluding spotted source labels that contradict that distinction.
- Banking: neighboring intents and none-of-the-listed choices account for many mistakes. Ordering a card versus obtaining/delivering it, top-up status versus failure/reversal, and conversion action versus fees/rates need finer distinctions. Include observed confusion neighbors in 24-option training sets and paired gold-absent examples. These use source intent labels; removing the gold intent does not eliminate all possible semantic ambiguity among broad candidate names.
- Sentiment: 41 of 51 mistakes are adjacent-scale disagreements. Both models struggle. Use bounded source-labeled examples; do not manufacture strong positive/negative labels from superficial keywords, and do not train on benchmark sentences.
- Inference: 20 of 30 mistakes cross entailment versus neutral. Some involve unsupported time/details; others involve paraphrase or ambiguous annotations. Add explicitly labeled count, color, and missing-date/detail contrasts, preserving broader MNLI replay.
- Yes/no: errors include temporal status, similar-but-not-identical entities, and mixed/questionable annotations. Use a smaller fresh component rather than making this the main intervention.

`error-analysis.json` contains all original errors with gold/predicted keys and Jev correctness. Its excerpts retain original source licensing; they are evidence, not new training supervision. It was generated from the 1,000 v1 records and saved regex results, joined by record ID.

## Fresh candidate mining

Before mining, reserve 100 unused upstream-training records per source for development and 100 per source for holdout. Exclude normalized states from every previous v1/v2/v2.1 split and all official upstream held-out partitions. Freeze hashes in `data/v3/holdout-lock.json`. This is an in-domain holdout, not an unseen-domain test.

Score 4,000 other fresh upstream-training records with the deployed 4B, regex, logprobs 5, temperature zero, thinking disabled, total concurrency 8. All calls succeeded:

| Source | Candidate count | Errors |
|---|---:|---:|
| SST5 | 800 | 367 |
| Banking77 | 1,000 | 137 |
| AG News | 600 | 54 |
| MNLI | 1,200 | 186 |
| BoolQ | 400 | 47 |

Spot-audit exclusions are in `data/v3/quarantine.json`: ten ambiguous/fragmentary/questionable examples are excluded, not relabeled. This is a limited audit, not complete human adjudication. Remaining original labels can still be noisy. Sampling caps errors at half of each source's selected fresh examples, so the next run is not exclusively an error-repetition exercise. Mining predictions are metadata only and never replace labels.

The holdout has not been queried. Do not use it to adjust v3 training choices. The historical benchmark is already a development benchmark. Evaluate next checkpoints on development/regression sets first, then use the holdout once for the selected checkpoint; report all slices, not only a pooled score.
