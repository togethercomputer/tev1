# Source provenance

The exact source revisions are in `sources.lock.json`, copied into every manifest.
Each public record also includes its repository, revision, original split and row
index. Synthetic records include the executable rule tree, facts and generator version.

| Source | Upstream / dataset card | License metadata at pinned revision |
|---|---|---|
| MultiNLI | https://huggingface.co/datasets/nyu-mll/multi_nli | cc-by-3.0, cc-by-sa-3.0, MIT, other (mixed-source corpus) |
| BoolQ | https://huggingface.co/datasets/google/boolq | cc-by-sa-3.0 |
| Banking77 | https://huggingface.co/datasets/legacy-datasets/banking77 | cc-by-4.0 |
| AG News | https://huggingface.co/datasets/fancyzhx/ag_news | unknown |
| SST-5 | https://huggingface.co/datasets/SetFit/sst5 | unspecified in repository card metadata |
| Synthetic policies | `build_dataset.py`, generator policy-v1 | Created for this project; no external text or teacher model |

These are recorded metadata, not a conclusion that every source is cleared for
commercial redistribution. AG News and SST-5 in particular need provenance/license
review before publishing a combined dataset. Do not assume a model repository's
code license applies to its training corpora. This build remains local.

Public source records were downloaded unchanged, then sampled and converted.
No data was obtained from Jev, and no JevBench items were used. No teacher-generated
answers or probabilities are included. Original labels can be ambiguous or noisy;
small manual spot checks are not a full annotation audit.
