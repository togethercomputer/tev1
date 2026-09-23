# Building the tev1-4B-experimental dataset

The final dataset is `data/new-v1`: 37,840 training examples and 4,568 development
examples. Training contains 16,929,529 tokens; development contains 2,218,673.
The longest exported sequence is 1,526 tokens.

For a fresh checkout, run `uv run python fetch_sources.py` followed by
`uv run python build_all.py`. The wrapper runs the stages below in order, stops
on the first failure, and refuses existing output directories before starting.
`--dry-run` prints the build order without writing files.

## Build order

From the repository root, after `uv sync --locked`:

```bash
uv run python fetch_sources.py
uv run python build_dataset.py
uv run python validate_dataset.py data/v1
uv run python build_v2.py
uv run python validate_v2.py data/v2
uv run python build_v21.py
uv run python validate_v21.py data/v2.1
uv run python build_new_v1.py
```

`fetch_sources.py` downloads public datasets and a tokenizer at the revisions in
`sources.lock.json`; it does not download model weights. The following builders
use that cache. Keep the source lock unchanged when reproducing the saved data.
The historical exports use the pinned **Qwen3.5-2B tokenizer**, including its
non-thinking chat template; retaining this pin preserves the 4B recipe's original
bytes. This is a data-format pin, not the training base model.

| Builder | Role |
|---|---|
| `build_dataset.py` | Original public-data and synthetic-policy mixture |
| `build_v2.py` | Additional public data, missing-information policies, priority routing, and replay |
| `build_v21.py` | Adds fictional research-classification exercises using `configs/research-taxonomy.json` |
| `build_new_v1.py` | Merges v1 and v2.1 train/dev, removing 6,000 exact replay duplicates |

No v3/v3.1 augmentation enters the final data. The final builder checks every
available earlier non-training split for normalized-state overlap, including
v3/v3.1 if those local datasets exist. Those optional checks do not require the
removed experimental builders.

All builders refuse to overwrite an output directory. On an existing checkout,
validate the intermediates and skip their build commands. To rebuild only the
final merge, choose a fresh path with `build_new_v1.py --output data/new-v1-check`.
Intermediate inputs remain at their documented `data/v*` paths.

## Contents

| Training source | Examples |
|---|---:|
| MultiNLI | 5,000 |
| BoolQ | 3,000 |
| Banking77 | 3,000 |
| AG News | 1,500 |
| SST-5 | 2,000 |
| Original synthetic policies | 1,500 |
| Additional synthetic policies | 12,000 |
| Priority routing | 6,000 |
| Synthetic research classification | 3,840 |
| **Total** | **37,840** |

Each build exports `records/` (labels and provenance), `sft/` (chat messages), and
`instruction/` (rendered prompt/completion pairs). Upload the instruction files.
The final merge independently checks chat-template alignment, answer-plus-EOS
completions, token counts, conflicting duplicates, and split isolation.

`manifest.json` records counts and hashes; `membership.json` maps final examples
to their inputs; `validation.json` records the checks. The original final
[manifest](../runs/new-v1/dataset-manifest.json) is retained for comparison.
Builder hashes change when code changes; reproduced instruction-file hashes
should match the original manifest.

## Adapt the recipe

For a new task, use `state`, `question`, and `options` with consecutive labels A–X,
unique semantic keys, and descriptions. Supply `answer` and `answer_key` for
supervision. Use the same system instruction and JSON rendering as
`build_dataset.messages`; use the pinned tokenizer with `enable_thinking=False`
to render the prompt and append the correct letter plus its EOS token as the
completion. Keep every variant of a source document or synthetic case in one split.

Public labels can be noisy. Synthetic research examples share vocabulary and
templates; their count does not represent independent real papers. Historical
benchmark feedback informed the recipe, so those benchmarks are development
evidence. Reserve a new final test for your own model.

Review [source provenance](../DATA_SOURCES.md) before redistributing data.
