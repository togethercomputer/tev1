# Release guide

Nothing in this preparation publishes a repository, model, dataset, or endpoint.
The current release candidate is **2B v2.1**, trained by job `ft-0289519a-742b`
from parent `ft-e3f96627-90fa`. The latest measured deployment is
`hassan/Qwen3.5-2B-jev-v2-1-9fea2a3a-203bb784`.

[Latest training metadata](../release/v2.1/training.json),
[latest model card](../MODEL_CARD.md), and [latest evaluation](../evaluation/v2.1/README.md)
belong together. The [original card](../release/v1/MODEL_CARD.md),
[original metadata](../release/training.json), and evaluation/v1 remain historical.
[Training costs](../release/costs.json): $4.000 + $9.671 = $13.671, excluding hosting
and evaluation. Charges are confirmed by the user.

## Where the weights should live

Use **GitHub for the project and Hugging Face for model artifacts**.

| Location | Advantages | Tradeoffs |
|---|---|---|
| GitHub | Code review, issues, releases, CI, familiar contribution workflow | Ordinary Git is a poor fit for weight binaries; LFS adds a separate workflow. Release assets must each be under 2 GiB. |
| Hugging Face | Model discovery, model cards, revisioned downloads, large-file storage, standard model-loading ecosystem | A second release to manage; license, base revision, and complete loading assets must be explicit. Public storage remains subject to Hub policies. |

Publish the adapter first if its base revision can be confirmed, and optionally a
merged safetensors model for easier loading. An adapter is smaller but requires
compatible base weights. A merged release is larger and self-contained with the
right tokenizer/configuration. Do not put either into this code repository.

Sources: [HF model uploads](https://huggingface.co/docs/hub/models-uploading),
[HF storage](https://huggingface.co/docs/hub/storage-limits),
[GitHub release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

## Weight export status

No weights have been downloaded or locally validated. The original v1 job's
CLI and direct download attempts returned a download failure / HTTP 403 on
September 20, 2026. **v2.1 export has not been attempted**; its access status is
unknown. Use the latest training job ID below, not the deployment name or parent
job, when preparing the latest release.

```bash
uv tool install 'together[cli]'
export TOGETHER_API_KEY='your-key'
tg fine-tuning download ft-0289519a-742b \
  --checkpoint-type adapter --output-dir artifacts/qwen3.5-2b-jev-v2.1/adapter
# Optional merged export:
tg fine-tuning download ft-0289519a-742b \
  --checkpoint-type merged --output-dir artifacts/qwen3.5-2b-jev-v2.1/merged
```

These commands download only. They do not deploy or publish. CLI output may be a
compressed archive; inspect its member paths and extract into a fresh local
folder. Consult [Together's export documentation](https://docs.together.ai/docs/fine-tuning/deployment)
for archive handling. Never publish optimizer states, cached datasets, account
metadata, or signed download URLs with the model.

## Stage a Hugging Face folder

After extraction, point the script at the actual directory containing
`adapter_config.json` or `config.json`:

```bash
python scripts/prepare_release.py artifacts/qwen3.5-2b-jev-v2.1/adapter/extracted \
  --version v2.1 --output dist/huggingface-v2.1
```

It copies only recognized model assets and upstream license/notice files, adds
the selected version's draft card as `README.md`, matching training metadata, and
the cost history, then computes
`SHA256SUMS`. It rejects missing adapter files, missing indexed shards, symlinks,
and an existing destination. It does not validate tensor contents or perform
inference or identify the checkpoint from its tensors. Verify the export's job ID
yourself; `--version` selects documentation, not weights. Use `--version v1` only
for the original checkpoint. Only v1 currently has verified uploaded-data hashes;
those are never presented as verification of v2.1. No upload function is included.

## Before publishing

- [ ] Download the latest v2.1 artifact and record its hashes and job provenance.
- [x] Match uploaded train/dev files to the v1 instruction export hashes
      (recorded in `release/training-data-verification.json`).
- [ ] Match uploaded v2.1 train/dev files to local v2.1 hashes.
- [ ] Verify whether the continued adapter export is self-contained relative to
      base Qwen or requires parent artifacts; document the tested loading path.
- [ ] The job reports `train_on_inputs=auto`; verify the actual loss-mask behavior.
- [ ] Verify Together's base weight revision, adapter targets, tokenizer, and
      non-thinking template. A tokenizer pin alone does not identify base weights.
- [ ] Choose the weight license after checking upstream model and dataset terms;
      include applicable license/notice files and add HF `license` metadata.
      MIT in this repo applies to original code/docs, not third-party data/weights.
- [ ] Run the downloaded checkpoint on representative examples and compare its
      predictions against the recorded endpoint. Document environment and hardware.
- [ ] Replace draft/download-blocker language in the card after resolving it.
      Set adapter vs merged metadata accurately. Add the real GitHub and HF links.
- [ ] Keep evaluation/v1 and evaluation/v2.1 frozen. If test failures influence new training data,
      use a fresh final holdout and a new results directory for the next checkpoint.
- [ ] Run `python -m unittest -v` and `git diff --check`; inspect the files selected
      for Git. Ensure no secrets, caches, model binaries, or third-party data text
      are staged. Do not run `git add -f` on ignored artifacts.
- [ ] When ready, publish the code repository and upload **only the inspected
      model folder** to the chosen HF model repository. Record the Hub commit SHA
      in the GitHub release so code, weights, and evaluation can be matched.

The public code repository can precede the weights, provided it retains the
explicit “weights not yet published” status. A finished model release still
requires the unchecked artifact and licensing steps above.
