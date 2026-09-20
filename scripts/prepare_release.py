"""Stage an extracted adapter or merged checkpoint locally. Never uploads anything."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = {"adapter_config.json", "config.json", "generation_config.json",
                "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja",
                "preprocessor_config.json", "processor_config.json"}


def inspect_checkpoint(source):
    if not source.is_dir() or source.is_symlink():
        raise ValueError("Checkpoint must be a real directory")
    adapter = (source / "adapter_config.json").is_file()
    required = {"adapter_config.json", "adapter_model.safetensors"} if adapter else {
        "config.json", "tokenizer.json", "tokenizer_config.json"}
    missing = required - {p.name for p in source.iterdir() if p.is_file()}
    if missing:
        raise ValueError(f"Missing required files: {sorted(missing)}")
    weights = sorted(source.glob("*.safetensors"))
    if not weights:
        raise ValueError("No safetensors weights; convert/inspect the export first")
    selected = sorted(p for p in source.iterdir() if p.name in CONFIG_FILES
                      or p.name.endswith(".safetensors")
                      or p.name.endswith(".safetensors.index.json")
                      or p.name.startswith(("LICENSE", "NOTICE")))
    for p in selected:
        if p.is_symlink() or not p.is_file() or p.stat().st_size == 0:
            raise ValueError(f"Unsafe or empty artifact: {p.name}")
        if p.suffix == ".json":
            data = json.loads(p.read_text())
            if p.name.endswith(".index.json"):
                shards = set(data["weight_map"].values())
                if not shards <= {w.name for w in weights}:
                    raise ValueError("Index references missing weight shards")
    return selected, "adapter" if adapter else "merged"


def stage(source, destination, version="v2.1"):
    if version not in {"v1", "v2.1"}:
        raise ValueError("Unknown release version")
    selected, kind = inspect_checkpoint(source)
    if destination.resolve() == source.resolve() or source.resolve() in destination.resolve().parents:
        raise ValueError("Stage outside the source checkpoint")
    destination.mkdir(parents=True, exist_ok=False)
    for path in selected:
        shutil.copyfile(path, destination / path.name)
    card = ROOT / ("release/v1/MODEL_CARD.md" if version == "v1" else "MODEL_CARD.md")
    metadata = ROOT / ("release/training.json" if version == "v1" else "release/v2.1/training.json")
    shutil.copyfile(card, destination / "README.md")
    shutil.copyfile(metadata, destination / "training.json")
    shutil.copyfile(ROOT / "release/costs.json", destination / "costs.json")
    if version == "v1":
        shutil.copyfile(ROOT / "release/training-data-verification.json",
                        destination / "training-data-verification.json")
    # The source model card describes the adapter; correct merged package metadata.
    if kind == "merged":
        card = destination / "README.md"
        card.write_text(card.read_text().replace("base_model_relation: adapter", "base_model_relation: finetune")
                        .replace("library_name: peft", "library_name: transformers"))
    hashes = {}
    for path in sorted(destination.iterdir()):
        with path.open("rb") as handle:
            hashes[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    (destination / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in hashes.items()))
    return kind, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Extracted, inspected checkpoint directory")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/huggingface")
    parser.add_argument("--version", choices=["v1", "v2.1"], required=True,
                        help="Version of the exported weights; selects matching card and metadata")
    args = parser.parse_args()
    try:
        kind, hashes = stage(args.checkpoint, args.output, args.version)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Cannot stage release: {exc}\n")
    print(f"Staged {kind}: {len(hashes)} files in {args.output}")
    print("Local draft only. Verify license, base revision, loading, and predictions before upload.")


if __name__ == "__main__":
    main()
