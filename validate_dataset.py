"""Offline validation of a finished build and its aligned exports."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from build_dataset import ROOT, COUNTS, messages, policy_label, validate_partitions


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", type=Path, default=ROOT / "data/v1")
    args = parser.parse_args()
    root = args.directory
    assert not (root / "FAILED.json").exists(), "Build failed"
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert actual == expected, f"Checksum mismatch: {name}"
    partitions = {split: read(root / "records" / f"{split}.jsonl") for split in manifest["splits"]}
    integrity = validate_partitions(partitions)
    assert integrity == manifest["integrity"]
    for split, rows in partitions.items():
        expected_count = 10000 if split == "train" else 300 if split == "policy_transfer" else 1000
        assert len(rows) == expected_count
        if split != "policy_transfer":
            expected_sources = COUNTS if split == "train" else {k:v//10 for k,v in COUNTS.items()}
            assert dict(Counter(r["source"] for r in rows)) == expected_sources
        sft = read(root / "sft" / f"{split}.jsonl")
        instruction = read(root / "instruction" / f"{split}.jsonl")
        assert len(rows) == len(sft) == len(instruction)
        groups = defaultdict(list)
        for row, chat, raw in zip(rows, sft, instruction):
            assert chat == {"messages":messages(row)}
            assert raw["completion"] == row["answer"] + manifest["eos_token"]
            assert raw["prompt"].endswith(manifest["assistant_prefix_suffix"])
            assert row["token_count"] <= manifest["max_tokens"]
            if row["source"] == "policy":
                groups[row["group_id"]].append(row)
                p = row["provenance"]
                assert policy_label(p["tree"], p["atom_facts"]) == row["answer_key"]
            else:
                original = row["provenance"]["original_split"]
                assert (original == "train") == (split != "test")
        for group in groups.values():
            assert len(group) == 3
            assert {r["answer_key"] for r in group} == {"eligible", "ineligible", "unknown"}
    print(json.dumps({"status":"PASS", "integrity":integrity, "splits":{s:len(r) for s,r in partitions.items()}}, indent=2))


if __name__ == "__main__":
    main()
