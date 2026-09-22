"""Build data/new-v1 end to end from the pinned HuggingFace sources.

Runs the existing builders in order, each in its own subprocess, and stops at
the first failure. Stages whose output directory already exists are reused,
because every builder is deterministic and refuses to overwrite; delete a
stage's directory if you want that stage rebuilt. Run fetch_sources.py first
if .cache/huggingface is not already populated.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("v1 from HF datasets", ["build_dataset.py", "--output", "data/v1"]),
    ("v2 policy continuation", ["build_v2.py", "--output", "data/v2"]),
    ("v2.1 research augmentation", ["build_v21.py", "--output", "data/v2.1"]),
    ("new-v1 merge of v1 + v2.1", ["build_new_v1.py"]),
]
FINAL = ROOT / "data" / "new-v1"


def main():
    for i, (name, cmd) in enumerate(STAGES):
        output = FINAL if i == len(STAGES) - 1 else ROOT / "data" / ["v1", "v2", "v2.1"][i]
        if output.exists():
            print(f"[skip] {name}: {output} exists, reusing it", flush=True)
            continue
        print(f"[build] {name} -> {output}", flush=True)
        result = subprocess.run([sys.executable, *cmd], cwd=ROOT)
        if result.returncode != 0:
            print(f"FAILED at stage '{name}' (exit {result.returncode})", file=sys.stderr)
            return result.returncode
    print(f"DONE: final dataset at {FINAL}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())