"""Build the complete new-v1 dataset from cached sources, in dependency order."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = (
    ('build_dataset.py', 'v1'),
    ('build_v2.py', 'v2'),
    ('build_v21.py', 'v2.1'),
    ('build_new_v1.py', 'new-v1'),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Show build order without writing files')
    args = parser.parse_args()
    if not args.dry_run:
        existing = [str(ROOT / 'data' / version) for _, version in STAGES
                    if (ROOT / 'data' / version).exists()]
        if existing:
            parser.error('Refusing to overwrite existing datasets: ' + ', '.join(existing)
                         + '. See docs/DATASET.md to validate and resume individual stages.')
    for script, version in STAGES:
        print(f'{script} -> data/{version}', flush=True)
        if not args.dry_run:
            subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)
    if not args.dry_run:
        print('Ready: data/new-v1/instruction/train.jsonl and dev.jsonl')


if __name__ == '__main__':
    main()
