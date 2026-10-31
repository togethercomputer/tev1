"""Evaluate an existing Together endpoint. Makes billed API calls when run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.decide import payload, select


REQUIRED_FIELDS = ('id', 'source', 'group_id', 'answer', 'answer_key')


def validate_records(rows):
    """Reject malformed records before any billed request is sent."""
    if not rows:
        raise ValueError('Records must be nonempty')
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f'Record {index} must be a JSON object')
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f'Record {row.get("id", index)!r} is missing {", ".join(missing)}')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Record IDs must be unique')


def summarize(rows):
    return {"requests": len(rows), "correct": sum(r['correct'] for r in rows),
            "accuracy": sum(r['correct'] for r in rows) / len(rows) if rows else None,
            "valid": sum(r['valid'] for r in rows),
            "http_errors": sum(r['status'] != 200 for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=['together'], default='together')
    parser.add_argument('--model', required=True, help='Your Together deployment')
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--limit', type=int, help='Optional positive smoke-test limit')
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 16:
        parser.error('Use concurrency 1–16; this runner is not a load test')
    if args.limit is not None and args.limit < 1:
        parser.error('Limit must be positive')
    key_name = 'TOGETHER_API_KEY'
    key = os.environ.get(key_name)
    if not key:
        parser.error(f'Set {key_name}')
    raw = args.records.read_bytes()
    try:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        parser.error(f'Invalid JSON record: {exc}')
    if args.limit:
        rows = rows[:args.limit]
    try:
        validate_records(rows)
        for row in rows:
            payload(row, args.model)
            selected = next((o for o in row['options'] if o['label'] == row['answer']), None)
            if selected is None or selected['key'] != row['answer_key']:
                raise ValueError('Gold label and semantic key disagree')
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f'Output directory already exists: {args.output}. Choose a new --output.')
    url = 'https://api.together.ai/v1/chat/completions'

    def one(row):
        out = {k: row[k] for k in ['id', 'source', 'group_id']}
        out.update(expected=row['answer'], expected_key=row['answer_key'], status=0,
                   valid=False, correct=False, predicted_key=None)
        body = payload(row, args.model)
        start = time.perf_counter()
        try:
            request = urllib.request.Request(url, data=json.dumps(body).encode(),
                headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json',
                         'User-Agent': 'tev1-4B-experimental/0.1'})
            with urllib.request.urlopen(request, timeout=30) as response:
                out['status'] = response.status
                data = json.load(response)
            out['logprobs'] = data['choices'][0].get('logprobs')
            chosen = select(data, row['options'])
            out.update(valid=True, correct=chosen['label'] == row['answer'], predicted_key=chosen['key'],
                       resolved_model=data.get('model'))
        except urllib.error.HTTPError as exc:
            out.update(status=exc.code, error='HTTPError')
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            out['error'] = type(exc).__name__
        out['latency_ms'] = round((time.perf_counter() - start) * 1000, 2)
        return out

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        with (args.output / 'results.jsonl').open('x') as handle:
            for result in pool.map(one, rows):
                results.append(result)
                handle.write(json.dumps(result) + '\n')
                handle.flush()
    report = {'model': args.model, 'provider': args.provider,
              'finished_utc': datetime.now(timezone.utc).isoformat(),
              'records_sha256': hashlib.sha256(raw).hexdigest(), 'limit': args.limit,
              'concurrency': args.concurrency, 'retries': 0,
              'decoding': {'type': 'regex', 'options': 'per-record labels',
                            'logprobs': True, 'top_logprobs': 5, 'max_tokens': 8, 'temperature': 0,
                            'enable_thinking': False},
              'latency_note': 'urllib client; not connection-pooled like the historical run. Do not compare throughput.',
              **summarize(results),
              'by_source': {s: summarize([r for r in results if r['source'] == s])
                            for s in sorted({r['source'] for r in results})}}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if report['http_errors'] or report['valid'] != report['requests']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
