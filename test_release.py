"""Offline checks for the decision contract and the frozen release evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from examples.decide import payload, select, SYSTEM
from scripts.prepare_release import stage
from scripts.evaluate import jev_payload, summarize
from build_dataset import SYSTEM as TRAINING_SYSTEM

ROOT = Path(__file__).resolve().parent


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.row = json.loads((ROOT / 'examples/return-window.json').read_text())

    def test_training_contract_and_no_gold_leakage(self):
        row = dict(self.row, answer='A', answer_key='yes', provenance={'secret': True})
        body = payload(row, 'fixture-endpoint')
        self.assertEqual(SYSTEM, TRAINING_SYSTEM)
        self.assertEqual(json.loads(body['messages'][1]['content']), self.row)
        self.assertFalse(body['chat_template_kwargs']['enable_thinking'])

    def test_duplicate_labels_and_keys_rejected(self):
        for field in ['label', 'key']:
            row = copy.deepcopy(self.row)
            row['options'][1][field] = row['options'][0][field]
            with self.assertRaises(ValueError):
                payload(row, 'fixture')

    def test_latest_24_option_contract(self):
        row = copy.deepcopy(self.row)
        row['options'] = [{'label': label, 'key': label, 'description': label}
                          for label in 'ABCDEFGHIJKLMNOPQRSTUVWX']
        payload(row, 'fixture')
        self.assertEqual(select({'choices': [{'message': {'content': 'X'}}]},
                                row['options']), {'label': 'X', 'key': 'X'})
        row['options'].append({'label': 'Y', 'key': 'Y', 'description': 'Y'})
        with self.assertRaises(ValueError):
            payload(row, 'fixture')

    def test_no_prose_or_unlisted_answers(self):
        for text in ['A because it is 12 days.', 'D', '', None]:
            with self.assertRaises(ValueError):
                select({'choices': [{'message': {'content': text}}]}, self.row['options'])
        self.assertEqual(select({'choices': [{'message': {'content': 'A'}}]},
                                self.row['options']), {'label': 'A', 'key': 'yes'})

    def test_jev_preserves_options_without_gold_and_errors_count(self):
        row = dict(self.row, answer='A', answer_key='yes')
        body = jev_payload(row, 'fixture')
        self.assertNotIn('answer', body)
        self.assertEqual(body['questions']['decision']['criteria'], {
            o['label']: {'key': o['key'], 'description': o['description']}
            for o in self.row['options']})
        result = summarize([{'correct': True, 'valid': True, 'status': 200},
                            {'correct': False, 'valid': False, 'status': 503}])
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['http_errors'], 1)


class EvidenceTests(unittest.TestCase):
    def test_continuation_results_and_costs(self):
        from decimal import Decimal
        costs = json.loads((ROOT / 'release/costs.json').read_text())
        self.assertEqual(sum(Decimal(r['training_cost_usd']) for r in costs['runs']),
                         Decimal(costs['total_training_cost_usd']))
        base = ROOT / 'evaluation/v2.1'
        report = json.loads((base / 'report.json').read_text())
        self.assertEqual(report['model'], 'hassan/Qwen3.5-2B-jev-v2-1-9fea2a3a-203bb784')
        for split in ['test', 'policy_transfer']:
            rows = [json.loads(line) for line in (base / f'{split}.jsonl').read_text().splitlines()]
            prior = {r['id']: r for r in map(json.loads,
                     (ROOT / f'evaluation/v1/qwen/{split}.jsonl').read_text().splitlines())}
            self.assertEqual({r['id'] for r in rows}, prior.keys())
            self.assertEqual(len(rows), len(prior))
            self.assertEqual(sum(r['correct'] for r in rows), report[split]['correct'])
            for row in rows:
                self.assertEqual(row['expected_key'], prior[row['id']]['expected_key'])
                self.assertEqual(row['correct'], row['predicted_key'] == row['expected_key'])
            for source, summary in report[split]['by_source'].items():
                self.assertEqual(sum(r['correct'] for r in rows if r['source'] == source), summary['correct'])
            if split == 'policy_transfer':
                self.assertEqual(sum(r['correct'] for r in rows if r['expected_key'] == 'unknown'), 6)

    def test_training_hashes_match_frozen_manifest(self):
        manifest = json.loads((ROOT / 'evaluation/v1/dataset-manifest.json').read_text())
        verified = json.loads((ROOT / 'release/training-data-verification.json').read_text())
        for split in ['train', 'dev']:
            self.assertTrue(verified[split]['matches_local_v1'])
            self.assertEqual(verified[split]['sha256'], manifest['files'][f'instruction/{split}.jsonl'])

    def test_saved_outcomes_match_reports_and_each_other(self):
        base = ROOT / 'evaluation/v1'
        models = {}
        for model in ['qwen', 'jev']:
            report = json.loads((base / model / 'report.json').read_text())
            models[model] = {}
            for split, count in [('test', 1000), ('policy_transfer', 300)]:
                rows = [json.loads(line) for line in (base / model / f'{split}.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), count)
                self.assertEqual(len({r['id'] for r in rows}), count)
                models[model][split] = {r['id']: r for r in rows}
                for row in rows:
                    self.assertEqual(row['status'], 200)
                    self.assertTrue(row['valid'])
                    self.assertEqual(row['correct'], row['expected_key'] == row['predicted_key'])
                self.assertEqual(sum(r['correct'] for r in rows), report[split]['correct'])
                for source, summary in report[split]['by_source'].items():
                    self.assertEqual(sum(r['correct'] for r in rows if r['source'] == source), summary['correct'])
        report = json.loads((base / 'jev/report.json').read_text())
        for split in ['test', 'policy_transfer']:
            q, j = models['qwen'][split], models['jev'][split]
            self.assertEqual(q.keys(), j.keys())
            for key in q:
                for field in ['source', 'group_id', 'expected', 'expected_key']:
                    self.assertEqual(q[key][field], j[key][field])
            paired = {'both_correct': sum(q[k]['correct'] and j[k]['correct'] for k in q),
                      'jev_only_correct': sum(not q[k]['correct'] and j[k]['correct'] for k in q),
                      'qwen_only_correct': sum(q[k]['correct'] and not j[k]['correct'] for k in q)}
            self.assertEqual(paired, report[split]['paired'])
        for model, expected in [('qwen', 3), ('jev', 99)]:
            rows = models[model]['policy_transfer'].values()
            unknowns = [r for r in rows if r['expected_key'] == 'unknown']
            self.assertEqual(len(unknowns), 100)
            self.assertEqual(sum(r['correct'] for r in unknowns), expected)


class PackagingTests(unittest.TestCase):
    def test_staging_filters_secrets_and_hashes_files(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'source'
            source.mkdir()
            (source / 'adapter_config.json').write_text('{}')
            (source / 'adapter_model.safetensors').write_bytes(b'fixture-only')
            (source / '.env').write_text('PRIVATE=not-a-real-key')
            (source / 'optimizer.pt').write_bytes(b'omit')
            dest = Path(temp) / 'staged'
            kind, hashes = stage(source, dest)
            self.assertEqual(kind, 'adapter')
            self.assertEqual(json.loads((dest / 'training.json').read_text())['id'], 'ft-0289519a-742b')
            self.assertFalse((dest / 'training-data-verification.json').exists())
            old_dest = Path(temp) / 'v1'
            stage(source, old_dest, 'v1')
            self.assertEqual(json.loads((old_dest / 'training.json').read_text())['id'], 'ft-e3f96627-90fa')
            self.assertTrue((old_dest / 'training-data-verification.json').exists())
            self.assertIn('first checkpoint', (old_dest / 'README.md').read_text())
            self.assertFalse((dest / '.env').exists())
            self.assertFalse((dest / 'optimizer.pt').exists())
            for name, digest in hashes.items():
                self.assertEqual(hashlib.sha256((dest / name).read_bytes()).hexdigest(), digest)
            with self.assertRaises(FileExistsError):
                stage(source, dest)

    def test_incomplete_adapter_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'source'
            source.mkdir()
            (source / 'adapter_config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Missing'):
                stage(source, Path(temp) / 'staged')


if __name__ == '__main__':
    unittest.main()
