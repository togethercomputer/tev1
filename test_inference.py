"""Offline checks for the decision contract and evaluation scoring."""
import copy
import json
from pathlib import Path
import unittest

from examples.decide import payload, select, SYSTEM
from scripts.evaluate import summarize, validate_records
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
        self.assertIs(body['logprobs'], True)
        self.assertEqual(body['top_logprobs'], 5)

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

    def test_errors_count_in_evaluation(self):
        result = summarize([{'correct': True, 'valid': True, 'status': 200},
                            {'correct': False, 'valid': False, 'status': 503}])
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['http_errors'], 1)


class EvaluationInputTests(unittest.TestCase):
    """Malformed records must fail before any billed request is sent."""

    def setUp(self):
        row = json.loads((ROOT / 'examples/return-window.json').read_text())
        row.update(id='r1', source='fixture', group_id='g1', answer='A', answer_key='yes')
        self.row = row

    def test_missing_context_field_rejected(self):
        for field in ('id', 'source', 'group_id', 'answer', 'answer_key'):
            records = [dict(self.row)]
            del records[0][field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                validate_records(records)

    def test_duplicate_and_empty_records_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unique'):
            validate_records([self.row, dict(self.row)])
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            validate_records([])


if __name__ == '__main__':
    unittest.main()
