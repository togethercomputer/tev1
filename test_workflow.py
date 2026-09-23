"""Offline regression checks for the retained workflow; no paid API calls."""
import json
from pathlib import Path
import subprocess
import tempfile
import os
import build_all
import sys
import unittest
from unittest.mock import Mock, patch

from examples.train_together import SETTINGS, upload_ready

ROOT = Path(__file__).resolve().parent


class WorkflowTests(unittest.TestCase):
    def test_preview_does_not_require_sdk_or_credentials(self):
        # -S excludes site packages, so an accidental SDK import fails here.
        result = subprocess.run(
            [sys.executable, '-S', 'examples/train_together.py'], cwd=ROOT,
            env={}, capture_output=True, text=True, check=True)
        self.assertIn('Preview only', result.stdout)
        self.assertIn('Qwen/Qwen3.5-4B', result.stdout)

    def test_upload_waits_for_ingestion_and_rejects_failed_files(self):
        client = Mock()
        client.files.upload.return_value.id = 'file-fixture'
        client.files.retrieve.side_effect = [Mock(processing_status='PROCESSING'),
                                            Mock(processing_status='COMPLETED')]
        with patch('examples.train_together.time.sleep'):
            self.assertEqual(upload_ready(client, Path('train.jsonl')), 'file-fixture')
        client.files.retrieve.side_effect = [Mock(processing_status='INVALID_FORMAT')]
        with self.assertRaises(RuntimeError):
            upload_ready(client, Path('train.jsonl'))

    def test_sdk_serializes_completion_only_lora_recipe(self):
        try:
            from together import Together
        except ImportError:
            self.fail('Run uv sync --locked to install the required Together SDK')
        client = Together(api_key='fixture')
        # Intercept submission and model-limits lookup so this test is offline.
        with patch.object(client.fine_tuning, 'estimate_price',
                          return_value=Mock(estimation_available=False)), \
             patch.object(client, 'post', return_value={}) as post:
            from together.types.finetune import FinetuneTrainingLimits, FinetuneLoraTrainingLimits
            limits = FinetuneTrainingLimits(
                max_num_epochs=10, max_learning_rate=0.001, min_learning_rate=1e-8,
                min_max_seq_length=256, max_seq_length_sft=8192, max_seq_length_dpo=8192,
                lora_training=FinetuneLoraTrainingLimits(
                    max_batch_size=8, min_batch_size=8, max_rank=64,
                    target_modules=['all-linear']))
            client.fine_tuning.create(training_file='file-train', validation_file='file-dev',
                                      model_limits=limits, **SETTINGS)
            self.assertTrue(post.called)
            body = post.call_args.kwargs['body']
            self.assertEqual(body['model'], 'Qwen/Qwen3.5-4B')
            self.assertEqual(body['training_type']['lora_r'], 8)
            self.assertFalse(body['training_method']['train_on_inputs'])

    def test_saved_results_match_report(self):
        base = ROOT / 'runs/new-v1'
        report = json.loads((base / 'evaluation.json').read_text())
        rows = [json.loads(line) for line in (base / 'results.jsonl').read_text().splitlines()]
        self.assertEqual(len({(r['split'], r['id']) for r in rows}), len(rows))
        for split in ['test', 'policy_transfer']:
            selected = [r for r in rows if r['split'] == split]
            self.assertEqual(len(selected), report[split]['n'])
            self.assertEqual(sum(r['correct'] for r in selected), report[split]['correct'])
            for row in selected:
                self.assertEqual(row['correct'], row['output'].strip() == row['gold'])
                self.assertEqual(row['status'], 200)


class BlogWorkflowTests(unittest.TestCase):
    def test_wrapper_orders_builders_and_stops_on_failure(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(build_all, 'ROOT', Path(directory)), \
             patch.object(sys, 'argv', ['build_all.py']), \
             patch.object(build_all.subprocess, 'run') as run:
            build_all.main()
            self.assertEqual([Path(c.args[0][1]).name for c in run.call_args_list],
                             [name for name, _ in build_all.STAGES])
            run.reset_mock()
            run.side_effect = [None, subprocess.CalledProcessError(1, 'build_v2.py')]
            with self.assertRaises(subprocess.CalledProcessError):
                build_all.main()
            self.assertEqual(run.call_count, 2)

    def test_wrapper_preserves_existing_data_before_any_build(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(build_all, 'ROOT', Path(directory)), \
             patch.object(sys, 'argv', ['build_all.py']), \
             patch.object(build_all.subprocess, 'run') as run:
            (Path(directory) / 'data' / 'v2').mkdir(parents=True)
            with self.assertRaises(SystemExit):
                build_all.main()
            run.assert_not_called()

    def test_blog_model_alias_and_example_contracts(self):
        for preferred, legacy, expected in [('', 'legacy-endpoint', 'legacy-endpoint'),
                                             ('preferred-endpoint', 'legacy-endpoint', 'preferred-endpoint')]:
            env = dict(os.environ, TOGETHER_MODEL=preferred, JEV_MODEL=legacy)
            for example in (ROOT / 'examples').glob('*.json'):
                result = subprocess.run([sys.executable, 'examples/decide.py', str(example),
                                         '--dry-run'], cwd=ROOT, env=env,
                                        capture_output=True, text=True, check=True)
                body = json.loads(result.stdout)
                self.assertEqual(body['model'], expected)
                self.assertEqual(json.loads(body['messages'][1]['content']),
                                 json.loads(example.read_text()))


if __name__ == '__main__':
    unittest.main()
