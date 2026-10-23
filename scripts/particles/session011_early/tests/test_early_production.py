"""No native/GPU execution: test the early-launch policy and mocked scheduler only."""
import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import early_gate
import early_run
import early_submit
import workflow
import launch


class EarlyProductionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='s11-early-unit-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.current = {'unit_fixture_only': True, 'early_production': {'mode': early_gate.MODE}}

    def git(self, *args):
        return {('branch', '--show-current'): workflow.BRANCH,
                ('status', '--porcelain'): '', ('rev-parse', 'HEAD'): 'unit-commit'}[args]

    def checks(self):
        return {'native_inputs_and_tools': {'passed': True, 'details': self.current},
                'checkout': {'passed': True, 'details': {'commit': 'unit-commit'}}}

    def test_queued_validation_manifest_is_unchanged(self):
        path = early_gate.SESSION / 'evidence/submissions/validation_20261023T014715Z_97d0cd0bc5.json'
        submitted = json.loads(path.read_text())['preflight']['native_inputs_and_tools']['details']
        self.assertEqual(workflow.manifest(), submitted)

    def test_actual_policy_binds_native_model_and_declares_unvalidated(self):
        current = early_gate.manifest()
        mode = current['early_production']
        self.assertEqual(mode['mode'], 'user_authorized_unvalidated')
        self.assertFalse(mode['gpu_validation_verified']); self.assertFalse(mode['push_verified'])
        self.assertIn('early_run.py', mode['files'])
        self.assertIn('production_policy.json', mode['files'])

    def test_different_native_or_executable_not_authorized(self):
        current = workflow.manifest()
        for kind in ('native_source_sha256', 'executable_sha256'):
            changed = copy.deepcopy(current)
            if kind == 'native_source_sha256':
                changed[kind] = 'different-native-source'
            else:
                changed[kind]['gi'] = 'different-gi-binary'
            with patch.object(early_gate, 'validation_manifest', return_value=changed), self.assertRaisesRegex(RuntimeError, 'authorization'):
                early_gate.manifest()

    def test_no_direct_launch_without_submitted_hash_binding(self):
        with patch.object(early_gate, 'manifest', return_value=self.current), \
             patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, 'hashes'):
            early_gate.readiness()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_readiness_records_waiver_without_creating_validation_or_push_receipts(self):
        root = self.root / 'runtime/production_1234'; root.mkdir(parents=True)
        env = dict(SLURM_JOB_ID='1234', S11_PRODUCTION_COMMIT='unit-commit',
                   S11_PRODUCTION_MANIFEST_SHA256=early_gate.digest(self.current))
        with patch.object(early_gate, 'SESSION', self.root), patch.object(early_gate, 'manifest', return_value=self.current), \
             patch.object(early_gate, 'git', side_effect=self.git), patch.dict(os.environ, env, clear=True):
            self.assertEqual(early_gate.readiness(), self.current)
        mode = json.loads((root / 'RUN_MODE.json').read_text())
        self.assertFalse(mode['scientifically_validated'])
        self.assertFalse(mode['gpu_validation_verified']); self.assertFalse(mode['push_verified'])
        self.assertTrue(mode['resource_guards_retained'])
        self.assertFalse((self.root / 'evidence/VALIDATION_PASS.json').exists())
        self.assertFalse((self.root / 'evidence/PUSH_VERIFIED.json').exists())

    def test_changed_queued_hash_or_commit_rejected(self):
        env = dict(SLURM_JOB_ID='1234', S11_PRODUCTION_COMMIT='unit-commit',
                   S11_PRODUCTION_MANIFEST_SHA256=early_gate.digest(self.current))
        for key in ('S11_PRODUCTION_COMMIT', 'S11_PRODUCTION_MANIFEST_SHA256'):
            with patch.object(early_gate, 'manifest', return_value=self.current), \
                 patch.object(early_gate, 'git', side_effect=self.git), \
                 patch.dict(os.environ, env | {key: 'changed'}, clear=True), self.assertRaises(RuntimeError):
                early_gate.readiness()

    def test_pending_validation_allows_production_and_existing_production_blocks_it(self):
        validation = '1091645|s11-ah-validation                   |PENDING'
        with patch.object(early_submit.legacy, 'command', return_value=validation):
            result = early_submit.existing_jobs('production')
            self.assertEqual(result['other_stage_jobs_allowed'], [validation])
            with self.assertRaisesRegex(RuntimeError, 'same-stage'):
                early_submit.existing_jobs('validation')
        with patch.object(early_submit.legacy, 'command', return_value='1234|s11-final-150M|RUNNING'):
            with self.assertRaisesRegex(RuntimeError, 'same-stage'):
                early_submit.existing_jobs('production')

    def test_vista_banner_response_and_simple_job_ids(self):
        attempt = early_gate.SESSION / 'evidence/submissions/validation_20261023T014715Z_97d0cd0bc5.json'
        self.assertEqual(early_submit.job_id(json.loads(attempt.read_text())['stdout']), '1091645')
        for raw in ('1234\n', '1234;vista\n', 'Banner\nSubmitted batch job 1234\n'):
            self.assertEqual(early_submit.job_id(raw), '1234')
        for raw in ('', 'banner only', '1234\nsubmission failed'):
            with self.assertRaises(RuntimeError):
                early_submit.job_id(raw)

    def test_early_preflight_does_not_require_or_call_validated_push_gate(self):
        with contextlib.ExitStack() as stack:
            for name in ('submission_location', 'checkout', 'allocation', 'partition', 'modules', 'storage'):
                stack.enter_context(patch.object(early_submit.legacy, name, return_value={}))
            stack.enter_context(patch.object(early_submit, 'manifest', return_value=self.current))
            stack.enter_context(patch.object(early_submit, 'existing_jobs', return_value={}))
            strict = stack.enter_context(patch.object(early_submit.legacy, 'ready', side_effect=AssertionError('strict gate called')))
            checks = early_submit.preflight('production')
        strict.assert_not_called()
        self.assertTrue(all(item['passed'] for item in checks.values()))
        self.assertFalse(checks['early_production_policy']['gpu_validation_verified'])

    @contextlib.contextmanager
    def submission(self):
        with patch.object(early_submit, 'SESSION', self.root), patch.object(early_submit.legacy, 'SESSION', self.root), \
             patch.object(early_submit.legacy, 'submission_location', return_value={}), \
             patch.object(early_submit, 'preflight', return_value=self.checks()), \
             patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
            yield

    def test_single_normal_production_submission_retains_banner_and_hashes(self):
        with self.submission(), patch.object(early_submit.legacy, 'command', return_value='test-only OK') as dry, \
             patch.object(early_submit.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'Vista banner\n1234\n', '')) as scheduler:
            early_submit.submit_job('production')
            with self.assertRaisesRegex(RuntimeError, 'already exists'):
                early_submit.submit_job('production')
        scheduler.assert_called_once()
        args = scheduler.call_args.args[0]
        self.assertEqual(args[-1], str(HERE / 'production.sbatch'))
        self.assertNotIn('--hold', args); self.assertFalse(any('--dependency' in arg for arg in args))
        self.assertIn('--test-only', dry.call_args.args[0])
        env = scheduler.call_args.kwargs['env']
        self.assertEqual(env['S11_PRODUCTION_MANIFEST_SHA256'], early_gate.digest(self.current))
        self.assertEqual(env['S11_PRODUCTION_COMMIT'], 'unit-commit')
        record = json.loads(next((self.root / 'evidence/submissions').glob('production_*.json')).read_text())
        self.assertEqual(record['job_id'], '1234'); self.assertEqual(record['state'], 'accepted')
        self.assertEqual(record['stdout'], 'Vista banner\n1234\n')
        self.assertEqual(record['mode'], early_gate.MODE)
        self.assertFalse(record['gpu_validation_verified']); self.assertFalse(record['push_verified'])

    def test_timeout_retains_partial_response_and_blocks_retry(self):
        timeout = subprocess.TimeoutExpired(['sbatch'], 30, output=b'Vista partial banner\n', stderr=b'partial stderr')
        with self.submission(), patch.object(early_submit.legacy, 'command', return_value='test-only OK'), \
             patch.object(early_submit.subprocess, 'run', side_effect=timeout) as scheduler:
            with self.assertRaises(subprocess.TimeoutExpired):
                early_submit.submit_job('production')
            with self.assertRaisesRegex(RuntimeError, 'already exists'):
                early_submit.submit_job('production')
        scheduler.assert_called_once()
        record = json.loads(next((self.root / 'evidence/submissions').glob('production_*.json')).read_text())
        self.assertEqual(record['state'], 'ambiguous')
        self.assertEqual(record['stdout'], 'Vista partial banner\n')
        self.assertEqual(record['stderr'], 'partial stderr')

    def test_check_default_has_no_scheduler_or_filesystem_side_effect(self):
        with patch.object(early_submit, 'preflight', return_value=self.checks()), \
             patch.object(early_submit.legacy, 'pending_attempts', return_value=None), \
             patch.object(early_submit.subprocess, 'run') as scheduler, \
             patch.object(sys, 'argv', ['early_submit.py']), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(early_submit.main(), 0)
        scheduler.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_production_runner_and_guards_are_reused(self):
        production = launch.production
        with patch.object(launch, 'workflow_ready'), patch.object(launch, 'manifest'), \
             patch.object(launch, 'main') as runner, patch.object(sys, 'argv', ['early_run.py', 'production']), \
             contextlib.redirect_stdout(io.StringIO()):
            early_run.main()
            self.assertIs(launch.production, production)
            self.assertIs(launch.workflow_ready, early_gate.readiness)
            self.assertIs(launch.manifest, early_gate.manifest)
        runner.assert_called_once()


if __name__ == '__main__':
    unittest.main()
