"""Non-science checks for explicit retries and controller-bound pending-job updates."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import early_gate
import early_submit
import queued_binding
import update_production
import launch
from prepare import sha


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='s11-repair-unit-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.folder = self.root/'evidence/submissions'; self.folder.mkdir(parents=True)
        self.current = dict(native_source_sha256='native', kokkos_commit='kokkos',
                            files={'scripts/inputs/production.athinput': 'input'},
                            executable_sha256={'gi': 'new-binary'})
        self.old = copy.deepcopy(self.current); self.old['executable_sha256']['gi'] = 'old-binary'
        self.original = dict(epoch=1, job_id='1234', state='accepted',
                             submitted_manifest_sha256=early_gate.digest(self.old), submitted_commit='old-commit',
                             preflight={'native_inputs_and_tools': {'details': self.old}})
        self.path = self.folder/'production_unit.json'; self.path.write_text(json.dumps(self.original))

    def binding(self):
        path = self.root/'evidence/production_updates/1234.json'; path.parent.mkdir(exist_ok=True)
        value = dict(schema_version=1, job_id='1234', mode=early_gate.MODE,
                     gpu_validation_verified=False, push_verified=False, native_or_physics_changed=False,
                     original_submission_file=self.path.name, original_submission_sha256=sha(self.path),
                     original_manifest_sha256=self.original['submitted_manifest_sha256'], original_commit='old-commit',
                     replacement_manifest_sha256=early_gate.digest(self.current), replacement_manifest=self.current,
                     replacement_commit='new-commit')
        path.write_text(json.dumps(value))
        env = dict(SLURM_JOB_ID='1234', S11_PRODUCTION_COMMIT='old-commit',
                   S11_PRODUCTION_MANIFEST_SHA256=self.original['submitted_manifest_sha256'])
        return path, value, env

    def test_local_replacement_receipt_requires_controller_binding(self):
        path, _, env = self.binding()
        with patch.object(queued_binding, 'SESSION', self.root), patch.dict(os.environ, env, clear=True):
            with patch.object(queued_binding, 'job_info', return_value=({'Comment': 'none'}, 'unit')), \
                 self.assertRaisesRegex(RuntimeError, 'bound'):
                queued_binding.verify(self.current, early_gate.digest(self.current))
            with patch.object(queued_binding, 'job_info', return_value=({'Comment': 'S11_UPDATE_'+sha(path)}, 'unit')):
                result = queued_binding.verify(self.current, early_gate.digest(self.current))
        self.assertEqual(result['replacement_commit'], 'new-commit')

    def test_altered_original_manifest_or_replacement_cannot_rebind(self):
        path, _, env = self.binding()
        with patch.object(queued_binding, 'SESSION', self.root), patch.dict(os.environ, env, clear=True), \
             patch.object(queued_binding, 'job_info', return_value=({'Comment': 'S11_UPDATE_'+sha(path)}, 'unit')):
            changed = copy.deepcopy(self.current); changed['executable_sha256']['gi'] = 'changed-again'
            with self.assertRaisesRegex(RuntimeError, 'hashes'):
                queued_binding.verify(changed, early_gate.digest(changed))
            self.path.write_text(self.path.read_text()+' ')
            with self.assertRaisesRegex(RuntimeError, 'hashes'):
                queued_binding.verify(self.current, early_gate.digest(self.current))

    def test_physics_or_input_changes_cannot_use_same_job_update(self):
        for target in ('native', 'input'):
            path, value, env = self.binding()
            changed = copy.deepcopy(self.current)
            if target == 'native': changed['native_source_sha256'] = 'different'
            else: changed['files']['scripts/inputs/production.athinput'] = 'different'
            value.update(replacement_manifest=changed, replacement_manifest_sha256=early_gate.digest(changed))
            path.write_text(json.dumps(value))
            with patch.object(queued_binding, 'SESSION', self.root), patch.dict(os.environ, env, clear=True), \
                 patch.object(queued_binding, 'job_info', return_value=({'Comment': 'S11_UPDATE_'+sha(path)}, 'unit')), \
                 self.assertRaisesRegex(RuntimeError, 'hashes'):
                queued_binding.verify(changed, early_gate.digest(changed))

    def test_amended_readiness_checks_new_commit_and_records_original_binding(self):
        path, _, env = self.binding()
        run = self.root/'runtime/production_1234'; run.mkdir(parents=True)
        def git(*args):
            return {('branch', '--show-current'): 'project/GI-in-cluster',
                    ('status', '--porcelain'): '', ('rev-parse', 'HEAD'): 'new-commit'}[args]
        with patch.object(queued_binding, 'SESSION', self.root), patch.object(early_gate, 'SESSION', self.root), \
             patch.object(early_gate, 'manifest', return_value=self.current), \
             patch.object(early_gate, 'git', side_effect=git), patch.dict(os.environ, env, clear=True), \
             patch.object(queued_binding, 'job_info', return_value=({'Comment': 'S11_UPDATE_'+sha(path)}, 'unit')):
            self.assertEqual(early_gate.readiness(), self.current)
        mode = json.loads((run/'RUN_MODE.json').read_text())
        self.assertEqual(mode['submission_update']['original_commit'], 'old-commit')
        self.assertEqual(mode['commit'], 'new-commit')
        self.assertFalse(mode['gpu_validation_verified']); self.assertFalse(mode['push_verified'])

    def test_running_job_update_is_refused(self):
        with patch.object(update_production, 'job_info', return_value=({'JobState': 'RUNNING'}, 'unit')), \
             self.assertRaisesRegex(RuntimeError, 'no longer pending'):
            update_production.pending_job('1234')

    def test_update_changes_only_comment_and_keeps_original_submission(self):
        for kind in ('gi', 'os', 'spin'):
            p = self.root/f'evidence/build/{kind}.json'; p.parent.mkdir(exist_ok=True)
            p.write_text(json.dumps({'clean_build': True}))
        checks = {'native_inputs_and_tools': {'passed': True, 'details': self.current},
                  'checkout': {'passed': True, 'details': {'commit': 'new-commit'}}}
        original_sha = sha(self.path)
        def info(job):
            p = self.root/'evidence/production_updates/1234.json'
            return {'Comment': 'S11_UPDATE_'+sha(p)}, 'unit after'
        with patch.object(update_production, 'SESSION', self.root), \
             patch.object(update_production.legacy, 'submission_location', return_value={}), \
             patch.object(update_production, 'pending_job', return_value=({}, 'unit before')), \
             patch.object(update_production, 'preflight', return_value=checks), \
             patch.object(update_production, 'job_info', side_effect=info), \
             patch.object(update_production.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', '')) as control, \
             contextlib.redirect_stdout(io.StringIO()):
            update_production.update('1234')
        control.assert_called_once()
        self.assertEqual(control.call_args.args[0][:3], ['scontrol', 'update', 'JobId=1234'])
        self.assertTrue(control.call_args.args[0][3].startswith('Comment=S11_UPDATE_'))
        self.assertEqual(len(control.call_args.args[0]), 4)
        self.assertEqual(sha(self.path), original_sha)
        self.assertFalse((self.root/'evidence/VALIDATION_PASS.json').exists())
        self.assertFalse((self.root/'evidence/PUSH_VERIFIED.json').exists())

    def test_retry_requires_latest_job_terminal_accounting_and_empty_queue(self):
        p = self.folder/'validation_unit.json'
        p.write_text(json.dumps(dict(epoch=1, returncode=0, stdout='Vista banner\n4321\n')))
        with patch.object(early_submit, 'SESSION', self.root):
            with patch.object(early_submit.legacy, 'command', return_value='PENDING'), \
                 self.assertRaisesRegex(RuntimeError, 'still in the queue'):
                early_submit.reviewed_retry('validation', '4321')
            with patch.object(early_submit.legacy, 'command', side_effect=['', '4321|FAILED|1:0|00:00:19|4|76|']):
                result = early_submit.reviewed_retry('validation', '4321')
            self.assertEqual(result['terminal_state'], 'FAILED')
            with patch.object(early_submit.legacy, 'command', side_effect=['', '4321|RUNNING|0:0|00:00:19|4|76|']), \
                 self.assertRaisesRegex(RuntimeError, 'not a confirmed'):
                early_submit.reviewed_retry('validation', '4321')
            p2 = self.folder/'validation_newer.json'; p2.write_text(json.dumps(dict(epoch=2, state='prepared')))
            with self.assertRaisesRegex(RuntimeError, 'latest'):
                early_submit.reviewed_retry('validation', '4321')

    def test_native_startup_uses_output_directory_as_cwd(self):
        with patch.object(launch, 'job_end', return_value=10**12), \
             patch.dict(os.environ, {'SLURM_JOB_ID': '4321'}), \
             patch.object(launch, 'manifest', return_value={'unit': True}), \
             patch.object(launch, 'endpoint', return_value={'cycle': 2}), \
             patch.object(launch.subprocess, 'Popen', return_value=Mock(wait=Mock(return_value=0))) as native:
            out = launch.run_case(self.root/'validation', 'compact_gi', 'gi', deck=Path('/unit.athinput'))
        self.assertEqual(native.call_args.kwargs['cwd'], out)
        command = json.loads((out.parent/'COMMAND.json').read_text())
        self.assertEqual(command['cwd'], str(out))


if __name__ == '__main__':
    unittest.main()
