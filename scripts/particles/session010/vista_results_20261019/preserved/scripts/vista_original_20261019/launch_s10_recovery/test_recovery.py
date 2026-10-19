import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import recover
from telemetry_reader import StableSamplerReader


def sample(host='node0', used=60):
    return dict(host=host, epoch=time.time(), host_total_bytes=100, host_used_bytes=50,
                gpus=[dict(uuid='GPU-'+host, used_mib=used, total_mib=100)])


class Telemetry(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'telemetry'; self.root.mkdir()
        self.cfg = recover.base.config()
        (self.root / 'node0.json').write_text(json.dumps(sample()))

    def test_healthy_read_and_fixed_path_cache(self):
        reader = StableSamplerReader(retry_seconds=0)
        self.assertEqual(reader(self.root, 1, self.cfg)[0]['host'], 'node0')
        with patch.object(Path, 'glob', side_effect=AssertionError('must not relist during evolution')):
            self.assertEqual(reader(self.root, 1, self.cfg)[0]['host'], 'node0')

    def test_transient_json_read_is_retried_and_logged(self):
        reader = StableSamplerReader(retry_seconds=0); original = Path.read_text; calls = []
        def read(path, *args, **kwargs):
            if path.name == 'node0.json' and not calls:
                calls.append(1); return '{'
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', read):
            rows = reader(self.root, 1, self.cfg)
        self.assertEqual(len(rows), 1)
        events = [json.loads(x) for x in (self.root.parent/'telemetry_read_events.jsonl').read_text().splitlines()]
        self.assertEqual([x['event'] for x in events], ['read_error', 'read_recovered'])

    def test_persistent_read_error_stops_after_three_attempts(self):
        (self.root/'node0.json').write_text('{')
        with self.assertRaisesRegex(RuntimeError, 'after 3 attempts'):
            StableSamplerReader(retry_seconds=0)(self.root, 1, self.cfg)
        self.assertEqual(len((self.root.parent/'telemetry_read_events.jsonl').read_text().splitlines()), 3)

    def test_stale_measurement_still_stops(self):
        row = sample(); row['epoch'] -= 121
        (self.root/'node0.json').write_text(json.dumps(row))
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            StableSamplerReader(retry_seconds=0)(self.root, 1, self.cfg)

    def test_memory_threshold_still_stops_immediately(self):
        (self.root/'node0.json').write_text(json.dumps(sample(used=85)))
        rows = StableSamplerReader(retry_seconds=0)(self.root, 1, self.cfg)
        self.assertEqual(recover.base.stop_reason(rows, self.cfg), 'resource_limit_gpu_memory')
        self.assertFalse((self.root.parent/'telemetry_read_events.jsonl').exists())

    def test_explicit_gpu_sampler_error_is_not_hidden(self):
        row = sample(); row['error'] = 'NVML failed'; row.pop('gpus')
        (self.root/'node0.json').write_text(json.dumps(row))
        rows = StableSamplerReader(retry_seconds=0)(self.root, 1, self.cfg)
        self.assertEqual(recover.base.stop_reason(rows, self.cfg), 'telemetry_failure')

    def test_missing_cached_node_is_not_ignored(self):
        reader = StableSamplerReader(retry_seconds=0); reader(self.root, 1, self.cfg)
        (self.root/'node0.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'after 3 attempts'):
            reader(self.root, 1, self.cfg)


class Recovery(unittest.TestCase):
    def test_exactly_two_24_hour_gh_jobs(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            recover.base.atomic(run/'config.json', recover.base.config())
            def scheduler(args, **kwargs):
                calls.append(args)
                return subprocess.CompletedProcess(args, 0, f'Welcome to Vista\n{1000+len(calls)}\n', '')
            with patch.object(recover.base, 'sbatch_args', recover.recovery_args):
                recover.base.submit_chain(run, recover.SPECS, runner=scheduler)
        self.assertEqual(len(calls), 2)
        for cmd in calls:
            self.assertIn('--partition=gh', cmd); self.assertIn('--nodes=16', cmd)
            self.assertIn('--time=24:00:00', cmd); self.assertIn('--no-requeue', cmd)
            self.assertIn('--mail-user=jiaxiwu@caltech.edu', cmd)
            self.assertIn('--mail-type=BEGIN,END,FAIL', cmd)
            self.assertIn(str(recover.HERE/'job.sbatch'), cmd)
            self.assertNotIn('--partition=gh-dev', cmd)
        self.assertFalse(any(a.startswith('--dependency=') for a in calls[0]))
        self.assertIn('--dependency=afterok:1001', calls[1])

    def test_compute_submission_is_refused(self):
        with patch.object(recover.READY.__class__, 'exists', return_value=True), patch.object(recover.base, 'read', return_value={'run': '/tmp/recovery-test'}), patch.object(recover.socket, 'gethostname', return_value='i614-003.vista.tacc.utexas.edu'):
            with self.assertRaisesRegex(RuntimeError, 'login shell'):
                recover.submit()

    def fixture(self):
        cp = dict(time=55.00625, cycle=31488)
        result = dict(stop_reason='telemetry_failure', exit_code=0, checkpoint=cp,
                      memory=dict(gpu_peak_fractions={'gpu': .68}, host_peak_fractions={'host': .46}))
        return dict(status='telemetry_failure', checkpoint=cp), result

    def test_only_clean_monitoring_stop_is_authorized(self):
        state, result = self.fixture()
        self.assertEqual(recover.validate_stop(state, result)['cycle'], 31488)
        for reason in ['user_stop', 'numerical_failure', 'resource_limit_gpu_memory']:
            with self.assertRaises(RuntimeError):
                recover.validate_stop(dict(state, status=reason), result)
        with self.assertRaises(RuntimeError):
            recover.validate_stop(state, dict(result, exit_code=1))

    def test_actual_memory_limit_cannot_be_bypassed(self):
        state, result = self.fixture(); result['memory']['gpu_peak_fractions']['gpu'] = .86
        with self.assertRaises(RuntimeError): recover.validate_stop(state, result)

    def test_accounting_retains_prior_jobs_without_double_count(self):
        old = dict(rows=[dict(job_id='1', node_hours=10)], final=True)
        new = dict(rows=[dict(job_id='2', node_hours=20)], final=False)
        def read(path):
            return old if Path(path).name == 'PREVIOUS_ACCOUNTING.json' else dict(jobs=[dict(job_id='2')])
        with patch.object(Path, 'exists', return_value=True), patch.object(recover.base, 'read', side_effect=read), patch.object(recover, 'ORIGINAL_ACCOUNTING', return_value=new):
            result = recover.combined_accounting(Path('/tmp/run'))
        self.assertEqual(result['actual_node_hours'], 30)
        self.assertFalse(result['final'])

    def test_unsubmitted_recovery_does_not_count_current_gg_allocation(self):
        old = dict(rows=[dict(job_id='1', node_hours=10)], final=True)
        with patch.object(Path, 'exists', return_value=False), patch.object(recover.base, 'read', return_value=old), patch.object(recover, 'ORIGINAL_ACCOUNTING') as accounting:
            result = recover.combined_accounting(Path('/tmp/run'))
            accounting.assert_not_called()
        self.assertEqual(result['actual_node_hours'], 10)


if __name__ == '__main__': unittest.main()
