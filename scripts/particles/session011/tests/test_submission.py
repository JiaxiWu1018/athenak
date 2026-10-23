"""Non-science regression tests; every scheduler/application invocation is mocked."""
import contextlib
import csv
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import launch
import output_audit
import submit
import workflow
import prepare
from checkpoint_io import audit_particles
from prepare import parse, sha


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='s11-unit-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def csv(self, name, fields, rows):
        path = self.root / name
        with path.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        return path

    def test_actual_input_contract(self):
        workflow.check_inputs()
        for name in ('os_validation', 'spin_validation'):
            self.assertEqual(parse((HERE / f'inputs/{name}.athinput').read_text())['mesh']['nx2'], '64')
        self.assertEqual(parse((HERE / 'inputs/production.athinput').read_text())['output7']['last_time'], '-1.0')

    def test_malformed_validation_input_rejected(self):
        (self.root / 'inputs').mkdir()
        (self.root / 'inputs/os_validation.athinput').write_text('<mesh>\nnx1=64\nx2=64\nnx3=64\n')
        with patch.object(workflow, 'HERE', self.root), self.assertRaisesRegex(RuntimeError, 'nx2'):
            workflow.check_inputs()

    def test_generator_reproduces_corrected_inputs_without_touching_real_receipts(self):
        import shutil
        (self.root / 'inputs').mkdir()
        for name in ('session010_reference.athinput', 'compact_reference.athinput'):
            shutil.copyfile(HERE / 'inputs' / name, self.root / 'inputs' / name)
        with patch.object(prepare, 'HERE', self.root), patch.object(prepare, 'SESSION', self.root), \
             patch.object(prepare, 'source_hash', return_value='unit-source'), \
             patch.object(prepare.subprocess, 'check_output', return_value='unit-commit'):
            prepare.generate()
        for name in ('production', 'compact_validation', 'os_validation', 'spin_validation'):
            self.assertEqual((self.root / f'inputs/{name}.athinput').read_text(),
                             (HERE / f'inputs/{name}.athinput').read_text())

    def test_existing_builds_match_and_manifest_binds_every_input_and_executable(self):
        result = workflow.manifest()
        self.assertEqual(set(result['executable_sha256']), {'gi', 'os', 'spin'})
        for path in (HERE / 'inputs').glob('*.athinput'):
            self.assertEqual(result['files'][str(path.relative_to(workflow.CODE))], sha(path))
        self.assertIn('scripts/particles/session011/launch.py', result['files'])
        self.assertEqual(len(result['cases']), 5)

    def test_legacy_validation_receipt_rejected(self):
        (self.root / 'evidence').mkdir()
        (self.root / 'evidence/VALIDATION_PASS.json').write_text(json.dumps({'passed': True}))
        with patch.object(workflow, 'SESSION', self.root), self.assertRaisesRegex(RuntimeError, 'stale'):
            workflow.validated({'schema_version': 2})

    def test_retained_validation_artifact_changes_are_rejected(self):
        # PASS reads are mocked. No passing validation receipt is written anywhere.
        current = {'schema_version': 2, 'unit_fixture_only': True}
        root = self.root / 'runtime/validation_1234'; root.mkdir(parents=True)
        artifacts = {}
        for label in workflow.CASES:
            case = root / label; case.mkdir()
            for name, value in [('COMMAND.json', '{}'), ('athena.log', 'UNIT TEST ONLY'),
                                ('RESULT.json', json.dumps({'passed': True, 'job_id': '1234'}))]:
                path = case / name; path.write_text(value)
                artifacts[str(path.relative_to(root))] = sha(path)
        cp = root / 'checkpoint.unit'; cp.write_text('UNIT FIXTURE, NOT A CHECKPOINT')
        receipt = dict(passed=True, manifest=current, validation_job='1234', outputs=str(root),
                       artifacts=artifacts, checkpoint=str(cp), checkpoint_sha256=sha(cp))
        original_read = Path.read_text
        def read(path, *args, **kwargs):
            if path.name in ('PASS.json', 'VALIDATION_PASS.json'):
                return json.dumps(receipt)
            return original_read(path, *args, **kwargs)
        with patch.object(workflow, 'SESSION', self.root), patch.object(Path, 'read_text', read):
            self.assertEqual(workflow.validated(current)['validation_job'], '1234')
            (root / 'spin_fine/athena.log').write_text('changed fixture')
            with self.assertRaisesRegex(RuntimeError, 'artifact'):
                workflow.validated(current)

    def test_changed_push_receipt_or_commit_rejected(self):
        (self.root / 'evidence').mkdir()
        validation = self.root / 'evidence/VALIDATION_PASS.json'; validation.write_text('{}')
        push = dict(passed=True, manifest={'current': True}, validation_job='1234',
                    validation_receipt_sha256=sha(validation), commit='old')
        (self.root / 'evidence/PUSH_VERIFIED.json').write_text(json.dumps(push))
        with patch.object(workflow, 'SESSION', self.root), patch.object(workflow, 'manifest', return_value={'current': True}), \
             patch.object(workflow, 'validated', return_value={'validation_job': '1234'}), \
             patch.object(workflow, 'git', return_value='new'), self.assertRaisesRegex(RuntimeError, 'push'):
            workflow.ready()

    def test_zero_exit_does_not_replace_required_checkpoint_endpoint(self):
        (self.root / 'rst').mkdir()
        path = self.root / 'rst/unit.rst'
        path.write_bytes(b'<par_end>\n' + bytes(232) + struct.pack('<ddi', 39., .01, 100))
        with self.assertRaisesRegex(RuntimeError, 'endpoint'):
            workflow.endpoint(self.root, target_time=40.)
        self.assertEqual(workflow.endpoint(self.root, target_time=39.)['cycle'], 100)

    def test_immediate_or_inherited_persistence_rejected(self):
        fields = ['quality_streak', 'quality_geometry_ok', 'published_this_candidate']
        for streak in (1, 3):
            path = self.csv('consumer.csv', fields, [dict(zip(fields, (streak, 1, 1)))])
            with self.assertRaises(RuntimeError):
                workflow.fresh_persistence(path)

    def test_three_fresh_hits_accepted(self):
        fields = ['quality_streak', 'quality_geometry_ok', 'published_this_candidate']
        path = self.csv('consumer.csv', fields, [dict(zip(fields, (i, 1, int(i == 3)))) for i in (1, 2, 3)])
        workflow.fresh_persistence(path)

    def test_invalid_published_horizon_rejected(self):
        row = dict(published='1', found='1', streak='3', mass='1', spin='.5', area='10',
                   hmean='0', rmin='1', rmax='2', local_dx='.1')
        self.assertEqual(workflow.horizon_quality([row]), [row])
        for key, value in [('mass', 'nan'), ('hmean', '3'), ('streak', '2'), ('rmin', '.1')]:
            with self.assertRaises(RuntimeError):
                workflow.horizon_quality([row | {key: value}])

    def test_full_size_census_requires_actual_t0_and_all_components(self):
        rows = [dict(component=str(i), count=str(n), cycle='0', time='0', leaf_blocks='4936', rest_mass='.1')
                for i, n in enumerate((3000000, 1000000, 1000000))]
        self.assertEqual(workflow.startup_census(reversed(rows))['leaf_blocks'], 4936)
        for changes in ({'time': '.1'}, {'leaf_blocks': '400'}, {'rest_mass': 'nan'}, {'component': '0'}):
            bad = [dict(r) for r in rows]; bad[2].update(changes)
            with self.assertRaises(RuntimeError):
                workflow.startup_census(bad)

    def test_incomplete_live_startup_csv_is_deferred(self):
        path = self.root / 'initial.csv'
        path.write_text('cycle,time\n0,0\n0,0\n0,')
        self.assertEqual(launch.initial_rows(path), [])
        path.write_text('cycle,time\n0,0\n0,0\n0,0\n')
        self.assertEqual(len(launch.initial_rows(path)), 3)

    def test_native_particle_audit_checks_tags_weights_and_gids(self):
        import numpy as np
        data = np.zeros((10, 3), dtype='<f8')
        data[0] = [1., .1, .1]; data[9] = [0, 3000000, 4000000]
        path = self.root / 'particle-table.unit'; data.tofile(path)
        cp = dict(path=str(path), particles=3, blocks=1, bytes=path.stat().st_size)
        self.assertEqual(audit_particles(cp)['component_counts'], [1, 1, 1])
        for row, col, bad_value in ((9, 1, 0), (0, 1, -1), (8, 1, 2), (9, 1, 1.5)):
            bad = data.copy(); bad[row, col] = bad_value; bad.tofile(path)
            with self.assertRaisesRegex(RuntimeError, 'audit failed'):
                audit_particles(cp)
        self.assertEqual(audit_particles(dict(particles=0))['particles'], 0)

    def test_missing_or_partial_diagnostic_state_rejected(self):
        fields = ['cycle', 'time', 'Jz']
        path = self.csv('data.csv', fields, [dict(cycle=0, time=0, Jz=0)] * 3
                        + [dict(cycle=1, time=.025, Jz=0)] * 2)
        with self.assertRaisesRegex(RuntimeError, 'Incomplete final'):
            output_audit.csv_states(path, group_rows=3)
        path = self.csv('data.csv', fields, [dict(cycle=0, time=0, Jz='nan')])
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite'):
            output_audit.csv_states(path, finite=('Jz',))

    def test_coverage_reports_wave_tail_without_exact_endpoint_claim(self):
        result = output_audit.coverage([.025, .05, .075], .025, 0, .1, initial=False, final=False)
        self.assertAlmostEqual(result['endpoint_gap'], .025)
        with self.assertRaisesRegex(RuntimeError, 'final'):
            output_audit.coverage([0., .025], .025, 0, .05)
        with self.assertRaisesRegex(RuntimeError, 'cadence'):
            output_audit.coverage([0., .075, .1], .025, 0, .1)

    def test_truncated_binary_payload_rejected(self):
        header = ('Athena binary output version=1.1\n  time=0\n  cycle=0\n'
                  '  size of location=8\n  size of variable=4\n  number of variables=1\n'
                  '  header offset=0\n').encode()
        path = self.root / 'unit.bin'
        payload = struct.pack('<6i', 0, 0, 0, 0, 0, 0) + bytes(64) + bytes(4)
        path.write_bytes(header + payload)
        self.assertEqual(output_audit.binary_state(path), (0., 0))
        path.write_bytes(header + payload[:-1])
        with self.assertRaisesRegex(RuntimeError, 'Truncated'):
            output_audit.binary_state(path)

    def test_missing_or_duplicate_removal_records_rejected(self):
        path = self.root / 'ledger.csv'
        self.assertEqual(output_audit.removal_ledger(path, 5000000)['records'], 0)
        with self.assertRaisesRegex(RuntimeError, 'Missing'):
            output_audit.removal_ledger(path, 4999999)
        path.write_text('# cycle,time,tag,reason,x1,x2,x3,u1_cov,u2_cov,u3_cov,component,mass\n'
                        '1,.1,0,lapse,0,0,0,0,0,0,0,.1\n')
        self.assertEqual(output_audit.removal_ledger(path, 4999999)['records'], 1)
        with path.open('a') as stream:
            stream.write('1,.1,0,lapse,0,0,0,0,0,0,0,.1\n')
        with self.assertRaisesRegex(RuntimeError, 'duplicate'):
            output_audit.removal_ledger(path, 4999998)

    def test_surface_quadrature_requires_exactly_one_owner_per_angle(self):
        base = 'unit'
        fields = ['cycle', 'time', 'published', 'angle', 'r']
        for rank in range(2):
            self.csv(f'{base}.ah_quadrature_0.rank{rank}.csv', fields,
                     [dict(cycle=1, time=.025, published=1, angle=i, r=1.) for i in range(rank, 200, 2)])
        result = output_audit.quadrature(self.root, base, 0, {(1, .025)}, 2)
        self.assertEqual(result['published_states'], 1)
        path = self.root / f'{base}.ah_quadrature_0.rank1.csv'
        with path.open('a') as stream:
            stream.write('1,.025,1,0,1.\n')
        with self.assertRaisesRegex(RuntimeError, 'Duplicate owner'):
            output_audit.quadrature(self.root, base, 0, {(1, .025)}, 2)

    def test_vtk_layout_and_truncation(self):
        path = self.root / 'unit.vtk'
        data = (b'# vtk DataFile Version 2.0\n# AthenaK particle data at time= 0 nranks= 4 cycle=0\n'
                b'BINARY\nDATASET UNSTRUCTURED_GRID\n\nPOINTS 1 float\n' + bytes(12) + b'\nPOINT_DATA 1\n')
        for name in ('gid', 'ptag', 'prtcl_vel', 'prtcl_energy', 'prtcl_mass'):
            if name == 'prtcl_vel':
                data += b'\nVECTORS prtcl_vel float\n' + bytes(12)
            else:
                data += f'\nSCALARS {name} float\nLOOKUP_TABLE default\n'.encode() + bytes(4)
        path.write_bytes(data)
        self.assertEqual(output_audit.vtk_state(path), (0., 0))
        path.write_bytes(data[:-1])
        with self.assertRaisesRegex(RuntimeError, 'payload'):
            output_audit.vtk_state(path)

    def test_final_success_requires_startup_telemetry_coverage_and_unchanged_manifest(self):
        args = [0, None, {'time': 150.}, True, 1, {'passed': True}, True]
        self.assertTrue(launch.production_complete(*args))
        for index, value in [(0, 1), (1, 'memory'), (2, {'time': 149.}), (3, False),
                             (4, 0), (5, {'passed': False}), (6, False)]:
            bad = list(args); bad[index] = value
            self.assertFalse(launch.production_complete(*bad))

    def test_native_launch_forbidden_in_gg(self):
        with patch.dict(os.environ, {'SLURM_JOB_PARTITION': 'gg'}), \
             patch.object(launch.subprocess, 'Popen') as native, self.assertRaises(RuntimeError):
            launch.require_gpu('gh-dev', 4)
        native.assert_not_called()

    def test_check_default_never_submits_or_writes(self):
        with patch.object(submit, 'SESSION', self.root), patch.object(submit, 'preflight', return_value={'test': {'passed': True}}), \
             patch.object(sys, 'argv', ['submit.py']), patch.object(submit.subprocess, 'run') as run, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(submit.main(), 0)
        run.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_submission_refuses_compute_node(self):
        with patch.object(submit.socket, 'getfqdn', return_value='i614-064.vista.tacc.utexas.edu'), \
             patch.object(submit.subprocess, 'run') as run, self.assertRaisesRegex(RuntimeError, 'login'):
            submit.submit('validation')
        run.assert_not_called()

    def test_submission_saves_job_id_and_submits_exactly_one_stage(self):
        def run(args, **kwargs):
            if '--test-only' in args:
                return subprocess.CompletedProcess(args, 0, '', 'test only')
            self.assertIn('validate.sbatch', args[-1])
            self.assertNotIn('--export', args)
            self.assertEqual(kwargs['env']['S11_ROOT'], str(self.root))
            return subprocess.CompletedProcess(args, 0, '1234\n', '')
        with patch.object(submit, 'SESSION', self.root), patch.object(submit.socket, 'getfqdn', return_value='login1.vista.tacc.utexas.edu'), \
             patch.object(submit, 'preflight', return_value={'test': {'passed': True}}), \
             patch.object(submit.subprocess, 'run', side_effect=run) as scheduler, \
             patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
            submit.submit('validation')
        self.assertEqual(scheduler.call_count, 2)  # one test-only request, one actual submission
        records = list((self.root / 'evidence/submissions').glob('*.json'))
        self.assertEqual(len(records), 1)
        record = json.loads(records[0].read_text())
        self.assertEqual(record['state'], 'accepted'); self.assertEqual(record['job_id'], '1234')
        self.assertEqual(record['runtime'], str(self.root / 'runtime/validation_1234'))

    def test_ambiguous_submission_is_retained_and_blocks_retry(self):
        def run(args, **kwargs):
            if '--test-only' in args:
                return subprocess.CompletedProcess(args, 0, '', '')
            raise subprocess.TimeoutExpired(args, 30)
        with patch.object(submit, 'SESSION', self.root), patch.object(submit.socket, 'getfqdn', return_value='login1.vista.tacc.utexas.edu'), \
             patch.object(submit, 'preflight', return_value={'test': {'passed': True}}), \
             patch.object(submit.subprocess, 'run', side_effect=run) as scheduler, \
             patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.TimeoutExpired):
                submit.submit('validation')
            with self.assertRaisesRegex(RuntimeError, 'already exists'):
                submit.submit('validation')
        self.assertEqual(scheduler.call_count, 2)
        record = json.loads(next((self.root / 'evidence/submissions').glob('*.json')).read_text())
        self.assertEqual(record['state'], 'ambiguous')

    def test_failed_preflight_never_contacts_sbatch(self):
        with patch.object(submit, 'SESSION', self.root), patch.object(submit.socket, 'getfqdn', return_value='login1.vista.tacc.utexas.edu'), \
             patch.object(submit, 'preflight', return_value={'test': {'passed': False, 'error': 'missing validation'}}), \
             patch.object(submit.subprocess, 'run') as scheduler, \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'Preflight'):
            submit.submit('production')
        scheduler.assert_not_called()

    def test_live_partition_limit_is_enforced(self):
        with patch.object(submit, 'command', return_value='State=UP MaxNodes=8 MinNodes=1 MaxTime=02:00:00'):
            submit.partition('validation')
            with self.assertRaises(RuntimeError):
                submit.partition('production')
        self.assertEqual(submit.duration('1-12:00:00'), 36 * 3600)

    def test_site_balance_and_membership(self):
        (self.root / 'project.map').write_text('AST21006 123\n')
        (self.root / 'projectuser.map').write_text('unituser 123\n')
        (self.root / 'usage.map').write_text('AST21006:Vista:2020-01-01:2099-01-01:1000:500:500\n')
        (self.root / 'projectbalance.map').write_text('123 500\n')
        with patch.object(submit.getpass, 'getuser', return_value='unituser'), patch.object(submit.os, 'getgroups', return_value=[123]):
            self.assertEqual(submit.allocation('validation', self.root)['available_su'], 500.)
            with self.assertRaisesRegex(RuntimeError, '576 required'):
                submit.allocation('production', self.root)


if __name__ == '__main__':
    unittest.main()
