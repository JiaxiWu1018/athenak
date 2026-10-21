#!/usr/bin/env python3
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import runtime

class RuntimeSafety(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        (self.root/'control').mkdir()
    def tearDown(self):self.temp.cleanup()
    def state(self,name,value):runtime.atomic(self.root/'control'/name,value)
    def fixture(self,time=2.,count=5,truncated=False):
        path=self.root/'state.rst'
        header=b'<job>\nbasename = fixture\n<par_end>\n'
        meta=bytearray(252)
        struct.pack_into('<ii',meta,0,1,0)
        struct.pack_into('<ddi',meta,232,time,.01,200)
        data=header+meta+bytes(20)+bytes(8+24)+struct.pack('<Q',25*2**3*8)
        data+=bytes(25*2**3*8)+struct.pack('<4d',count,0,0,0)+bytes(count*80)
        path.write_bytes(data[:-1] if truncated else data)
        return path
    def test_saved_ah_requires_nan_safe_verified_binary(self):
        from endpoint import saved_ah_executable
        gate=dict(executable='unchanged-production',executable_sha256='production-hash',
                  diagnostic_energy_repair=True)
        with self.assertRaises(RuntimeError):saved_ah_executable(self.root,gate)
        exe=self.root/'analysis';exe.write_bytes(b'analysis fixture')
        record=dict(status='pass',evolution_advances=0,bitwise_particle_check=True,
                    production_executable_sha256='production-hash',executable=str(exe),
                    executable_sha256=runtime.digest(exe))
        self.state('SAVED_AH_TOOLING.json',record)
        self.assertEqual(saved_ah_executable(self.root,gate),str(exe))
        exe.write_bytes(b'tampered')
        with self.assertRaises(RuntimeError):saved_ah_executable(self.root,gate)
        self.assertEqual(gate['executable'],'unchanged-production')
    def test_saved_ah_rejects_receipt_from_other_production(self):
        from endpoint import saved_ah_executable
        self.state('SAVED_AH_TOOLING.json',dict(status='pass',evolution_advances=0,
                   bitwise_particle_check=True,production_executable_sha256='other'))
        with self.assertRaises(RuntimeError):saved_ah_executable(self.root,dict(executable_sha256='current'))
    def test_aged_out_job_is_absent_from_live_queue(self):
        with patch('runtime.subprocess.check_output',return_value='42|RUNNING\n') as call:
            self.assertEqual(runtime.active_job_state('123'),'')
            self.assertNotIn('-j',call.call_args.args[0])
        with patch('runtime.subprocess.check_output',side_effect=RuntimeError('scheduler unavailable')):
            with self.assertRaises(RuntimeError):runtime.active_job_state('123')
    def test_unknown_energy_preserves_mass_but_physical_nan_fails(self):
        import numpy as np
        data=np.zeros((10,3));data[0]=[1.,2.,3.];data[9]=[0,3000000,4000000]
        data[1]=[1.,np.nan,np.inf]
        report=runtime.particle_diagnostic_quality(data)
        self.assertEqual(report['unknown_count'],2)
        self.assertEqual([c['unknown_rest_mass'] for c in report['components']],[0.,2.,3.])
        for field in (0,2,3,4,5,6,7,8,9):
            broken=data.copy();broken[field,0]=np.nan
            with self.assertRaises(RuntimeError):runtime.particle_diagnostic_quality(broken)
        self.assertTrue(np.isnan(data[1,1]))
    def test_failed_finalization_publishes_pause_and_never_submits(self):
        with patch('runtime.finish',side_effect=RuntimeError('audit failed')),patch('runtime.submit') as submit:
            with self.assertRaises(RuntimeError):runtime.finish_and_submit(self.root,self.root/'run',0)
            submit.assert_not_called()
        pause=json.loads((self.root/'control/PAUSED.json').read_text())
        self.assertIn('audit failed',pause['reason'])
    def test_endpoint_jobs_are_idempotent_and_analysis_only(self):
        from endpoint import submit_endpoint
        self.state('ENDPOINT_75.json',dict(time=75))
        self.state('submission.json',dict(nodes=6,capacity=256))
        with patch('endpoint.subprocess.check_output',side_effect=['100\n','101\n']) as call:
            submit_endpoint(self.root);submit_endpoint(self.root)
            self.assertEqual(call.call_count,2)
            self.assertIn('--nodes=6',call.call_args_list[0].args[0])
            self.assertIn('--dependency=afterany:100',call.call_args_list[1].args[0])
            self.assertTrue(call.call_args_list[0].args[0][-1].endswith('amd_saved_ah.sbatch'))
            self.assertTrue(call.call_args_list[1].args[0][-1].endswith('amd_products.sbatch'))
    def test_endpoint_rejects_early_or_ambiguous_submission(self):
        from endpoint import submit_endpoint
        self.state('ENDPOINT_75.json',dict(time=74))
        with self.assertRaises(RuntimeError):submit_endpoint(self.root)
        self.state('ENDPOINT_75.json',dict(time=75))
        self.state('submission.json',dict(nodes=6,capacity=256))
        self.state('endpoint_ah_submission.json',dict(status='prepared'))
        with patch('endpoint.subprocess.check_output') as call:
            with self.assertRaises(RuntimeError):submit_endpoint(self.root)
            call.assert_not_called()
    def test_preformation_failures_are_not_horizon_dropout(self):
        event=dict(failed_searches='1000')
        self.assertFalse(runtime.persistent_dropout(dict(ever_bits='0'),event,0))
        self.assertFalse(runtime.persistent_dropout(dict(ever_bits='1'),event,1))
        self.assertTrue(runtime.persistent_dropout(dict(ever_bits='1'),event,0))
        self.assertFalse(runtime.persistent_dropout(dict(ever_bits='1'),dict(failed_searches='3'),0))
    def test_checkpoint_schema(self):
        r=runtime.checkpoint(self.fixture(),False,5,1,2)
        self.assertEqual(r['particles'],5);self.assertEqual(r['cycle'],200)
    def test_truncation(self):
        with self.assertRaises(ValueError):runtime.checkpoint(self.fixture(truncated=True),False,5,1,2)
    def test_count_closure(self):
        with self.assertRaises(ValueError):runtime.checkpoint(self.fixture(),False,6,1,2)
    def test_beyond_endpoint(self):
        with self.assertRaises(ValueError):runtime.checkpoint(self.fixture(time=76),False,5,1,2)
    def test_unvalidated_never_submits(self):
        with patch('runtime.subprocess.check_output') as call:
            with self.assertRaises(RuntimeError):runtime.submit(self.root)
            call.assert_not_called()
    def test_endpoint_never_submits(self):
        self.state('VALIDATION_PASS.json',{})
        self.state('latest_checkpoint.json',dict(time=75))
        with patch('runtime.subprocess.check_output') as call:
            runtime.submit(self.root);call.assert_not_called()
    def test_pause_never_submits(self):
        self.state('PAUSED.json',dict(reason='AH dropout'))
        with patch('runtime.subprocess.check_output') as call:
            runtime.submit(self.root);call.assert_not_called()
    def test_no_second_fresh_trajectory(self):
        self.state('VALIDATION_PASS.json',{})
        self.state('submission.json',dict(job_id='123',segment=0))
        with patch('runtime.subprocess.check_output',return_value='') as call:
            with self.assertRaises(RuntimeError):runtime.submit(self.root)
            self.assertEqual(call.call_count,1)
    def test_current_active_job_not_duplicated(self):
        self.state('VALIDATION_PASS.json',{})
        self.state('submission.json',dict(job_id='999999999',segment=0))
        with patch('runtime.subprocess.check_output',return_value='999999999|RUNNING\n') as call:
            runtime.submit(self.root);self.assertEqual(call.call_count,1)
    def test_ambiguous_submission_does_not_retry(self):
        self.state('VALIDATION_PASS.json',{})
        self.state('submission_intent.json',dict(status='prepared',segment=0))
        with patch('runtime.subprocess.check_output') as call:
            with self.assertRaises(RuntimeError):runtime.submit(self.root)
            call.assert_not_called()

if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID'),'run checks in Slurm'
    unittest.main()
