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
        with patch('runtime.subprocess.check_output',return_value='RUNNING') as call:
            runtime.submit(self.root);self.assertEqual(call.call_count,1)

if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID'),'run checks in Slurm'
    unittest.main()
