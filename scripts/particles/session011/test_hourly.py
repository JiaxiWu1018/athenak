import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import hourly

class HourlySafety(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def test_snapshot_excludes_future_and_incomplete_records(self):
        source=self.root/'source.csv';out=self.root/'out.csv'
        source.write_bytes(b'cycle,time,count\n10,1.0,7\n11,1.1,8\n12,1.2,9\n13,1.3,')
        hourly.csv_snapshot(source,out,dict(time=1.1,cycle=11))
        with out.open() as f:rows=list(csv.DictReader(f))
        self.assertEqual([int(r['cycle']) for r in rows],[10,11])
        self.assertTrue(source.read_bytes().endswith(b'13,1.3,'))
    def test_event_state_time_controls_cutoff(self):
        source=self.root/'source.csv';out=self.root/'out.csv'
        source.write_text('cycle,time,state_cycle,state_time\n9,.9,10,1.0\n10,1.,11,1.1\n')
        hourly.csv_snapshot(source,out,dict(time=1.0,cycle=10))
        with out.open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['state_cycle'],'10')
    def test_schedule_is_hour_aligned_and_idempotent(self):
        with patch('hourly.time.time',return_value=3601.),patch('hourly.subprocess.check_output',return_value='42\n') as call:
            value=hourly.schedule(self.root)
            self.assertEqual(value['due_epoch'],7200.)
            self.assertIn('--begin=now+3599seconds',call.call_args.args[0])
        with patch('hourly.active_job_state',return_value='PENDING'),patch('hourly.subprocess.check_output') as call:
            self.assertEqual(hourly.schedule(self.root)['job_id'],'42');call.assert_not_called()
    def test_hourly_native_anchor_and_live_cutoff_have_distinct_labels(self):
        import products
        source=Path(products.__file__).read_text()
        self.assertIn("endpoint_checkpoint=checkpoint_latest",source)
        self.assertIn('Sampled live diagnostic state',source)
        self.assertIn('latest fully audited native checkpoint'.lower(),source.lower())

if __name__=='__main__':unittest.main()
