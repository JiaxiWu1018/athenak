import tempfile
import unittest
from pathlib import Path
from hourly import production_state,qualify_report

class HourlyStatusChecks(unittest.TestCase):
    def test_queued_report_uses_observed_state_without_claiming_evolution(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'REPORT.md'
            p.write_text('Hourly partial analysis; production continues.\nNative audit t=11.47.\n')
            health=dict(live_scheduler_state='PENDING',submission=dict(job_id='42'),sacct='42|PENDING|0|0|0:0\n')
            self.assertEqual(qualify_report(p,health),'PENDING')
            self.assertNotIn('production continues',p.read_text())
            self.assertIn('snapshot: PENDING',p.read_text())
            self.assertIn('Native audit t=11.47.',p.read_text())
    def test_terminal_scheduler_state_and_unknown_are_preserved(self):
        self.assertEqual(production_state(dict(live_scheduler_state='',submission=dict(job_id='42'),sacct='41|FAILED|0\n42|COMPLETED|10\n')),'COMPLETED')
        self.assertEqual(production_state(dict(live_scheduler_state='RUNNING')),'RUNNING')
        self.assertEqual(production_state({}),'unavailable')

if __name__=='__main__':unittest.main()
