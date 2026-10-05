#!/usr/bin/env python3
import importlib.util,json,sys,tempfile,time,types,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import recovery as r
spec=importlib.util.spec_from_file_location('recovery_archive_trigger',Path(__file__).with_name('archive_trigger.py'))
trigger=importlib.util.module_from_spec(spec);spec.loader.exec_module(trigger)

class RecoveryChecks(unittest.TestCase):
 def test_actual_spending_and_remaining_reservations(self):
  jobs=[dict(id=1,nodes=3,max_wall_hours=4),dict(id=2,nodes=3,max_wall_hours=4),dict(id=3,nodes=3,max_wall_hours=4)]
  with patch.object(r.subprocess,'check_output',return_value='1|COMPLETED|0:0|3600|3\n2|PENDING|0:0|0|0\n3|FAILED|1:0|26|3\n'):
   b=r.budget(jobs)
  self.assertAlmostEqual(b['actual_node_hours'],3+26*3/3600)
  self.assertAlmostEqual(b['maximum_possible_node_hours'],15+26*3/3600)

 def test_submission_accounting_lag_reserves_full_active_job(self):
  jobs=[dict(id=1,nodes=3,max_wall_hours=4),dict(id=2,nodes=3,max_wall_hours=4)]
  with patch.object(r.subprocess,'check_output',side_effect=['1|COMPLETED|0:0|3600|3\n','2|PENDING\n']):b=r.budget(jobs)
  self.assertEqual(b['maximum_possible_node_hours'],15)
  self.assertEqual(b['active_jobs_with_unmeasured_elapsed'],[2])

 def test_user_stop_is_not_cleared(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);control=root/'control';control.mkdir();(control/'USER_STOP').touch();(control/'REQUEST_STOP').touch()
   s=dict(status='scheduler_failure',gates_passed=True,segments_completed=2,stop_requested=False)
   with patch.object(r,'ROOT',root),patch.object(r,'CONTROL',control),patch.object(r,'CONFIG',control/'recovery.json'),patch.object(r.w,'readstate',return_value=s):
    with self.assertRaisesRegex(RuntimeError,'stop requires review'):r.setup()
   self.assertTrue((control/'REQUEST_STOP').exists())

 def test_reviewed_resume_preserves_failure_and_cap(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);control=root/'control';control.mkdir();(root/'evidence').mkdir();here=root/'recovery';here.mkdir();(here/'file.py').write_text('fixture')
   run=root/'runs/segment_03';run.mkdir(parents=True);(run/'run.log').write_text('selected pml ob1 selected pml ucx MPI_INIT has failed')
   for name in ('ARCHIVE_ERROR','REQUEST_STOP'):(control/name).touch()
   s=dict(status='scheduler_failure',gates_passed=True,segments_completed=2,stop_requested=False,jobs=[],checkpoint=dict(time=9.325))
   b=dict(actual_node_hours=21.6458333333333,maximum_possible_node_hours=21.6458333333333)
   with patch.object(r,'ROOT',root),patch.object(r,'CONTROL',control),patch.object(r,'CONFIG',control/'recovery.json'),patch.object(r,'HERE',here),patch.object(r.w,'readstate',return_value=s),patch.object(r.w,'bindings'),patch.object(r.w,'save'),patch.object(r,'budget',return_value=b):r.setup()
   c=json.loads((control/'recovery.json').read_text())
   self.assertAlmostEqual(c['additional_max_node_hours'],25.5)
   self.assertLess(s['maximum_possible_total_node_hours'],48)
   self.assertEqual(s['status'],'recovery_prepared')
   self.assertTrue((root/'evidence/recovery_20261005/prior_REQUEST_STOP').exists())

 def check_inspection(self,failed=False):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);control=root/'control';control.mkdir();run=root/'runs/segment_03_recovery_20261005';run.mkdir(parents=True)
   (run/'EXIT_CODE').write_text('1' if failed else '0');(run/'run.log').write_text('### FATAL ERROR' if failed else '[conservation OK]')
   s=dict(status='running_recovery_1',stop_requested=False,jobs=[dict(name='recovery_20261005_continuation1',id=123)])
   row=dict(path=str(run/'rst/fake.rst'),cycle=4000,time=12.)
   acct=['123','FAILED' if failed else 'COMPLETED','1:0' if failed else '0:0','100','3']
   with patch.object(r,'ROOT',root),patch.object(r,'CONTROL',control),patch.object(r.w,'readstate',return_value=s),patch.object(r.w,'save'),patch.object(r.w,'scheduler_status',return_value=acct),patch.object(r,'retain',return_value=[row]),patch.object(r,'seal'),patch.object(r,'guard',return_value=(s,dict(old_checkpoint=dict(time=9.325)),{})),patch.object(r.w,'cancel_future') as cancel:
    with self.assertRaises(SystemExit):r.inspect(1)
    cancel.assert_called_once()
   return s

 def test_endpoint_cancels_unused_continuation(self):
  s=self.check_inspection();self.assertEqual(s['status'],'complete_t12');self.assertEqual(s['recovery_continuations_completed'],1)

 def test_numerical_failure_stops_without_retry(self):
  self.assertEqual(self.check_inspection(True)['status'],'numerical_failure')

 def test_completed_archive_job_missing_from_queue_uses_accounting(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);e=root/'evidence';e.mkdir()
   (e/'trigger_config.json').write_text(json.dumps(dict(deadline=time.time()+3600,max_jobs=4,max_gpu_hours=24)))
   (e/'active_archive_jobs.json').write_text(json.dumps(dict(jobs=[dict(id=2134)],pending=None)))
   def remote(cmd):
    if cmd.startswith('cat '):return json.dumps(dict(status='recovery_prepared'))
    if cmd.startswith('find '):return '/work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/runs/segment_01\n'
    return ''
   calls=[]
   def command(cmd,**kwargs):
    calls.append(cmd)
    if cmd[0]=='squeue':return ''
    if cmd[0]=='sacct':return '2134|COMPLETED|0:0\n'
    if cmd[0]=='sbatch':return '2135\n'
    raise AssertionError(cmd)
   with patch.object(trigger,'DEST',root),patch.object(trigger,'remote',side_effect=remote),patch.object(trigger.subprocess,'check_output',side_effect=command),patch.object(trigger.os,'statvfs',return_value=types.SimpleNamespace(f_bavail=1024**4,f_frsize=1)):trigger.tick()
   jobs=json.loads((e/'active_archive_jobs.json').read_text())
   self.assertEqual(jobs['jobs'][-1]['id'],2135)
   self.assertTrue(any(cmd[0]=='sacct' for cmd in calls))
   self.assertFalse(any(cmd[0]=='squeue' and '-j' in cmd for cmd in calls))

if __name__=='__main__':unittest.main()
