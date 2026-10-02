#!/usr/bin/env python3
"""Critical orchestration and checkpoint tests; run through Slurm before release."""
import json,struct,tempfile,unittest,time
from pathlib import Path
from unittest.mock import patch
import runtime as r
import workflow as w

class Checks(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  (self.root/'control').mkdir();(self.root/'evidence').mkdir();(self.root/'runs').mkdir()
 def tearDown(self):self.tmp.cleanup()
 def test_real_header_layout_and_truncation(self):
  p=self.root/'fixture.rst';header=b'# preserved production header\n'*1800+b'<par_end>\n';offset=len(header);nmb=1
  with p.open('wb') as f:
   f.write(header);f.write(struct.pack('<ii',nmb,3));f.seek(offset+232);f.write(struct.pack('<ddi',.0125,.00625,2))
   f.seek(offset+252+nmb*20+56);f.write(struct.pack('<Q',25*40**3*8));f.seek(f.tell()+25*40**3*8)
   f.write(struct.pack('<4d',5000000.,0.,0.,0.));f.truncate(f.tell()+5000000*80)
  result=r.checkpoint(p,False)
  self.assertGreater(result['header_bytes'],40960);self.assertEqual(result['particles'],5000000)
  with p.open('r+b') as f:f.truncate(p.stat().st_size-1)
  with self.assertRaises(ValueError):r.checkpoint(p,False)
 def test_latest_three_and_active_write(self):
  known={}
  for i in range(5):
   run=self.root/'runs'/('segment_'+str(i));(run/'rst').mkdir(parents=True);p=run/'rst/a.rst';p.write_bytes(b'x')
   known[str(p)]=dict(path=str(p),cycle=i,time=float(i),sha256='fake',bytes=1)
  r.atomic(self.root/'control/checkpoints.json',known)
  kept=r.retain(self.root)
  self.assertEqual([x['cycle'] for x in kept],[2,3,4]);self.assertEqual(len(list((self.root/'runs').rglob('*.rst'))),3)
  self.assertTrue((self.root/'evidence/checkpoint_deletions.jsonl').exists())
 def test_finite_graph_and_duplicate_submit(self):
  with patch.object(w,'ROOT',self.root),patch.object(w,'CONTROL',self.root/'control'):
   w.save(dict(status='build_submitted',stop_requested=False,jobs=[dict(name='build',id=100,nodes=1,max_wall_hours=.5)],pending_submission=None))
   ids=iter(range(101,120))
   with patch.object(w.subprocess,'check_output',side_effect=lambda *args,**kw:str(next(ids))):w.submit()
   s=w.readstate();self.assertEqual(len(s['jobs']),9);self.assertEqual(s['maximum_reserved_node_hours'],44.5)
   with patch.object(w.subprocess,'check_output',side_effect=AssertionError('duplicate submission')):w.submit()
 def test_sticky_stop_and_cancel_scope(self):
  with patch.object(w,'ROOT',self.root),patch.object(w,'CONTROL',self.root/'control'):
   w.save(dict(status='configured',stop_requested=False,jobs=[dict(name='mine',id=123)],pending_submission=None))
   with patch.object(w.subprocess,'run') as cancel:w.request_stop(True)
   self.assertTrue(w.readstate()['stop_requested']);self.assertEqual(cancel.call_args[0][0],['scancel','123'])
   with self.assertRaises(RuntimeError):w.submit()
 def test_gate_rejects_changed_input(self):
  with patch.object(w,'ROOT',self.root),patch.object(w,'CONTROL',self.root/'control'):
   (self.root/'inputs').mkdir();(self.root/'inputs/gi_cluster_s8.athinput').write_text('changed')
   r.atomic(self.root/'control/config.json',dict(input_sha256='wrong'))
   with self.assertRaises(RuntimeError):w.bindings()
 def test_crash_window_recovers_unique_job(self):
  with patch.object(w.subprocess,'check_output',side_effect=['501|jn8_20261002_gate|\n','501|jn8_20261002_gate\n']):
   self.assertEqual(w.recover('jn8_20261002_gate'),501)
 def test_stale_archive_blocks_production(self):
  with patch.object(w,'ROOT',self.root),patch.object(w,'CONTROL',self.root/'control'),patch.object(w,'bindings'):
   w.save(dict(status='ready_segment_1',stop_requested=False,gates_passed=True,jobs=[]))
   with patch.dict(w.os.environ,{'SLURM_NNODES':'3','SLURM_NTASKS':'12'}):
    with self.assertRaisesRegex(RuntimeError,'heartbeat'):w.permit(1)
 def test_verified_endpoint_cancels_successors(self):
  run=self.root/'runs/segment_01';run.mkdir();(run/'EXIT_CODE').write_text('0');(run/'run.log').write_text('[conservation OK]')
  row=dict(path=str(run/'rst/a.rst'),time=12.,cycle=200,sha256='verified')
  with patch.object(w,'ROOT',self.root),patch.object(w,'CONTROL',self.root/'control'),patch.object(w,'bindings'),patch.object(w,'seal'),patch.object(w,'retain',return_value=[row]),patch.object(w,'scheduler_status',return_value=['123','COMPLETED','0:0','1','3']),patch.object(w.subprocess,'run') as cancel:
   w.save(dict(status='running_segment_1',stop_requested=False,time=10.,jobs=[dict(id=123),dict(id=124)]))
   with self.assertRaises(SystemExit):w.inspect(1,123)
   self.assertEqual(w.readstate()['status'],'complete_t12');self.assertTrue(cancel.called)
 def test_numerical_exit_is_terminal(self):
  run=self.root/'runs/segment_01';run.mkdir();(run/'EXIT_CODE').write_text('1')
  with patch.object(w,'ROOT',self.root),patch.object(w,'CONTROL',self.root/'control'),patch.object(w,'bindings'),patch.object(w,'scheduler_status',return_value=['123','COMPLETED','0:0','1','3']),patch.object(w.subprocess,'run'):
   w.save(dict(status='running_segment_1',stop_requested=False,time=1.,jobs=[dict(id=123)]))
   with self.assertRaises(SystemExit):w.inspect(1,123)
   self.assertEqual(w.readstate()['status'],'numerical_failure')

if __name__=='__main__':unittest.main()
