"""Targeted restart-retention, resource, gap and strain numerical invariants."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import runtime,workflow
from analyze_s9 import orbital_series
from science9 import ffi,ringdown_review
from mesh_audit import audit
class Session9(unittest.TestCase):
 def test_unarchived_checkpoints_never_deleted(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'control').mkdir();(root/'evidence').mkdir();known={}
   for i in range(5):
    p=root/'runs'/f'run{i}'/'rst'/'a.rst';p.parent.mkdir(parents=True);p.write_text('verified fixture')
    known[str(p)]=dict(path=str(p),cycle=i,sha256=f'h{i}',time=i)
   runtime.atomic(root/'control/checkpoints.json',known)
   runtime.retain(root)
   self.assertTrue(all(Path(p).exists() for p in known))
   runtime.atomic(root/'control/archived_checkpoints.json',{p:dict(sha256=r['sha256'],destination='verified Anta fixture') for p,r in known.items()})
   runtime.retain(root)
   self.assertEqual([Path(p).exists() for p in known],[False,False,True,True,True])
 def test_active_allocation_reserved_at_full_walltime(self):
  jobs=[dict(id=1,nodes=12,max_wall_hours=12),dict(id=2,nodes=1,max_wall_hours=1)]
  with patch('workflow.subprocess.check_output',side_effect=['1|RUNNING|0:0|3600|12\n2|COMPLETED|0:0|600|1\n','1|RUNNING\n']):
   b=workflow.budget(jobs)
  self.assertAlmostEqual(b['actual_node_hours'],12+1/6)
  self.assertAlmostEqual(b['maximum_possible_node_hours'],144+1/6)
 def test_gap_does_not_create_orbital_phase(self):
  t=np.array([0,1,2,10,11,12.]);angle=np.array([0,.1,.2,3,3.1,3.2]);v=np.c_[np.cos(angle),np.sin(angle),np.zeros(6)]
  d,p,w,dr,r,groups=orbital_series(t,v,np.ones(6,bool))
  self.assertEqual(len(groups),2);self.assertTrue(np.allclose(w,.1));self.assertTrue(np.allclose(dr,0))
 def test_ffi_recovers_supported_frequency_interior(self):
  t=np.arange(0,200,.025);f=.1;h=np.exp(2j*np.pi*f*t);z=-(2*np.pi*f)**2*h
  q,integrated=ffi(t,z,.01);middle=(q>40)&(q<160)
  # Edge detrending leaves low-frequency terms; compare the frequency coefficient.
  coefficient=np.mean(integrated[middle]*np.exp(-2j*np.pi*f*q[middle]))
  self.assertLess(abs(coefficient-1),.02)
 def test_no_common_horizon_cannot_trigger_ringdown_stop(self):
  r=ringdown_review(np.arange(200),np.ones((200,77),complex),[],dict(passed=True))
  self.assertFalse(r['ringdown_usable']);self.assertFalse(r['common_encloses_both'])
 def test_wave_floor_hole_rejected(self):
  data=dict(mb_geometry=np.array([[32,48,0,16,0,16]]))
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'evidence').mkdir()
   with self.assertRaisesRegex(RuntimeError,'propagation floor'):audit(root,data)
if __name__=='__main__':unittest.main()
