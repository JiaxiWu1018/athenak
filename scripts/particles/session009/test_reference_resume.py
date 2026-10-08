"""Regression checks for campaign profile selection and safe saved-run acceptance."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import validate_initial_s7 as v
from validate_reference import check_receipt
from runtime import digest
import workflow

class ReferenceResume(unittest.TestCase):
 def test_session9_profile_is_read_and_missing_profile_is_not_substituted(self):
  with tempfile.TemporaryDirectory() as td:
   run=Path(td);(run/'a.part.vtk').touch()
   np.savetxt(run/'gi_profile_M076_two_clump_s9.txt',[[0,0,1],[30,0,1]])
   particles=dict(path=str(run/'a.part.vtk'),time=0,count=6,
    position=np.array([[0,0,0],[.1,0,0],[-3,0,0],[-3,.1,0],[3,0,0],[3,.1,0.]]),
    momentum=np.zeros((6,3)),tag_float=np.array([0,1,3000000,3000001,4000000,4000001.]),
    energy=np.ones(6),mass=np.ones(6))
   with patch.object(v,'read_particles',return_value=particles):
    stats,ledger=v.written_particle_stats(run,profile_name='gi_profile_M076_two_clump_s9.txt')
    self.assertEqual([r['count'] for r in ledger],[2,2,2])
    self.assertEqual(Path(stats['profile']).name,'gi_profile_M076_two_clump_s9.txt')
    with self.assertRaises(FileNotFoundError):v.written_particle_stats(run)

 def test_saved_acceptance_rejects_changed_bindings_and_sealed_metadata(self):
  with tempfile.TemporaryDirectory() as td:
   run=Path(td)
   for n,t in [('SEALED',''),('EXIT_CODE','0\n'),('run.log','saved successful run'),('SCIENCE_MANIFEST.json','[]')]:
    (run/n).write_text(t)
   c=dict(input_sha256='input',script_hashes={'validator':'tested'},resume_reference_checkpoint={'sha256':'checkpoint'})
   receipt=dict(passed=True,input_sha256='input',script_hashes={'validator':'tested'},checkpoint={'sha256':'checkpoint'},
    log_sha256=digest(run/'run.log'),manifest_sha256=digest(run/'SCIENCE_MANIFEST.json'))
   check_receipt(receipt,c,run)
   for k,value in [('passed',False),('input_sha256','other'),('script_hashes',{'validator':'old'}),('checkpoint',{'sha256':'other'})]:
    changed=copy.deepcopy(receipt);changed[k]=value
    with self.assertRaises(RuntimeError):check_receipt(changed,c,run)
   (run/'run.log').write_text('altered')
   with self.assertRaisesRegex(RuntimeError,'metadata changed'):check_receipt(receipt,c,run)

 def test_resume_submission_validates_saved_outputs_before_full_allocation(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'evidence').mkdir();(root/'evidence/wave_gate.json').write_text('{"passed":true}')
   s=dict(status='reference_reviewed',stop_requested=False,jobs=[])
   with patch.object(workflow,'ROOT',root),patch.object(workflow,'readstate',return_value=s),\
        patch.object(workflow,'config',return_value=dict(reuse_wave_gate=True,resume_reference=True)),\
        patch.object(workflow,'submit_one',side_effect=[1,2,3]) as submit,patch.object(workflow,'save'):
    workflow.submit()
   self.assertEqual(submit.call_args_list[0].args[2],'amd_validate_reference.sbatch')
   self.assertEqual(submit.call_args_list[1].args[3],['--resume-reference'])
   self.assertEqual(submit.call_args_list[1].args[-1],'afterok:1')

if __name__=='__main__':unittest.main()
