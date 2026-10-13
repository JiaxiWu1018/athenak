import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from progress_accounting import finalize,allocated_node_hours
import workflow as wf

class Stops(unittest.TestCase):
 def test_resource_stop_promotes_final_checkpoint(self):
  cp=dict(path="final.rst",sha256="a"*64,bytes=64575856941,time=9.33125,cycle=2256)
  old=dict(time=.03125,cycle=5,segments_completed=0)
  new=finalize(old,cp,dict(actual_node_hours=72.7),"resource_limit",1)
  self.assertEqual(new["cycle"],2256);self.assertEqual(new["segments_completed"],1);self.assertEqual(old["cycle"],5)
 def test_user_stop_keeps_verified_progress(self):
  cp=dict(path="final.rst",sha256="b"*64,bytes=1,time=12.,cycle=100)
  self.assertEqual(finalize({},cp,{},"user_stop",2)["time"],12.)
 def test_invalid_checkpoint_cannot_promote(self):
  with self.assertRaises(ValueError):finalize({},dict(time=1),{},"resource_limit",1)
 def test_regression_rejected(self):
  with self.assertRaises(ValueError):finalize(dict(time=2,cycle=3),dict(path="x",sha256="x",bytes=1,time=1,cycle=2),{},"resource_limit",1)
 def test_actual_slurm_accounting_not_requested_wall(self):
  self.assertEqual(allocated_node_hours([dict(job_id="1",elapsed_seconds=900,nodes=16)]),4)
 def test_duplicate_accounting_rejected(self):
  r=dict(job_id="1",elapsed_seconds=900,nodes=16)
  with self.assertRaises(ValueError):allocated_node_hours([r,r])
 def test_notifications_in_every_submission(self):
  from unittest.mock import patch
  with patch.object(wf.subprocess,"run") as call:
   wf.scheduler({},"sbatch",["--parsable","script"]);cmd=call.call_args.args[0]
   self.assertIn("--mail-user=jiaxiwu@caltech.edu",cmd);self.assertIn("--mail-type=BEGIN,END,FAIL",cmd)

class IntegratedStop(unittest.TestCase):
 def test_controller_finalizes_clean_memory_stop_before_terminal_return(self):
  from unittest.mock import patch
  import test_workflow as fixtures
  with tempfile.TemporaryDirectory() as d:
   cfg=fixtures.config(Path(d));st=wf.load_or_initialize_state(cfg)
   res=fixtures.result(time=9.33125,cycle=2256,checkpoint="final.rst",stop_reasons=["resource_limit_gpu_memory"])
   with patch.object(wf,"read_result",return_value=res),patch.object(wf,"scheduler_state",return_value="COMPLETED"),patch.object(wf,"cancel_future"),patch.object(wf,"allocation_accounting",return_value=dict(actual_node_hours=72.7)):
    self.assertEqual(wf.cmd_inspect(cfg,1,"123"),42)
   saved=json.loads(wf.state_path(cfg).read_text());self.assertEqual(saved["last_cycle"],2256);self.assertEqual(saved["segments_completed"],1);self.assertEqual(saved["resources"]["actual_node_hours"],72.7);self.assertEqual(saved["status"],"resource_limit")
 def test_terminal_state_cannot_release_another_segment(self):
  import test_workflow as fixtures
  st=fixtures.state();st["status"]="resource_limit"
  action,_=wf.decision({},st,fixtures.result(),"COMPLETED")
  self.assertEqual(action,"resource_limit")

class ArchiveIntegrity(unittest.TestCase):
 def fixture(self,root):
  from checkpoint_io import digest
  f=root/'closed.txt';f.write_text('sealed science evidence\n')
  cp=dict(sha256='a'*64,bytes=123,time=.0125,cycle=2,blocks=4936,particles=5000000,removed=[0,0,0])
  return dict(files=[dict(path=f.name,bytes=f.stat().st_size,sha256=digest(f))],checkpoint_relative_path='rst/final.rst',native_checkpoint=cp),cp
 def test_relocated_inventory_and_native_progress_agree(self):
  from unittest.mock import patch
  import verify_archive as archive
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);manifest,cp=self.fixture(root)
   with patch.object(archive,'checkpoint',return_value=cp):self.assertEqual(archive.verify(root,manifest),cp)
 def test_same_size_corruption_is_rejected(self):
  import verify_archive as archive
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);manifest,_=self.fixture(root);f=root/'closed.txt';data=f.read_bytes();f.write_bytes(b'X'+data[1:])
   with self.assertRaises(ValueError):archive.verify(root,manifest)
 def test_manifest_cannot_escape_destination(self):
  import verify_archive as archive
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):archive.safe_path(Path(d),'../elsewhere')
 def test_checkpoint_progress_mismatch_is_rejected(self):
  from unittest.mock import patch
  import verify_archive as archive
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);manifest,cp=self.fixture(root);cp=dict(cp,cycle=1)
   with patch.object(archive,'checkpoint',return_value=cp):
    with self.assertRaises(ValueError):archive.verify(root,manifest)
