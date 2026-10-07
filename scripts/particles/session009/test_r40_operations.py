import tempfile,unittest
from unittest.mock import patch
import wake_check
from pathlib import Path
from input_contract import validate
class R40Operations(unittest.TestCase):
 def test_previous_missing_checkpoint_override_rejected(self):
  deck=Path(__file__).with_name('gi_cluster_s9.athinput').read_text()
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'old.athinput';p.write_text(deck.replace('last_time = 0.0',''))
   with self.assertRaisesRegex(ValueError,'output7/last_time'):validate(p)
 def test_final_all_override_keys_and_radius(self):
  p=Path(__file__).with_name('gi_cluster_s9.athinput');validate(p)
  text=p.read_text();self.assertIn('extraction_radius_0 = 40.0',text);self.assertIn('radius_4_rad = 46.0',text);self.assertIn('gi_clump2_bulk_vy = 0.133215',text)
 def test_radius_outputs_match_input(self):
  root=Path(__file__).parent
  for name in ('science9.py','workflow.py'):
   text=(root/name).read_text();self.assertIn('_0040.txt',text);self.assertNotIn('_0050.txt',text)
 def test_all_full_allocations_exclude_observed_bad_node(self):
  for name in ('amd_gate.sbatch','amd_segment.sbatch'):
   self.assertIn('#SBATCH --exclude=k003-[009-010]',Path(__file__).with_name(name).read_text())
   from mpi_nodes import NODE_SPEC
   self.assertIn('#SBATCH --nodelist='+NODE_SPEC,Path(__file__).with_name(name).read_text())
 def mpi_fixture(self):
  hosts=['k002-005']+['k003-'+str(i).zfill(3) for i in range(3,8)]+['k005-'+str(i).zfill(3) for i in [2,3,4,5,6,9]]
  return hosts,'\n'.join('rank='+str(i)+'/48 host='+hosts[i//4]+'.hpcfund alltoall_errors=0' for i in range(48))
 def test_witness_requires_every_rank_and_zero_errors(self):
  from mpi_nodes import witness_hosts
  hosts,text=self.mpi_fixture();self.assertEqual(set(witness_hosts(text)),set(hosts))
  with self.assertRaises(ValueError):witness_hosts(text.replace('rank=47/48','rank=46/48'))
  with self.assertRaises(ValueError):witness_hosts(text.replace('alltoall_errors=0','alltoall_errors=1',1))
 def test_actual_group_with_unwitnessed_node_is_rejected(self):
  from mpi_nodes import verify_hosts
  hosts,text=self.mpi_fixture();hosts[0]='k003-009'
  with self.assertRaises(ValueError):verify_hosts(hosts,text)
 def test_exact_witnessed_group_accepted_in_any_order(self):
  from mpi_nodes import verify_hosts
  hosts,text=self.mpi_fixture();self.assertEqual(set(verify_hosts(list(reversed(hosts)),text)),set(hosts))
 def test_wake_queues_to_existing_writer_without_starting_another(self):
  import subprocess
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'evidence').mkdir()
   with patch('wake_check.LOCAL',root),patch('wake_check.subprocess.run',return_value=subprocess.CompletedProcess([],0)) as run:
    result=wake_check.deliver()
   args=run.call_args.args[0]
   self.assertEqual(args,[wake_check.CODEX,'queue','--thread',wake_check.THREAD,'--message',wake_check.PROMPT]);self.assertEqual(result['status'],'wake_queued')
 def test_active_thread_is_not_interrupted_by_monitor(self):
  import json,time
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'control').mkdir();(root/'control/wake_config.json').write_text(json.dumps(dict(deadline_utc=time.time()+86400)))
   with patch('wake_check.LOCAL',root),patch('wake_check.metadata',return_value=dict(user_stop=False)),patch('wake_check.activity',return_value='active'),patch('wake_check.write'),patch('wake_check.deliver') as start:
    result=wake_check.check()
  self.assertEqual(result['status'],'checked_agent_already_active');start.assert_not_called()
 def test_idle_thread_wake_uses_existing_session(self):
  import json,time
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'control').mkdir();(root/'control/wake_config.json').write_text(json.dumps(dict(deadline_utc=time.time()+86400)))
   with patch('wake_check.LOCAL',root),patch('wake_check.metadata',return_value=dict(user_stop=False)),patch('wake_check.activity',return_value='idle'),patch('wake_check.write'),patch('wake_check.deliver',return_value=dict(status='wake_completed')) as start:
    result=wake_check.check()
  self.assertEqual(result['status'],'wake_completed');start.assert_called_once()
if __name__=='__main__':unittest.main()
