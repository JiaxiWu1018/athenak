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
