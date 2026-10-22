"""Focused safety checks for continuous evolution with saved-file audits."""
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import continuous
import runtime


class ContinuousSafety(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.run=self.root/'runs/segment_0000';self.run.mkdir(parents=True)
        (self.root/'control').mkdir()
    def tearDown(self):self.tmp.cleanup()
    def test_in_progress_checkpoint_is_not_audited(self):
        p=self.run/'test.rst';p.write_bytes(b'partial')
        with patch('continuous.checkpoint') as check:
            ready,stamp=continuous.eligible(p,None,100)
            self.assertFalse(ready);check.assert_not_called()
        with patch('continuous.checkpoint',return_value={'cycle':100}):
            self.assertFalse(continuous.eligible(p,stamp,100)[0])
            self.assertTrue(continuous.eligible(p,stamp,101)[0])
        p.write_bytes(b'changed checkpoint size')
        with patch('continuous.checkpoint') as check:
            self.assertFalse(continuous.eligible(p,stamp,101)[0]);check.assert_not_called()
    def test_live_audit_does_not_publish_boundary_ledger_or_stop(self):
        p=self.run/'s11_amd.00007.rst';p.write_bytes(b'saved')
        state=dict(path=str(p),cycle=300,time=12.)
        with patch('continuous.checkpoint',return_value=state),patch('continuous.audit_population',return_value={'exact_partition':True}) as audit,patch('continuous.retain_checkpoints'):
            continuous.audit_saved(self.root,self.run,p)
            self.assertEqual(audit.call_args.kwargs,dict(publish=False,require_online=False))
        self.assertFalse((self.run/'REQUEST_STOP').exists())
        self.assertFalse((self.root/'control/dead_tags.npy').exists())
        self.assertFalse((self.root/'control/latest_checkpoint.json').exists())
        self.assertTrue(json.loads((self.root/'control/live_checkpoint.json').read_text())['boundary_ledger_unchanged'])
    def test_retention_preserves_protected_states_and_two_recoveries(self):
        rows=[]
        for n in range(4):
            p=self.run/f's11_amd.{n:05d}.rst';p.write_bytes(str(n).encode())
            rows.append(dict(path=str(p),cycle=n,time=float(n),sha256=runtime.digest(p),protected=n==0))
        runtime.atomic(self.root/'control/checkpoint_inventory.json',rows[:2])
        runtime.retain_checkpoints(self.root,self.run,rows[2:])
        self.assertTrue(Path(rows[0]['path']).exists())
        self.assertFalse(Path(rows[1]['path']).exists())
        self.assertTrue(all(Path(r['path']).exists() for r in rows[2:]))
        # Repeated observations must not duplicate inventory entries or lose protection.
        runtime.retain_checkpoints(self.root,self.run,rows[2:])
        history=json.loads((self.root/'control/checkpoint_inventory.json').read_text())
        self.assertEqual(len(history),4);self.assertTrue(history[0]['protected'])
    def test_live_partition_excludes_future_deaths_and_keeps_base_ledger(self):
        import numpy as np
        control=self.root/'control'
        np.save(control/'initial_mass.npy',np.ones(5_000_000))
        dead=np.ones(5_000_000,dtype=bool);dead[[0,3_000_000,4_000_000]]=False
        np.save(control/'dead_tags.npy',dead)
        before=runtime.digest(control/'dead_tags.npy')
        (self.run/'s11_amd.prtcl_destroy.csv').write_text(
            'cycle,time,tag,mass\n100,10.1,0,1\n102,10.3,3000000,1\n')
        (self.run/'s11_amd.gi_particles.csv').write_text('cycle,component,shell,mask,count,rest_mass,Jz\n')
        data=np.zeros((10,2));data[0]=1;data[9]=[3_000_000,4_000_000]
        data[2]=[2,3];data[5]=[4,5]
        path=self.run/'state.rst';data.tofile(path)
        state=dict(path=str(path),particle_offset=0,particles=2,removed=[0,0,4_999_998],cycle=101,time=10.2)
        result=runtime.audit_population(self.root,self.run,[state],state,publish=False,require_online=False)
        self.assertTrue(result['exact_partition']);self.assertEqual(result['new_deaths'],1)
        self.assertFalse(result['online_profile_compared'])
        self.assertEqual([r['native_Jz'] for r in result['components']],[0.,8.,15.])
        self.assertEqual(before,runtime.digest(control/'dead_tags.npy'))
        self.assertFalse((self.run/'population_audit.json').exists())

    def test_changed_retained_copy_prevents_rotation(self):
        rows=[]
        for n in range(3):
            p=self.run/f's11_amd.{n:05d}.rst';p.write_bytes(str(n).encode())
            rows.append(dict(path=str(p),cycle=n+1,time=float(n+1),sha256=runtime.digest(p)))
        runtime.atomic(self.root/'control/checkpoint_inventory.json',rows)
        Path(rows[-1]['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'retained recovery checksum'):
            runtime.retain_checkpoints(self.root,self.run,[])
        self.assertTrue(Path(rows[0]['path']).exists())


if __name__=='__main__':unittest.main()
