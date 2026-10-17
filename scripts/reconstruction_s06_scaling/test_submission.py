"""Tiny metadata-only fixtures: no scheduler calls, scientific jobs or raw data."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import submit_production as S

class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.patches=[patch.object(S,'ROOT',self.root),patch.object(S,'require_release',return_value={'ready':True}),
                      patch.object(S,'source',return_value='fixture_submission_source')]
        for p in self.patches:p.start()
        self.calls=[];self.remote_patch=patch.object(S,'remote',side_effect=self.remote);self.remote_patch.start()
        self.write('evidence/STAGED_AMD_SOURCE_PIN.json',{'head':'fixture_amd_source'})
        self.write('evidence/STAGE_OK.json',{'source':'fixture_amd_source'})
        self.write('evidence/state.json',{'jobs':[]})
        review=dict(all_accepted=True,six_million_restart_accepted=True,memory_and_timing_reviewed=True,
                    physics_and_mesh_unchanged=True,preflight_sha256={},wall_hours={})
        for c in S.CASES:
            hours=16 if c['n']==6000000 else 8
            path=self.write('evidence/amd/evidence/preflight/'+c['id']+'.json',dict(case=c,passed=True,
                all_states_full_payload_IDs_masses_verified=True,restart_probe_passed=c['n']==6000000,
                recommended_wall_hours=hours))
            review['preflight_sha256'][c['id']]=S.sha(path);review['wall_hours'][c['id']]=hours
        self.write('evidence/PREFLIGHT_REVIEW.json',review)

    def tearDown(self):
        self.remote_patch.stop()
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return path

    def remote(self,args):
        self.calls.append(args)
        if args[0]=='sbatch':return str(900001+sum(c[0]=='sbatch'for c in self.calls)-1)+'\n'
        if args[0]=='sacct':return args[args.index('-j')+1]+'|COMPLETED\n'
        raise AssertionError(args)

    def change_review(self,key,value):
        path=self.root/'evidence/PREFLIGHT_REVIEW.json';review=json.loads(path.read_text());review[key]=value
        path.write_text(json.dumps(review))

    def test_three_unique_jobs_dependency_limits_to_two(self):
        S.main();jobs=[c for c in self.calls if c[0]=='sbatch'];self.assertEqual(len(jobs),3)
        self.assertFalse(any(a.startswith('--dependency=')for c in jobs[:2]for a in c))
        self.assertIn('--dependency=afterok:900001',jobs[2])
        self.assertEqual([c[-1]for c in jobs],['0','1','2'])
        self.assertTrue((self.root/'evidence/PRODUCTIONS_SUBMITTED.json').exists())

    def test_repeat_verifies_existing_ids_without_resubmission(self):
        S.main();self.calls=[];S.main()
        self.assertEqual([c[0]for c in self.calls],['sacct']*3)
        self.assertEqual(len(S.load(self.root/'evidence/state.json')['jobs']),3)

    def test_ambiguous_intent_blocks_duplicate(self):
        self.write('evidence/PRODUCTION_'+S.CASES[0]['id']+'_SUBMIT_INTENT.json',{'ambiguous':True})
        with self.assertRaisesRegex(AssertionError,'Unresolved intent'):S.main()
        self.assertEqual(self.calls,[])

    def test_unreviewed_memory_blocks_launch(self):
        self.change_review('memory_and_timing_reviewed',False)
        with self.assertRaises(AssertionError):S.main()
        self.assertEqual(self.calls,[])

    def test_missing_high_N_restart_review_blocks_launch(self):
        self.change_review('six_million_restart_accepted',False)
        with self.assertRaises(AssertionError):S.main()
        self.assertEqual(self.calls,[])

    def test_changed_receipt_blocks_launch(self):
        receipt=self.root/('evidence/amd/evidence/preflight/'+S.CASES[0]['id']+'.json')
        receipt.write_text(receipt.read_text()+' ')
        with self.assertRaises(AssertionError):S.main()
        self.assertEqual(self.calls,[])

if __name__=='__main__':unittest.main()
