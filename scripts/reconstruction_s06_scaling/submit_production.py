"""Metadata-only, review-bound production submission with resumable receipts."""
import re
import shlex
import subprocess
from cfg import *
from os_gate import require_release

def remote(args):
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
        'hpcfund.amd.com',shlex.join(args)],text=True,stderr=subprocess.STDOUT,timeout=90)

def main():
    gate=require_release(ROOT)
    review_path=ROOT/'evidence/PREFLIGHT_REVIEW.json';review=load(review_path)
    assert review['all_accepted'] is True and review['six_million_restart_accepted'] is True
    assert review['memory_and_timing_reviewed'] is True
    assert review['physics_and_mesh_unchanged'] is True
    staged=load(ROOT/'evidence/STAGED_AMD_SOURCE_PIN.json')
    assert staged['head']==load(ROOT/'evidence/STAGE_OK.json')['source']
    for case in CASES:
        receipt=ROOT/'evidence/amd/evidence/preflight'/(case['id']+'.json')
        assert sha(receipt)==review['preflight_sha256'][case['id']]
        result=load(receipt)
        assert result['passed'] and result['all_states_full_payload_IDs_masses_verified']
        assert result['case']==case
        if case['n']==6000000:assert result['restart_probe_passed']
        hours=review['wall_hours'][case['id']]
        assert isinstance(hours,int) and hours>=result['recommended_wall_hours']
    jobs=[]
    for index,case in enumerate(CASES):
        hours=review['wall_hours'][case['id']]
        command=['sbatch','--parsable','--job-name=st_s06ps_prod_'+case['id'],
                 '--time='+f'{hours:02d}:00:00','--export=ALL,ST_WALL_HOURS='+str(hours)]
        # Two initial cases may run together. The third can use the first
        # case's node only after that case completes successfully: max two.
        if index==2:command.append('--dependency=afterok:'+jobs[0]['id'])
        command.extend([str(AMD/'src/athenak/scripts/reconstruction_s06_scaling/amd.sbatch'),
                        'production',str(index)])
        stem='PRODUCTION_'+case['id'];intent=ROOT/'evidence'/(stem+'_SUBMIT_INTENT.json')
        receipt=ROOT/'evidence'/(stem+'_SUBMITTED.json')
        if receipt.exists():
            job=load(receipt);assert job['command']==command
            actual=remote(['sacct','-X','-j',job['id'],'-n','-P','--format=JobIDRaw,State'])
            assert any(line.startswith(job['id']+'|')for line in actual.splitlines())
            print('EXISTING_SUBMISSION_VERIFIED',case['id'],job['id'],flush=True)
        else:
            assert not intent.exists(),'Unresolved intent: inspect scheduler; never blindly retry sbatch'
            atomic(intent,dict(utc=now(),case=case,command=command,submission_source=source(),
                amd_runtime_source=staged['head'],review_sha256=sha(review_path),os_gate=gate))
            reply=remote(command);ids=re.findall(r'^(\d+)(?:;[^\n]*)?$',reply,re.M)
            assert len(ids)==1,reply
            job=dict(id=ids[0],host='amd',role='production',case=case['id'],index=index,
                wall_hours=hours,command=command,reply=reply,utc=now(),
                submission_source=source(),amd_runtime_source=staged['head'])
            atomic(receipt,job)
            print('PRODUCTION_SUBMITTED',case['id'],job['id'],flush=True)
        state=load(ROOT/'evidence/state.json')
        if not any(j['host']=='amd' and j['id']==job['id']for j in state['jobs']):
            state['jobs'].append(job)
        state.update(phase='production_submitted',updated_utc=now(),
            next='Monitor the three actual1000M production jobs. On completion stage/reduce each via allocated analysis, then compare/movies/actual visual review/report. Preserve failed runs and ambiguous intents.')
        atomic(ROOT/'evidence/state.json',state);jobs.append(job)
    atomic(ROOT/'evidence/PRODUCTIONS_SUBMITTED.json',dict(utc=now(),jobs=jobs,
        max_concurrent_production_nodes=2,review_sha256=sha(review_path),
        total_requested_wall_node_hours=sum(j['wall_hours']for j in jobs)))

if __name__=='__main__':main()
