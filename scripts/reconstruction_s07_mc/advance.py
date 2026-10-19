"""Unique, intent-recorded metadata submissions; production requires actual review."""
import argparse
import re
import shlex
import subprocess
from cfg import *

def remote(args):
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','hpcfund.amd.com',shlex.join(args)],text=True,stderr=subprocess.STDOUT,timeout=90)
def submit(key,role,index=None,hours=2,dependency=None):
    receipt=ROOT/'evidence'/(key+'_SUBMITTED.json');intent=ROOT/'evidence'/(key+'_SUBMIT_INTENT.json')
    command=['sbatch','--parsable','--job-name=st_s07_'+key.lower(),'--time='+f'{hours:02d}:00:00']
    if dependency:command+=['--dependency=afterok:'+','.join(dependency)]
    command+=[str(AMD/'src/athenak/scripts/reconstruction_s07_mc/amd.sbatch'),role]
    if index is not None:command.append(str(index))
    if receipt.exists():
        old=load(receipt);assert old['command']==command
        print('EXISTING',key,old['id'],flush=True);return old['id']
    assert not intent.exists(),'Ambiguous prior sbatch: inspect scheduler before recording a recovered receipt'
    atomic(intent,dict(utc=now(),command=command,source=source()))
    reply=remote(command);ids=re.findall(r'^(\d+)(?:;[^\n]*)?$',reply,re.M);assert len(ids)==1,reply
    job=dict(id=ids[0],host='amd',role=role,index=index,key=key,command=command,utc=now(),reply=reply)
    atomic(receipt,job);state=load(ROOT/'evidence/state.json');state['jobs'].append(job);state.update(phase=key.lower()+'_submitted',updated_utc=now());atomic(ROOT/'evidence/state.json',state)
    print('SUBMITTED',key,job['id'],flush=True);return job['id']
def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','production','analysis']);args=p.parse_args()
    if args.phase=='preflight':
        previous=load(ROOT/'evidence/PREPARE_SUBMITTED.json')['id']
        for index in [3,2,1,0]:previous=submit('PREFLIGHT_'+str(index),'preflight',index,1,[previous])
        submit('HEALTH','health',hours=1,dependency=[previous])
    elif args.phase=='production':
        review=load(ROOT/'evidence/PRODUCTION_REVIEW.json');assert review['accepted']and review['eight_million_restart_verified']and review['actual_health_reviewed']
        for case in CASES:
            file=ROOT/'evidence/amd/preflight'/(case['id']+'.json')
            assert load(file)['passed']and sha(file)==review['preflight_sha256'][case['id']]
        subprocess.run(['rsync','-a',str(ROOT/'evidence/PRODUCTION_REVIEW.json'),f'hpcfund.amd.com:{AMD}/evidence/PRODUCTION_REVIEW.json'],check=True)
        eight=submit('PRODUCTION_3','production',3,CASES[3]['wall_hours'])
        four=submit('PRODUCTION_2','production',2,CASES[2]['wall_hours'])
        two=submit('PRODUCTION_1','production',1,CASES[1]['wall_hours'],[four])
        one=submit('PRODUCTION_0','production',0,CASES[0]['wall_hours'],[two])
        atomic(ROOT/'evidence/PRODUCTIONS_SUBMITTED.json',dict(utc=now(),jobs=[eight,four,two,one],max_simultaneous_productions=2))
    else:
        reductions=[]
        for index in [0,1,2,3]:
            dependency=[load(ROOT/'evidence'/('PRODUCTION_'+str(index)+'_SUBMITTED.json'))['id']]
            if reductions:dependency.append(reductions[-1])
            reductions.append(submit('REDUCE_'+str(index),'reduce',index,3,dependency))
        compare=submit('COMPARE','compare',hours=1,dependency=[reductions[-1]])
        previous=compare
        for group in ['0','1','2','3','scaling']:previous=submit('MOVIE_'+group.upper(),'movies',group,1,[previous])
        submit('FINALIZE','finalize',hours=1,dependency=[previous])
if __name__=='__main__':main()
