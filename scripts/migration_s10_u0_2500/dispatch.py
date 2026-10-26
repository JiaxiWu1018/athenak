"""Completion-dependent self-chaining under one case lock and shared hard budgets."""
import fcntl,shlex,sys
from common import *
def submit(role,after=''):
    assert role in ['preflight','production','analysis'];registry=EXT/'evidence/job_registry.json';(EXT/'evidence').mkdir(parents=True,exist_ok=True)
    with (EXT/'evidence/submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);rows=load(registry) if registry.exists() else []
        if after:
            prior=next(x for x in rows if x['id']==after);end=load(EXT/'evidence/allocations'/after/'ALLOCATION_END.json');assert end['exit_status']==0 and (EXT/'evidence/allocations'/after/'WORKER_OK.json').exists()
            if any(x.get('after')==after for x in rows):return next(x for x in rows if x.get('after')==after)
        assert not (ROOT/'evidence/RESOURCE_STOP.json').exists();seconds=dict(preflight=1200,production=5400,analysis=1800)[role]
        assert PRIOR_REQUESTED_NODE_SECONDS+sum(x['requested_seconds'] for x in rows)+seconds<=50*3600,'Shared50nodeh request reservation exhausted'
        index=len(rows)+1;comment=f'st10u0:2500:{role}:{index:03d}';command=['sbatch','--parsable','--comment='+comment,'--export=ALL,S10_ROLE='+role+',ST8X_WALL_SECONDS='+str(seconds)]
        if after:command+=['--dependency=afterok:'+after]
        command+=[str(EXT/'scripts'/(role+'.sbatch'))]
        ticket=dict(role=role,case=CASE['key'],target=2500.,nodes=1,requested_seconds=seconds,comment=comment,command=command,after=after,utc=now(),source_commit=load(EXT/'evidence/SOURCE_PIN.json')['commit'])
        p=EXT/'evidence'/f'submission_{index:03d}.json';assert not p.exists(),'Ambiguous prior submission: reconcile exact comment before retry';atomic(p,ticket)
        q=subprocess.run(command,capture_output=True,text=True);(EXT/'logs'/f'submission_{index:03d}.log').write_text(q.stdout+q.stderr);assert q.returncode==0,q.stderr;ident=q.stdout.strip().splitlines()[-1].split(';')[0];assert ident.isdigit();ticket['id']=ident;rows.append(ticket);atomic(p,ticket);atomic(registry,rows);return ticket

def completed(ident):
    row=next(x for x in load(EXT/'evidence/job_registry.json') if x['id']==ident)
    if row['role']=='analysis':
        ready=load(EXT/'evidence/FINAL_PRODUCTS_READY.json');atomic(EXT/'evidence/PIPELINE_DONE.json',dict(utc=now(),job=ident,products=ready,actual_visual_review_pending=True));return
    if row['role']=='preflight':role='production'
    else:
        state=load(ROOT/'runs/yc0500/state.json')
        if not state['complete']:
            assert state['segments'] and state['segments'][-1]['simulation_advanced']>0
            reason=state['segments'][-1].get('stop_reason');assert reason in [None,'qualified_AH_checkpoint','ah_tail_complete'],reason
        role='analysis' if state['complete'] else 'production'
    ticket=submit(role,ident);atomic(EXT/'evidence/NEXT_JOB.json',ticket);print('REGISTERED_NEXT',ticket['id'],role)
if __name__=='__main__':
    try:
        if len(sys.argv)==2 and sys.argv[1]=='start':print(submit('preflight'))
        else:completed(sys.argv[1])
    except BaseException as error:
        atomic(EXT/'evidence/DISPATCH_FAILURE.json',dict(utc=now(),job=os.environ.get('SLURM_JOB_ID'),error=repr(error)));raise
