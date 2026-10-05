"""Login-safe Slurm accounting and reservations, restricted to registered jobs."""
import json,subprocess
from pathlib import Path
ACTIVE={'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED'}
def accounting(root):
    root=Path(root);jobs=json.loads((root/'state/amd_jobs.json').read_text())
    ids=','.join(j['job_id'] for j in jobs)
    result=subprocess.run(['sacct','-X','-n','-P','-j',ids,'--format=JobIDRaw,State,ElapsedRaw,AllocNodes,ExitCode'],check=True,capture_output=True,text=True)
    records={}
    for line in result.stdout.splitlines():
        fields=line.split('|')
        if len(fields)<5:continue
        ident,status,elapsed,nodes,code=fields[:5]
        records[ident]=dict(state=status.split()[0].rstrip('+'),elapsed_seconds=int(elapsed),nodes=int(nodes),exit_code=code)
    actual=reserved=0.
    for job in jobs:
        record=records.get(job['job_id'])
        if record is None:raise RuntimeError('accounting unavailable for registered job '+job['job_id'])
        actual+=record['elapsed_seconds']*record['nodes']/3600
        if record['state'] in ACTIVE:
            reserved+=max(0,job['limit_hours']-record['elapsed_seconds']/3600)*max(1,record['nodes'])
    report=dict(actual_node_hours=actual,reserved_remaining_node_hours=reserved,records=records,budget=200.)
    (root/'state/budget_actual.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
def require_budget(root,new_hours,postprocessing_reserve=0.):
    report=accounting(root)
    exposure=report['actual_node_hours']+report['reserved_remaining_node_hours']+new_hours+postprocessing_reserve
    if exposure>200:raise RuntimeError(f'node-hour reservation exceeds 200: {exposure:.3f}')
    return report
def register(root,job,kind,label,hours):
    path=Path(root)/'state/amd_jobs.json';jobs=json.loads(path.read_text())
    if any(j['job_id']==str(job) for j in jobs):raise RuntimeError('duplicate job registration')
    jobs.append(dict(job_id=str(job),kind=kind,label=label,limit_hours=hours))
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(jobs,indent=2)+'\n');temporary.replace(path)
