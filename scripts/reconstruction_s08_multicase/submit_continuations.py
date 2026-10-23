"""Register short, serial Slurm continuations; no daemon or new trajectories."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

N=Path('/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023')

def elapsed(value):
    days,clock=value.split('-',1) if '-' in value else ('0',value)
    parts=list(map(int,clock.split(':')))
    assert len(parts)==3
    return int(days)*86400+3600*parts[0]+60*parts[1]+parts[2]

def budget(state,live):
    total=0.;detail=[]
    for r in state['Perseus_jobs']:
        jid=str(r['id']);hours=r['hours']*r['nodes'];basis='declared upper bound'
        if jid not in live:
            if jid=='19799' and (N/'evidence/19799_INTERRUPTION_AND_RUNTIME_BOUND.json').exists():
                hours=json.loads((N/'evidence/19799_INTERRUPTION_AND_RUNTIME_BOUND.json').read_text())['node_hour_upper_bound'];basis='observed bounded allocation'
            else:
                for p in (N/'evidence').glob(f'SLURM_{jid}_*.txt'):
                    text=p.read_text()
                    if re.search(r'JobState=(COMPLETED|FAILED|CANCELLED|PREEMPTED|TIMEOUT|NODE_FAIL)',text):
                        runtime=re.search(r'RunTime=(\S+)',text);nodes=re.search(r'NumNodes=(\d+)',text)
                        if runtime and nodes:hours=elapsed(runtime[1])*int(nodes[1])/3600;basis='terminal scheduler';break
        total+=hours;detail.append(dict(job=jid,node_hours_upper=hours,basis=basis))
    if state['jobs']:
        ids=','.join(str(r['id']) for r in state['jobs'])
        args=['sacct','-X','-j',ids,'-n','-P','-o','JobIDRaw,State,ElapsedRaw,AllocNodes']
        raw=subprocess.check_output(['ssh','-o','BatchMode=yes','hpcfund.amd.com',shlex.join(args)],text=True,timeout=90)
        found={v[0]:v for v in [line.split('|') for line in raw.splitlines() if line.strip()]}
        for r in state['jobs']:
            row=found.get(str(r['id']));hours=r['hours']*r['nodes'];basis='declared upper bound'
            if row and not row[1].startswith(('RUNNING','PENDING','COMPLETING')):
                hours=int(row[2] or 0)*int(row[3] or 0)/3600;basis='AMD terminal scheduler'
            total+=hours;detail.append(dict(job=r['id'],node_hours_upper=hours,basis=basis))
    return total,detail

def main():
    lane,parent,count=sys.argv[1],sys.argv[2],int(sys.argv[3]);assert lane in ['long','weak','u0005'] and 1<=count<=6
    mode={'long':'yc0500_long700','weak':'weak_pair','u0005':'yc0500_u0005'}[lane]
    statefile=N/'evidence/state.json';state=json.loads(statefile.read_text())
    registered={str(r['id']):r for r in state['Perseus_jobs']};assert parent in registered and registered[parent]['mode']==mode
    assert not any(str(r.get('afterany',r.get('predecessor','')))==parent for r in state['Perseus_jobs']), 'Parent already has a registered successor; append at the current tail'
    q=subprocess.check_output(['squeue','-u','jiaxiwu','-h','-o','%i|%j|%T'],text=True)
    live={line.split('|')[0]:line.split('|')[2] for line in q.splitlines() if line.strip()}
    assert parent in live or any(p.exists() for p in [N/'evidence'/f'SLURM_{parent}_TERMINAL.txt',N/'evidence'/f'SLURM_{parent}_COMPLETED.txt'])
    used,detail=budget(state,live);assert used+count/3<state['node_hour_review']-1,(used,count,state['node_hour_review'])
    repo=N/'src/athenak';script=repo/'scripts/reconstruction_s08_multicase'/dict(long='run_long.sbatch',weak='run_weak.sbatch',u0005='run_u0005.sbatch')[lane]
    changes=subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).splitlines()
    diagnostics=[]
    for change in changes:
        assert change.startswith('?? ') and re.fullmatch(r'(yc\d{4}|frozen_ah)\.horizon_summary_\d+\.txt',change[3:]),change
        p=repo/change[3:];diagnostics.append(dict(path=str(p),bytes=p.stat().st_size,scope='Legacy native pre-chdir diagnostic, retained; excluded from source staging'))
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    created=[]
    for _ in range(count):
        reply=subprocess.check_output(['sbatch','--parsable','--dependency=afterany:'+parent,str(script)],text=True);jid=reply.strip().split(';')[0];assert jid.isdigit()
        r=dict(id=jid,host='perseus',mode=mode,nodes=1,gpus=2,cpus=2,hours=1/3,afterany=parent,
          script=str(script),script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),source_head=head,
          utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reply=reply,
          retained_untracked_native_diagnostics=diagnostics,
          concurrency_proof='One serial dependency chain per PS lane; at most two PS allocations/four GPUs, plus at most two registered AMD trajectories. No new scientific realization.',
          node_hour_upper_before=used,budget_evidence=detail)
        state['Perseus_jobs'].append(r);state.setdefault('continuation_chains',{}).setdefault(lane,[]).append(jid)
        state['utc']=r['utc'];state['source_head']=head
        (N/'evidence'/f'SUBMITTED_PERSEUS_{jid}.json').write_text(json.dumps(r,indent=2)+'\n')
        statefile.write_text(json.dumps(state,indent=2)+'\n');created.append(jid);parent=jid;used+=1/3
    print(json.dumps(dict(lane=lane,created=created,node_hour_upper_with_new_chain=used),indent=2))

if __name__=='__main__':main()
