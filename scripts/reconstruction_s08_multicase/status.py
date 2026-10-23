"""Lightweight one-shot scheduler/state inspection; no controller or submission."""
import datetime
import json
from pathlib import Path
import shlex
import subprocess

N=Path('/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023')
A='/work1/eliasmost/jiaxiwu/st_s08_scan_20261023'

def invoke(args,timeout=60):
    p=subprocess.run(args,text=True,capture_output=True,timeout=timeout)
    return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)

def main():
    state=json.loads((N/'evidence/state.json').read_text());stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();ps=[]
    for record in state['Perseus_jobs']:
        r=invoke(['scontrol','show','job',str(record['id'])]);r['job']=record['id'];ps.append(r)
        if r['returncode']==0:
            (N/'evidence'/f"SLURM_{record['id']}_LATEST.txt").write_text(r['stdout'])
            if any(f'JobState={s}' in r['stdout'] for s in ['COMPLETED','FAILED','PREEMPTED','CANCELLED','TIMEOUT','NODE_FAIL']):
                (N/'evidence'/f"SLURM_{record['id']}_TERMINAL.txt").write_text(r['stdout'])
    remote=invoke(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','hpcfund.amd.com',
      shlex.join(['sacct','-X','-j',','.join(r['id'] for r in state['jobs']),'-n','-P','-o','JobIDRaw,State,ExitCode,ElapsedRaw,AllocNodes'])])
    query=invoke(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','hpcfund.amd.com',
      shlex.join(['squeue','-u','jiaxiwu','-h','-o','%i|%j|%T|%M|%D|%R'])])
    local={}
    for key in state['cases']:
        root=N/'perseus'/key;native='yc'+str(round(state['cases'][key]['yc']*1000)).zfill(4);work=root/'runs'/native
        p=work/'state.json'
        if p.exists():
            s=json.loads(p.read_text());local[key]=dict(complete=s['complete'],latest=s['latest'],segments=len(s['segments']))
        for name in ['INITIAL_CONSTRAINT_ADMISSION.json','PRODUCTION_ADMISSION.json','CASE_NUMERICAL_ACCEPTANCE.json','RESOURCE_STOP.json']:
            p=root/'evidence'/name
            if p.exists():local.setdefault(key,{})[name]=json.loads(p.read_text())
    result=dict(utc=stamp,Perseus_scheduler=ps,AMD_accounting=remote,AMD_queue=query,Perseus_cases=local,
      scope='Scheduler and small state receipts only. No scientific completion inferred from a terminal job. No raw hashing, reductions, plotting or submissions.')
    (N/'evidence/STATUS_LATEST.json').write_text(json.dumps(result,indent=2)+'\n')
    history=N/'evidence/status_history';history.mkdir(exist_ok=True)
    (history/(stamp.replace(':','-')+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('UTC',stamp)
    for key,s in local.items():
        if 'latest' in s:print(key,'checkpoint_M',s['latest']['time'],'N',s['latest']['particle_count'],'AH_onset',s['latest']['ah_onset'],'complete',s['complete'])
    print(query['stdout'].strip())

if __name__=='__main__':main()
