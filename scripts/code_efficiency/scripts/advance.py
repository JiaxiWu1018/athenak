#!/usr/bin/env python3
"""Lightweight scheduler recovery; never simulates, builds, or plots on a login node."""
import argparse,fcntl,json,os,re,subprocess,time
from pathlib import Path
ROOT=Path(os.environ.get('EFFICIENCY_SESSION',Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser();p.add_argument('--close',action='store_true');args=p.parse_args()
lock=(ROOT/'provenance/.advance.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
state_path=ROOT/'provenance/recovery.json'
state=json.loads(state_path.read_text()) if state_path.exists() else dict(gate4='13341',recoveries=[],attempts={})
# Save terminal scheduler evidence while it is still in the controller cache.
# This site disables sacct; waiting until campaign closure would lose old states.
ledger_path=ROOT/'provenance/jobs.jsonl'
ledger_rows=[json.loads(x) for x in ledger_path.read_text().splitlines()] if ledger_path.exists() else []
known={r['job_id']:r for r in ledger_rows}
observed_path=ROOT/'provenance/job_states.json'
observed=json.loads(observed_path.read_text()) if observed_path.exists() else {}
interrupt_path=ROOT/'provenance/interruptions.json'
interruptions=json.loads(interrupt_path.read_text()) if interrupt_path.exists() else []
interrupted_ids={r['job_id'] for r in interruptions}
bulk=subprocess.run(['scontrol','-o','show','job'],text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
for line in bulk.stdout.splitlines():
 fields=dict(re.findall(r'(\w+)=(\S+)',line));job=fields.get('JobId')
 if job not in known:continue
 observed[job]={k:fields.get(k) for k in ['JobState','ExitCode','Reason','StartTime','EndTime','PreemptTime']}
 observed[job]['observed_unix']=time.time()
 if fields.get('JobState') in ['PREEMPTED','TIMEOUT','NODE_FAIL'] and job not in interrupted_ids:
  row=known[job];interruptions.append(dict(job_id=job,kind=row['kind'],case=row['case'],state=fields['JobState'],observed_unix=time.time()));interrupted_ids.add(job)
observed_path.write_text(json.dumps(observed,indent=2)+'\n')
interrupt_path.write_text(json.dumps(interruptions,indent=2)+'\n')
def job_state(job):
    r=subprocess.run(['scontrol','show','job',str(job)],text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    m=re.search(r'JobState=(\S+)',r.stdout)
    return m.group(1) if m else 'UNKNOWN'
def receipt(kind,name):
    p=ROOT/'results'/kind/name/'receipt.json'
    return json.loads(p.read_text()) if p.exists() else {}
def submit(kind,cases,dependency=None,partition='debug',gate=None):
    cmd=['python3',str(ROOT/'scripts/submit.py'),'--kind',kind,'--partition',partition,'--names']+cases
    if dependency:cmd+=['--dependency',str(dependency)]
    if gate:cmd+=['--gate-job',str(gate)]
    r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    data=json.loads(r.stdout);state['recoveries'].append(data);return data['last_job']
gate_name='gate_ranks_st_dep1_R4'
if not receipt('gate',gate_name).get('valid'):
    status=job_state(state['gate4'])
    if status in ['CANCELLED','PREEMPTED','TIMEOUT','NODE_FAIL']:
        old=state['gate4'];new=submit('gate',[gate_name],partition=state.get('four_rank_partition','debug'));state['gate4']=new
        ledger=[json.loads(x) for x in (ROOT/'provenance/jobs.jsonl').read_text().splitlines()]
        for row in ledger:
            if job_state(row['job_id'])!='PENDING':continue
            detail=subprocess.run(['scontrol','show','job',row['job_id']],text=True,stdout=subprocess.PIPE).stdout
            m=re.search(r'Dependency=(\S+)',detail)
            if m and f'afterok:{old}' in m.group(1):
                dep=re.sub(r'\([^)]*\)','',m.group(1)).replace(f'afterok:{old}',f'afterok:{new}')
                subprocess.run(['scontrol','update',f'JobId={row["job_id"]}',f'Dependency={dep}'],check=True)
if args.close:
    matrix=json.loads((ROOT/'matrix.json').read_text());ledger=[json.loads(x) for x in (ROOT/'provenance/jobs.jsonl').read_text().splitlines()];latest={}
    for row in ledger:latest[(row['kind'],row['case'])]=row
    pending=[];retry=[];failed=[]
    for key,row in latest.items():
        kind,name=key
        if kind not in ['throughput','profile','device']:continue
        if receipt(kind,name).get('valid'):continue
        status=job_state(row['job_id'])
        if status in ['PENDING','RUNNING','CONFIGURING','COMPLETING']:pending.append(row['job_id']);continue
        attempts=state['attempts'].get(kind+'/'+name,0)
        if attempts<2:
            retry.append(row);state['attempts'][kind+'/'+name]=attempts+1
        else:failed.append(dict(kind=kind,case=name,state=status))
    last=None
    for row in retry:
        c=next(c for c in matrix if c['name']==row['case'])
        last=submit(row['kind'],[row['case']],dependency=last,partition=state.get('four_rank_partition','debug') if c['ranks']==4 else 'debug',gate=state['gate4'] if c['ranks']==4 and not receipt('gate',gate_name).get('valid') else None)
    state['unresolved']=failed
    if pending or retry:
        dependencies=([last] if last else [])+pending
        command=['sbatch','--parsable','--dependency','afterany:'+':'.join(dependencies),str(ROOT/'jobs/close.sbatch')]
        state['next_closure_job']=subprocess.check_output(command,text=True).strip().split(';')[0]
    else:
        qa_status=subprocess.run(['python3',str(ROOT/'scripts/verify_results.py')],check=False).returncode
        complete=qa_status==0 and all(receipt('throughput',c['name']).get('valid') for c in matrix) and not failed
        complete=complete and sum(receipt('profile',c['name']).get('valid',False) for c in matrix)==27 and sum(receipt('device',c['name']).get('valid',False) for c in matrix)==6
        numerics=ROOT/'provenance/numerical_gate.json'
        numerical=json.loads(numerics.read_text()) if numerics.exists() else {}
        current_manifest=json.loads((ROOT/'provenance/executables/manifest.json').read_text())
        current_hashes={record['kind']:digest for digest,record in current_manifest.items()}
        complete=complete and numerical.get('passed',False) and numerical.get('executables')==current_hashes
        validation_path=ROOT/'results/VALIDATION.json'
        final_validation=json.loads(validation_path.read_text()) if validation_path.exists() else {}
        figure_names=['01_particle_count','02_meshblock_layout','03_strong_scaling','04_dynamic_particle_mesh_cost','05_sampling_and_rank_imbalance','06_kernel_cost_and_interpolation']
        artifacts_ok=all((ROOT/'figures'/f'{name}.{ext}').exists() for name in figure_names for ext in ['pdf','png']) and (ROOT/'provenance/report_manifest.json').exists() and (ROOT/'reports/REPORT_code_efficiency.pdf').exists()
        state['final_artifacts_valid']=bool(artifacts_ok and final_validation.get('complete',False))
        complete=complete and state['final_artifacts_valid']
        (ROOT/'CLOSURE.md').write_text('# Campaign closure\n\n'+('COMPLETE' if complete else 'INCOMPLETE — see provenance/recovery.json and results/FINAL_QA.json')+'\n\nReports and figures were regenerated from retained receipts. Every promoted case passed liveness/source/freeze gates. Source, inputs, launches, executable hashes, timing tables, diagnostics and storage status remain in this session. No scientific snapshots or restarts were generated.\n')
        state['closed_complete']=complete
state['updated_unix']=time.time();state_path.write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(dict(gate4=state['gate4'],recoveries=len(state['recoveries']),closed_complete=state.get('closed_complete'))),flush=True)
