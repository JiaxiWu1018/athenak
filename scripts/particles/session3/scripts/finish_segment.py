"""Allocated-node final ledger/checksums; never prune checkpoints or other output."""
import hashlib,json,re,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from health import evaluate,read_rows
from restart_header import read_header
root=Path(sys.argv[1]);label=sys.argv[2];job=sys.argv[3];rc=int(sys.argv[4]);elapsed=int(sys.argv[5])
slurm_state=sys.argv[6] if len(sys.argv)>6 else 'COMPLETED' if rc==0 else 'FAILED'
run=root/'runs'/label;log=(run/f'segment_{job}.log').read_text(errors='replace')
health=evaluate(run);last=health['last_healthy_time']
checkpoints=list((run/'out/rst').glob('*.rst'))
headers=[]
for checkpoint in checkpoints:
    try:headers.append((read_header(checkpoint),checkpoint))
    except (ValueError,OSError):continue  # An interrupted write is never promoted.
state=dict(job_id=job,returncode=rc,elapsed_seconds=elapsed,node_hours=elapsed/3600,
           health=health,completed=('Terminating on time limit' in log or 'Terminating on cycle limit' in log) and health['valid'],
           cycle_limit='Terminating on cycle limit' in log,
           slurm_state=slurm_state,
           resource_failure=slurm_state in ('OUT_OF_MEMORY','TIMEOUT','NODE_FAIL','PREEMPTED'))
if last is not None:
    healthy_times={int(row['cycle']):float(row['time']) for row in read_rows(health['ledger'])
                   if float(row['time'])<=last and int(row['healthy'])==1}
    # Validate finite physical-stop checkpoints too. Hard-stop records never
    # authorize retry, but an earlier healthy checkpoint is retained for diagnosis.
    for header,latest in sorted(headers,key=lambda item:(item[0]['time'],item[0]['cycle']),reverse=True):
        checkpoint_time=header['time'];recorded=healthy_times.get(header['cycle'])
        if recorded is None or abs(checkpoint_time-recorded)>1.e-12*max(1,abs(recorded)):continue
        digest=hashlib.sha256()
        with latest.open('rb') as f:
            for data in iter(lambda:f.read(8*1024*1024),b''):digest.update(data)
        state.update(checkpoint=str(latest),checkpoint_time=checkpoint_time,
                     checkpoint_header=header,checkpoint_sha256=digest.hexdigest())
        break
manifest=[]
for path in sorted(run.rglob('*')):
    if path.is_file():manifest.append(dict(path=str(path.relative_to(run)),size=path.stat().st_size))
(run/f'manifest_{job}.json').write_text(json.dumps(manifest,indent=2)+'\n')
(run/'segment_state.json').write_text(json.dumps(state,indent=2)+'\n')
with (root/'state/segments.jsonl').open('a') as f:f.write(json.dumps(dict(label=label,**state))+'\n')
print(json.dumps(state,indent=2))
if not health['valid']:sys.exit(2)
