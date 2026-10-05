"""Allocated-node final ledger/checksums; never prune checkpoints or other output."""
import hashlib,json,re,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from health import evaluate
from restart_header import read_header
root=Path(sys.argv[1]);label=sys.argv[2];job=sys.argv[3];rc=int(sys.argv[4]);elapsed=int(sys.argv[5])
run=root/'runs'/label;log=(run/f'segment_{job}.log').read_text(errors='replace')
health=evaluate(run);last=health['last_healthy_time']
checkpoints=list((run/'out/rst').glob('*.rst'))
latest=max(checkpoints,key=lambda p:p.stat().st_mtime) if checkpoints else None
state=dict(job_id=job,returncode=rc,elapsed_seconds=elapsed,node_hours=elapsed/3600,
           health=health,completed=('Terminating on time limit' in log or 'Terminating on cycle limit' in log) and health['valid'],
           cycle_limit='Terminating on cycle limit' in log,
           resource_failure=bool(re.search(r'out of memory|OOM|TIMEOUT|NODE_FAIL',log,re.I)))
if latest and health['valid'] and rc==0:
    # Finalization writes health first and restart last. Verify that final state time
    # matches the final logged cycle; a checkpoint is never inferred healthy by name.
    header=read_header(latest)
    checkpoint_time=header['time']
    if checkpoint_time is not None and abs(checkpoint_time-last)<=1.e-12*max(1,abs(last)):
        digest=hashlib.sha256()
        with latest.open('rb') as f:
            for data in iter(lambda:f.read(8*1024*1024),b''):digest.update(data)
        state.update(checkpoint=str(latest),checkpoint_time=checkpoint_time,
                     checkpoint_header=header,checkpoint_sha256=digest.hexdigest())
manifest=[]
for path in sorted((run/'out').rglob('*')):
    if path.is_file():manifest.append(dict(path=str(path.relative_to(run)),size=path.stat().st_size))
(run/f'manifest_{job}.json').write_text(json.dumps(manifest,indent=2)+'\n')
(run/'segment_state.json').write_text(json.dumps(state,indent=2)+'\n')
with (root/'state/segments.jsonl').open('a') as f:f.write(json.dumps(dict(label=label,**state))+'\n')
print(json.dumps(state,indent=2))
if not health['valid']:sys.exit(2)
