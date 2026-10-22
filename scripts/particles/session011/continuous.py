"""Audit and rotate completed periodic checkpoints alongside uninterrupted evolution.

Uses the production allocation's spare CPU. A checkpoint must have a stable size
and mtime AND a later published simulation cycle before it is eligible. Audit
failures are retained for review; this process never requests an evolution stop.
"""
import argparse
import fcntl
import json
import time
from pathlib import Path
from runtime import (atomic, audit_population, checkpoint, last_csv,
                     require_compute, retain_checkpoints)


def eligible(path, previous_stat, current_cycle):
    st=path.stat()
    stamp=(st.st_size,st.st_mtime_ns)
    if stamp!=previous_stat:return False,stamp
    row=checkpoint(path,hashing=False)
    return current_cycle>row['cycle'],stamp


def audit_saved(root,run,path):
    state=checkpoint(path)
    # Saved restart times need not coincide with the compact profile cadence.
    # Native fields, exact IDs, all rest weights and death partition are mandatory.
    audit=audit_population(root,run,[state],state,publish=False,require_online=False)
    receipt=run/'checkpoint_audits'/f'{path.stem}.json'
    receipt.parent.mkdir(exist_ok=True)
    atomic(receipt,dict(status='pass',checkpoint=state,population=audit,
                       evolution_interrupted=False,epoch=time.time()))
    # Keep the pre-existing boundary death ledger intact until normal finalization.
    retain_checkpoints(root,run,[state])
    atomic(root/'control/live_checkpoint.json',dict(state,audit_receipt=str(receipt),
           boundary_ledger_unchanged=True))
    return state


def watch(root,run):
    require_compute()
    seen={};done=set()
    while not (run/'AUDIT_FINISH').exists():
        status=last_csv(run/'s11_amd.gi_status.csv')
        cycle=int(status['cycle']) if status else -1
        for path in sorted((run/'rst').glob('*.rst')):
            if str(path) in done:continue
            try:
                ready,seen[str(path)]=eligible(path,seen.get(str(path)),cycle)
                if not ready:continue
                with (root/'control/checkpoint.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    row=audit_saved(root,run,path)
                done.add(str(path))
                print(json.dumps(dict(status='audited',time=row['time'],cycle=row['cycle'])),flush=True)
            except Exception as exc:
                error=dict(status='review_required',checkpoint=str(path),
                           reason=f'{type(exc).__name__}: {exc}',epoch=time.time(),
                           evolution_interrupted=False)
                atomic(run/'CHECKPOINT_AUDIT_REVIEW.json',error)
                print(json.dumps(error),flush=True)
        for _ in range(12):
            if (run/'AUDIT_FINISH').exists():return
            time.sleep(5)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    watch(a.root,a.run)
