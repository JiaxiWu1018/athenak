"""Read-only restart snapshot export, permitted by archive rule6.

This preparation never closes, renames, edits or deletes the Perseus source,
changes trajectory ownership, or starts evolution. Only a new AMD staging copy
and new export receipts are written. A later handoff needs fresh source/job
checks and actual destination checksum/zero-step qualification.
"""
import fcntl
import re
import shlex
import sys
from common import *
from checkpoint import validate

def main(source_job):
    job();assert MACHINE=='perseus' and CASE['key']=='yc0500_long700'
    with (EXT/'evidence/locks'/f"{CASE['key']}.lock").open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        scheduler=subprocess.run(['scontrol','show','job',source_job],text=True,capture_output=True)
        if scheduler.returncode==0:
            assert not re.search(r'JobState=(RUNNING|PENDING|COMPLETING)',scheduler.stdout)
        work=ROOT/'runs'/CASE['id'];state=load(work/'state.json')
        latest=validate(state['latest']['path']);latest.pop('parameters')
        assert latest['sha256']==state['latest']['sha256'] and latest['time']>=500
        record=ROOT/'evidence/jobs'/job();record.mkdir(parents=True,exist_ok=True)
        files=[];retained=[]
        for name in ['runs','inputs','evidence']:
            for p in sorted((ROOT/name).rglob('*')):
                if not p.is_file() or record in p.parents:continue
                if p.name=='REQUEST_STOP':continue
                if p.suffix=='.rst' and p!=Path(latest['path']):
                    retained.append(dict(path=str(p),bytes=p.stat().st_size,source_retained=True));continue
                r=receipt(p);r['relative']=str(p.relative_to(ROOT));files.append(r)
        stage=str(AMD/'staging'/f"snapshot_{CASE['key']}_{job()}")
        manifest=dict(utc=now(),job=job(),case=CASE,source_root=str(ROOT),source_job=source_job,
          source_scheduler_stdout=scheduler.stdout,source_scheduler_stderr=scheduler.stderr,
          source_case_lock_acquired=True,checkpoint=latest,files=files,
          omitted_old_restarts=retained,logical_bytes=sum(r['bytes'] for r in files),destination_stage=stage,
          source_unchanged=True,ownership_unchanged=True,raw_tail_requires_prefix_validation=True,
          destination_validation_and_restart_identity_pending=True)
        path=record/'SNAPSHOT_MANIFEST.json';atomic(path,manifest)
        selection=record/'transfer_files.txt';selection.write_text(''.join(r['relative']+'\n' for r in files))
        remote=shlex.join(['mkdir','-p',stage])
        cmd(['ssh','-o','BatchMode=yes','hpcfund.amd.com',remote])
        cmd(['rsync','-a','--files-from='+str(selection),str(ROOT)+'/', 'hpcfund.amd.com:'+stage+'/'])
        cmd(['rsync','-a',path,'hpcfund.amd.com:'+stage+'/SNAPSHOT_MANIFEST.json'])
        atomic(record/'SNAPSHOT_EXPORTED.json',dict(utc=now(),job=job(),manifest=receipt(path),
          destination_stage=stage,source_unchanged=True,source_can_still_resume=True,
          no_native_evolution_started=True,destination_validation_pending=True))

if __name__=='__main__':main(sys.argv[1])
