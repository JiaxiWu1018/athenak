#!/usr/bin/env python3
"""Bounded, one-way compact-product pull. Runs on a Perseus Slurm allocation."""
import datetime
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

SESSION=Path('/data/jiaxiwu/NRPIC/GI_in_cluster/session_011_amd_ah_orbit_20261021')
REMOTE='/work1/eliasmost/jiaxiwu/gi_s011_amd_20261021'
HOST='hpcfund.amd.com'

def ssh(command):
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',HOST,command],text=True,timeout=90)

def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def main():
    assert os.environ.get('SLURM_JOB_ID'),'Slurm required for file verification'
    control=SESSION/'archive_control';control.mkdir(exist_ok=True)
    with (control/'watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (control/'COMPLETE.json').exists():return
        snapshot=SESSION/'evidence'/f'archive_watch_{os.environ["SLURM_JOB_ID"]}';snapshot.mkdir(exist_ok=False)
        status={};error=None
        try:
            code="from pathlib import Path; import json; p=Path("+repr(REMOTE+'/control')+"); print(json.dumps({q.name:json.loads(q.read_text()) for q in p.glob('*.json')}))"
            status=json.loads(ssh('python3 -c '+shlex.quote(code)))
            atomic(snapshot/'amd_control.json',status)
            atomic(control/'LATEST_AMD_CONTROL.json',status)
            (snapshot/'queue.txt').write_text(ssh('squeue -u jiaxiwu -o '+shlex.quote('%.16i %.24j %.10T %.10M %.6D %R')))
            if 'PRODUCTS_COMPLETE.json' in status:
                source=Path(status['PRODUCTS_COMPLETE.json']['path'])
                if source.parent!=Path(REMOTE)/'products':raise RuntimeError('unexpected product source')
                manifest=json.loads(ssh('cat '+shlex.quote(str(source/'manifest.json'))))
                names=[]
                for row in manifest:
                    name=row['path']
                    if Path(name).name!=name or Path(name).suffix not in ('.md','.pdf','.png','.mp4','.json','.csv','.txt'):
                        raise RuntimeError('unexpected review artifact')
                    names.append(name)
                destination=SESSION/'products'/source.name;destination.mkdir(parents=True,exist_ok=True)
                listing=snapshot/'files.txt';listing.write_text('\n'.join(names+['manifest.json'])+'\n')
                subprocess.run(['rsync','-rt','--files-from='+str(listing),HOST+':'+str(source)+'/',str(destination)+'/'],check=True,timeout=900)
                for row in manifest:
                    path=destination/row['path'];h=hashlib.sha256()
                    with path.open('rb') as f:
                        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
                    if path.stat().st_size!=row['bytes'] or h.hexdigest()!=row['sha256']:
                        raise RuntimeError('checksum mismatch: '+str(path))
                receipt=dict(remote=str(source),local=str(destination),files=len(manifest),bytes=sum(r['bytes'] for r in manifest),
                             verified=True,job_id=os.environ['SLURM_JOB_ID'],utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                atomic(destination/'PERSEUS_TRANSFER.json',receipt);atomic(control/'COMPLETE.json',receipt)
                return
        except Exception as exc:
            error=str(exc);atomic(snapshot/'ERROR.json',dict(error=error))
        if 'PAUSED.json' in status:
            atomic(control/'REVIEW_REQUIRED.json',status['PAUSED.json']);return
        number=int(os.environ.get('ARCHIVE_ITERATION','0'))
        if number>=719:
            atomic(control/'REVIEW_REQUIRED.json',dict(reason='60-day archive watch bound reached',last_error=error));return
        command=['sbatch','--parsable','--begin=now+2hours',f'--export=ALL,ARCHIVE_ITERATION={number+1}',
                 '--output='+str(SESSION/'evidence/archive_watch_%j.slurm.log'),str(SESSION/'scripts/perseus_archive_watch.sbatch')]
        atomic(control/'submission_intent.json',dict(status='prepared',command=command))
        output=subprocess.check_output(command,text=True).strip();job=output.splitlines()[-1].split(';')[0]
        if not job.isdigit():raise RuntimeError('ambiguous archive scheduler receipt')
        atomic(control/'submission_intent.json',dict(status='submitted',job_id=job,iteration=number+1,command=command))

if __name__=='__main__':main()
