"""One-way hourly checksum mirror of compact review artifacts, never raw frames."""
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time

SESSION=Path('/data/jiaxiwu/NRPIC/GI_in_cluster/session_011_amd_ah_orbit_20261021')
REMOTE=Path('/work1/eliasmost/jiaxiwu/gi_s011_amd_20261021')
HOST='hpcfund.amd.com'

def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def ssh_python(code):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',HOST,'python3','-'],
        input=code,text=True,capture_output=True,check=True,timeout=90)
    return json.loads(r.stdout)

def main():
    assert os.environ.get('SLURM_JOB_ID'),'Slurm compute required for transfer checksums'
    control=SESSION/'hourly_archive_control';control.mkdir(exist_ok=True)
    with (control/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        out=SESSION/'evidence'/f'hourly_archive_{os.environ["SLURM_JOB_ID"]}';out.mkdir()
        error=None
        try:
            code="from pathlib import Path\nimport json\nr=Path("+repr(str(REMOTE))+ ")\nprint(json.dumps([json.loads(p.read_text()) for p in sorted((r/'products').glob('hourly_*/RESULT.json'))]))"
            receipts=ssh_python(code)
            atomic(out/'remote_results.json',receipts)
            for receipt in receipts:
                if receipt['status']!='pass':continue
                source=Path(receipt['path'])
                assert source.name=='review' and source.parent.parent==REMOTE/'products'
                destination=SESSION/'products'/source.parent.name
                if (destination/'PERSEUS_TRANSFER.json').exists():continue
                manifest=ssh_python('import json\nfrom pathlib import Path\nprint((Path('+repr(str(source))+')/"manifest.json").read_text())')
                rows=[r for r in manifest if Path(r['path']).suffix in ('.md','.png','.pdf','.mp4','.json','.txt')]
                for row in rows:assert Path(row['path']).name==row['path']
                listing=out/'files.txt';listing.write_text('\n'.join(r['path'] for r in rows)+'\n')
                destination.mkdir(parents=True,exist_ok=True)
                subprocess.run(['rsync','-rt','--files-from='+str(listing),HOST+':'+str(source)+'/',str(destination)+'/'],check=True,timeout=900)
                for row in rows:
                    p=destination/row['path'];h=hashlib.sha256()
                    with p.open('rb') as f:
                        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
                    assert p.stat().st_size==row['bytes'] and h.hexdigest()==row['sha256'],str(p)
                atomic(destination/'REVIEW_MANIFEST.json',rows)
                transferred=dict(status='pass',remote=str(source),local=str(destination),
                    files=len(rows),bytes=sum(r['bytes'] for r in rows),checksums_verified=True,
                    job_id=os.environ['SLURM_JOB_ID'],utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    diagnostic_time=receipt['diagnostic_time'],audited_checkpoint_time=receipt['audited_checkpoint_time'])
                atomic(destination/'PERSEUS_TRANSFER.json',transferred);atomic(control/'LATEST.json',transferred)
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}';atomic(out/'ERROR.json',dict(reason=error))
        # Keep cadence even if a transfer failed; retry retained completed results.
        due=(math.floor((time.time()-900)/3600)+1)*3600+900
        script=Path(__file__).with_name('perseus_hourly_archive.sbatch')
        command=['sbatch','--parsable','--begin=now+'+str(max(1,math.ceil(due-time.time())))+'seconds',
                 '--output='+str(SESSION/'evidence/hourly_archive_%j.slurm.log'),str(script)]
        response=subprocess.check_output(command,text=True).strip();job=response.splitlines()[-1].split(';')[0]
        assert job.isdigit(),response
        atomic(control/'submission.json',dict(job_id=job,due_epoch=due,command=command,last_error=error))

if __name__=='__main__':main()
