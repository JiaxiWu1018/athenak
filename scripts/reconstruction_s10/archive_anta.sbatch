#!/bin/bash
#SBATCH --job-name=st10_archive
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 - "${1:?pass the reviewed archive manifest on Anta}" <<'PY'
import hashlib,json,os,re,shutil,socket,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path

job=os.environ['SLURM_JOB_ID']
assert re.fullmatch(r'[0-9]+',job)
plan_path=Path(sys.argv[1]).resolve()
plan=json.loads(plan_path.read_text())
source='/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102'
archive=Path('/data2/jiaxiwu/NRPIC/reconstruction_ST/10_rk3_predictor_20261102')
assert plan['source_root']==source
assert plan['status']=='READY_FOR_ARCHIVE'
assert plan['deletion_authorized'] is False
assert plan['projected_all_copies_bytes'] < 300*2**30
assert plan['projected_total_node_hours'] < 40
assert plan['archive_job_max_node_hours']==1
rows=plan['files']
assert rows and len({r['path'] for r in rows})==len(rows)
for r in rows:
    p=Path(r['path'])
    assert not p.is_absolute() and '..' not in p.parts and str(p)==r['path']
    assert '\n' not in r['path'] and '\r' not in r['path'] and '\0' not in r['path']
    assert isinstance(r['bytes'],int) and r['bytes']>=0
    assert re.fullmatch(r'[0-9a-f]{64}',r['sha256'])
assert sum(r['bytes'] for r in rows)==plan['total_bytes']
archive.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(archive).free > plan['total_bytes']+10*2**30
out=archive/f'archive_job{job}'
out.mkdir(exist_ok=False)
(out/'ARCHIVE_PLAN.json').write_bytes(plan_path.read_bytes())
filelist=out/'FILES.txt'
filelist.write_text(''.join(r['path']+'\n' for r in rows))
command=['rsync','-a','--stats','--files-from='+str(filelist),
         '-e','ssh -o BatchMode=yes -o ConnectTimeout=30',
         'hpcfund.amd.com:'+source+'/',str(out/'data')+'/']
start=time.monotonic()
record=dict(job_id=job,host=socket.gethostname(),
            started_utc=datetime.now(timezone.utc).isoformat(),
            source_root=source,destination=str(out/'data'),
            plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            command=command,source_originals_retained=True,deletion_authorized=False,
            expected_files=len(rows),expected_bytes=plan['total_bytes'],status='RUNNING')
(out/'ARCHIVE_RESULT.json').write_text(json.dumps(record,indent=2)+'\n')
try:
    with (out/'transfer.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (out/'VERIFIED.jsonl').open('w') as verified:
        for r in rows:
            p=out/'data'/r['path']
            assert p.is_file() and not p.is_symlink() and p.stat().st_size==r['bytes'],p
            h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
            assert h.hexdigest()==r['sha256'],p
            verified.write(json.dumps(r)+'\n')
    record.update(status='VERIFIED',verified_files=len(rows),verified_bytes=plan['total_bytes'])
except BaseException as exc:
    record.update(status='FAILED',error=repr(exc))
    raise
finally:
    record.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.monotonic()-start)
    (out/'ARCHIVE_RESULT.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)
PY
