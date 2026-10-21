#!/bin/bash
#SBATCH -J st_s08_transfer
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -t 00:45:00
#SBATCH --no-requeue
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/perseus/logs/%x.%j.log
set -euo pipefail
source /data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/src/athenak/scripts/reconstruction_s08/perseus_env.sh
record=$ROOT/evidence/jobs/$SLURM_JOB_ID
mkdir -p "$record"
{ date -u; hostname; scontrol show job "$SLURM_JOB_ID"; df -B1 /data; } > "$record/environment.txt"
python3 - <<'PYCODE'
from common import *
import shutil
assert shutil.disk_usage(ROOT).free>2*TIB
assert not (ROOT/'evidence/IMPORT_SUSPENDED.json').exists(),'Source must be re-inventoried before resuming this suspended copy'
if (ROOT/'runs').exists():
    # Only resume the preempted, never-evolved copy of this same immutable source.
    assert (ROOT/'evidence/jobs/18652/SOURCE_HOLD.json').exists()
    assert not (ROOT/'evidence/IMPORT_VERIFIED.json').exists()
    assert not (ROOT/'evidence/MIGRATION_RESTART_OK.json').exists()
    assert not (ROOT/'evidence/PRODUCTION_ADMISSION.json').exists()
    assert load(ROOT/'evidence/PERSEUS_EXPORT.json')['job']=='473944'
q=subprocess.check_output(['ssh','-o','BatchMode=yes','hpcfund.amd.com','scontrol show job 473719'],text=True)
assert 'JobState=PENDING' in q and 'JobHeldUser' in q,q
atomic(ROOT/'evidence/jobs'/job()/'SOURCE_HOLD.json',dict(utc=now(),scontrol=q))
PYCODE
rsync -a --partial --partial-dir=.rsync-partial --stats --files-from="$ROOT/evidence/PERSEUS_EXPORT.files" hpcfund.amd.com:/work1/eliasmost/jiaxiwu/st_reconstruction_s08_production_20261021/ "$ROOT/" > "$record/rsync.log" 2>&1
python3 "$ROOT/scripts/perseus_import.py" > "$record/verify.log" 2>&1
du -s -B1 "$ROOT" > "$record/storage.txt"
