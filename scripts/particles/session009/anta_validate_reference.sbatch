#!/bin/bash
#SBATCH -J jeans9-reference-validation
#SBATCH -p long
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=/data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
failed() {
 export REFERENCE_VALIDATION_RC=$?
 trap - ERR
 python3 - <<'PY'
import os,sys,time,subprocess
from pathlib import Path
root=Path('/data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005')
sys.path.insert(0,str(root/'scripts'))
from runtime import atomic
from anta_archive import SSH
row=dict(utc=time.time(),job=os.environ['SLURM_JOB_ID'],exit_code=int(os.environ['REFERENCE_VALIDATION_RC']),reason='Saved-reference validation/publication failed; stop for review, no automatic retry.')
atomic(root/'evidence/REFERENCE_VALIDATION_FAILURE.json',row)
atomic(root/'evidence/ARCHIVE_HALTED.json',row)
code="""import sys
from pathlib import Path
root=Path('/work1/eliasmost/jiaxiwu/gi_s009_amd_20261005')
sys.path.insert(0,str(root/'scripts'))
from workflow import locked,readstate,terminal,TERMINAL
with locked():
 s=readstate()
 if s['status'] not in TERMINAL:terminal(s,'configuration_failure','Allocated Anta reference validation/publication failed; inspect Anta evidence, no automatic retry')
 (root/'control/ARCHIVE_ERROR').touch()
"""
subprocess.run(SSH+['python3 -'],input=code,text=True,check=True,timeout=120)
PY
 exit "$REFERENCE_VALIDATION_RC"
}
trap failed ERR
/home/jiaxiwu/miniconda3/bin/python "$ROOT/scripts/validate_s9.py" --root "$ROOT"
/home/jiaxiwu/miniconda3/bin/python "$ROOT/scripts/publish_reference_validation.py"
