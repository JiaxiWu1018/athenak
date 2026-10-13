#!/bin/bash
#SBATCH -J jeans10-feedback
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH --mail-user=jiaxiwu@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
SESSION=/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013
export PATH=/home/emost/codes/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$SESSION/python_deps:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
ROOT=$SESSION/evidence/feedback_${SLURM_JOB_ID}
REPO=$SESSION/code
BUILD=$SESSION/evidence/focused_12766/feedback
mkdir -p "$ROOT"
{
 date -u;hostname;scontrol show job "$SLURM_JOB_ID";sha256sum "$BUILD/src/athena"
 git -C "$REPO" rev-parse HEAD
} > "$ROOT/provenance.txt" 2>&1
ln -sfn "$BUILD" "$REPO/tst/build"
python3 - "$REPO" "$BUILD" "$ROOT" <<'PY' > "$ROOT/tests.log" 2>&1
import os,sys,shutil,time,json,hashlib
from pathlib import Path
repo,build,root=map(Path,sys.argv[1:]);os.chdir(repo/'tst');sys.path.insert(0,str(repo/'tst'))
import test_suite.testutils as utils
import test_suite.particles.part_helpers as helpers
import pytest
original=utils.run_command
def launch(command,**kwargs):
 if command[0]=='mpirun':command=[command[0],'--bind-to','none','--map-by','slot']+command[1:]
 return original(command,**kwargs)
utils.run_command=launch
helpers.rank_counts=lambda requested:[1,2]
def preserve(*paths):
 for name in paths:
  p=Path(name)
  if not p.exists():continue
  dest=root/(p.name+'_'+str(time.time_ns()));rows=[]
  for f in p.rglob('*'):
   if f.is_file():rows.append(dict(path=str(f.relative_to(p)),bytes=f.stat().st_size))
  shutil.move(str(p),str(dest));(dest/'INVENTORY.json').write_text(json.dumps(rows,indent=2)+'\n')
helpers.remove_dirs=preserve
os.chdir(build/'src')
sys.exit(pytest.main(['-v','--tb=short',str(repo/'tst/test_suite/particles/test_part_feedback_amr_mpicpu.py')]))
PY
cp "$REPO/tst/test_log.txt" "$ROOT/application.log"
printf '{"passed":true,"job_id":%s,"root":"%s","crossing_deposition_job":12766}\n' "$SLURM_JOB_ID" "$ROOT" > "$SESSION/evidence/focused_receipt.json"
