#!/bin/bash
#SBATCH -J jeans10-focused
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH --mail-user=jiaxiwu@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
SESSION=/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013
export PATH=/home/emost/codes/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:${LD_LIBRARY_PATH:-}
export OMPI_MCA_hwloc_base_binding_policy=none OMPI_MCA_rmaps_base_mapping_policy=slot
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PIP_CACHE_DIR=$SESSION/evidence/pip_cache
mkdir -p "$PIP_CACHE_DIR" "$SESSION/python_deps"
python3 -m pip install --target "$SESSION/python_deps" pytest==8.3.5 > "$SESSION/evidence/pytest_install_${SLURM_JOB_ID}.log" 2>&1
export PYTHONPATH=$SESSION/python_deps:${PYTHONPATH:-}
ROOT=$SESSION/evidence/focused_${SLURM_JOB_ID}
REPO=$SESSION/code
mkdir -p "$ROOT"
{
 date -u;hostname;scontrol show job "$SLURM_JOB_ID";mpicxx --showme
 git -C "$REPO" rev-parse HEAD;git -C "$REPO" submodule status
 echo 'CPU/MPI tests; One GPU requested solely to satisfy Perseus CPU/GPU placement policy.'
} > "$ROOT/provenance.txt" 2>&1
for kind in crossing deposition feedback; do
 case "$kind" in
 crossing) problem=particles/part_crossing;;
 deposition) problem=particles/part_tmunu_test;;
 feedback) problem=z4c/z4c_one_puncture;;
 esac
 build=$ROOT/$kind
 if [ "$kind" = crossing ]; then build=$SESSION/evidence/focused_12758/crossing; fi
 cmake -S "$REPO" -B "$build" -DAthena_ENABLE_MPI=ON -DKokkos_ENABLE_SERIAL=ON \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DPROBLEM="$problem" \
  > "$ROOT/$kind.configure.log" 2>&1
 cmake --build "$build" -j 2 > "$ROOT/$kind.build.log" 2>&1
 ln -sfn "$build" "$REPO/tst/build"
 cp -a "$REPO/tst/inputs" "$build/src/inputs"
 python3 - "$REPO" "$build" "$kind" <<'PY' > "$ROOT/$kind.tests.log" 2>&1
import os,sys
from pathlib import Path
repo,build=map(Path,sys.argv[1:3]);kind=sys.argv[3]
os.chdir(repo/'tst');sys.path.insert(0,str(repo/'tst'))
import test_suite.testutils
_original_run=test_suite.testutils.run_command
def _allocated_run(command,**kwargs):
 if command[0]=="mpirun":command=[command[0],"--bind-to","none","--map-by","slot"]+command[1:]
 return _original_run(command,**kwargs)
test_suite.testutils.run_command=_allocated_run
import test_suite.particles.part_helpers as helpers
helpers.rank_counts=lambda requested: [r for r in (1,2) if r==1 or any(x>1 for x in requested)]
import pytest
files=dict(crossing=['test_part_crossing_mpicpu.py','test_part_amr_mpicpu.py','test_part_destroy_burst_mpicpu.py','test_part_farhop_fatal_mpicpu.py'],deposition=['test_part_tmunu_xlevel_mpicpu.py'],feedback=['test_part_feedback_amr_mpicpu.py'])[kind]
paths=[str(repo/'tst/test_suite/particles'/x) for x in files]
os.chdir(build/'src')
args=['-v','--tb=short']+paths
if kind=='deposition':args+=['-k','conservative']
sys.exit(pytest.main(args))
PY
 cp "$REPO/tst/test_log.txt" "$ROOT/$kind.application.log"
 sha256sum "$build/src/athena" > "$ROOT/$kind.sha256"
done
printf '{"passed":true,"job_id":%s,"root":"%s"}\n' "$SLURM_JOB_ID" "$ROOT" > "$SESSION/evidence/focused_receipt.json"
