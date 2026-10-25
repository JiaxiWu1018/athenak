#!/bin/bash
#SBATCH -J st9_support
#SBATCH -A eliasmost
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 00:30:00
#SBATCH --no-requeue
#SBATCH -o /work1/eliasmost/jiaxiwu/st_s09_perturbed_production_20261024/logs/%x.%j.log
set -euo pipefail
batch=/work1/eliasmost/jiaxiwu/st_s09_perturbed_production_20261024
export S8_MACHINE=amd ST8X_ROOT="$batch" ST8_CASE=yc0283_u0005 ROOT="${ST9_REVIEW_ROOT:-$batch/final}"
if [ "${1:-}" = closeout ]; then
  export ST8_CASE="$2" ROOT="$batch/$2"
fi
export ST8X_START_EPOCH=$(date +%s)
export ST8X_WALL_SECONDS=$(python3 - <<'LIMIT'
import os,re,subprocess
text=subprocess.check_output(['scontrol','show','job','-o',os.environ['SLURM_JOB_ID']],text=True)
value=re.search(r'\bTimeLimit=(\S+)',text)[1]
days,clock=value.split('-',1) if '-' in value else ('0',value)
h,m,s=map(int,clock.split(':'))
print(int(days)*86400+h*3600+m*60+s)
LIMIT
)
export ST8X_BASE=/work1/eliasmost/jiaxiwu/st_reconstruction_s08_production_20261021
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=UTC
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
record="$ROOT/evidence/jobs/$SLURM_JOB_ID"
mkdir -p "$record/workflow" "$batch/evidence/locks"
cp -p "$batch/scripts/"*.py "$record/workflow/"
cp -p "$0" "$record/executed.sbatch"
export PYTHONPATH="$record/workflow:$ST8X_BASE/dependencies/site:$ST8X_BASE/src/athenak/scripts/reconstruction_s05:$ST8X_BASE/src/athenak/vis/python"
exec 9>"$batch/evidence/locks/CPU_SUPPORT.lock"
flock -n 9
{ hostname; date -u; module list; scontrol show job "$SLURM_JOB_ID"; } > "$record/environment.txt" 2>&1
python3 - "$record" <<'SOURCE'
from common import *
import sys
repo=EXT/'src/athenak'
assert not subprocess.check_output(['git','-C',str(repo),'diff','2d1f2525','--','src','CMakeLists.txt','cmake','kokkos'],text=True)
record=Path(sys.argv[1])
atomic(record/'DEPLOYMENT.json',dict(source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),python_files={p.name:sha(p) for p in (record/'workflow').glob('*.py')},native_source_unchanged=True))
SOURCE
python3 "$record/workflow/extension_resources.py" --cpu > "$record/cpu_watchdog.log" 2>&1 &
guard_pid=$!
finish() { kill "$guard_pid" 2>/dev/null || true; }
trap finish EXIT
python3 "$record/workflow/support.py" "$@" > "$record/support.log" 2>&1
scontrol show job "$SLURM_JOB_ID" > "$record/scontrol_before_exit.txt"
