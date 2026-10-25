#!/bin/bash
set -euo pipefail
ulimit -c 0
batch=/work1/eliasmost/jiaxiwu/st_s09_perturbed_production_20261024
export S8_MACHINE=amd ST8X_ROOT="$batch" ST8_CASE="$1" ROOT="$batch/$1"
export ST8X_BASE=/work1/eliasmost/jiaxiwu/st_reconstruction_s08_production_20261021
export ST8X_U="$2" ST8X_START_EPOCH=$(date +%s)
read -r st9_allocated_ranks ST8X_WALL_SECONDS < <(python3 - <<'LIMIT'
import os,re,subprocess
text=subprocess.check_output(['scontrol','show','job','-o',os.environ['SLURM_JOB_ID']],text=True)
tasks=int(re.search(r'\bNumTasks=(\d+)',text)[1])
nodes=int(re.search(r'\bNumNodes=(\d+)',text)[1])
partition=re.search(r'\bPartition=(\S+)',text)[1]
assert nodes==1 and tasks=={'mi2104x':4,'mi2508x':8}[partition],text
value=re.search(r'\bTimeLimit=(\S+)',text)[1]
requested=int(os.environ.get('ST8X_WALL_SECONDS','21600'))
if value=='UNLIMITED':print(tasks,requested)
else:
    days,clock=value.split('-',1) if '-' in value else ('0',value)
    hours,minutes,seconds=map(int,clock.split(':'))
    actual=int(days)*86400+hours*3600+minutes*60+seconds
    assert actual>120
    print(tasks,min(requested,actual))
LIMIT
)
# Slurm resource amendments can leave submission-time rank variables stale.
# Bind admission, capacity, and the MPI environment to the actual allocation.
export ST8X_WALL_SECONDS SLURM_NTASKS="$st9_allocated_ranks"
export SLURM_NPROCS="$st9_allocated_ranks" SLURM_NTASKS_PER_NODE="$st9_allocated_ranks" SLURM_TASKS_PER_NODE="$st9_allocated_ranks"
export S8_MAX_TIME=301 ST8X_GUARDS_CORRECTED=1
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg TZ=UTC
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export UCX_TLS='^rocm_ipc' UCX_MEMTYPE_CACHE=n UCX_LOG_LEVEL=info
record="$ROOT/evidence/jobs/$SLURM_JOB_ID"
mkdir -p "$record/workflow" "$ST8X_ROOT/evidence/locks"
exec 9>"$ST8X_ROOT/evidence/locks/$ST8_CASE.lock"
flock -n 9 || { echo 'Duplicate case allocation'; exit 1; }
cp -p "$batch/scripts/"*.py "$record/workflow/"
cp -p "$0" "$record/launch_amd.sh"
export PYTHONPATH="$record/workflow:$ST8X_BASE/dependencies/site:$ST8X_BASE/src/athenak/scripts/reconstruction_s05:$ST8X_BASE/src/athenak/vis/python"
finish() {
 status=$?
 if [ -n "${guard_pid:-}" ]; then kill "$guard_pid" 2>/dev/null || true; fi
 python3 - "$record" "$status" <<'END'
from common import *
import sys,time
atomic(Path(sys.argv[1])/'ALLOCATION_END.json',dict(job=job(),utc=now(),nodes=int(os.environ['SLURM_NNODES']),ranks=int(os.environ['SLURM_NTASKS']),elapsed_before_exit_seconds=time.time()-float(os.environ['ST8X_START_EPOCH']),exit_status=int(sys.argv[2])))
END
}
trap finish EXIT
{ hostname; date -u; module list; hipcc --version; scontrol show job "$SLURM_JOB_ID"; mpirun --version; rocm-smi --showproductname --showmeminfo vram; } > "$record/environment.txt" 2>&1
sha256sum "$record/workflow/"* > "$record/workflow.sha256"
python3 - "$record" <<'PROVENANCE'
from common import *
import sys
record=Path(sys.argv[1]);repo=EXT/'src/athenak'
assert not subprocess.check_output(['git','-C',str(repo),'diff','2d1f2525','--','src','CMakeLists.txt','cmake','kokkos'],text=True)
atomic(record/'DEPLOYMENT.json',dict(source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),python_files={p.name:sha(p) for p in (record/'workflow').glob('*.py')},native_source_unchanged=True))
PROVENANCE
python3 "$record/workflow/extension_resources.py" > "$record/watchdog.log" 2>&1 &
guard_pid=$!
python3 "$record/workflow/trajectory.py" > "$record/campaign.log" 2>&1
