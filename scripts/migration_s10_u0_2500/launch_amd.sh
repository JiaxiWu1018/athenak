#!/bin/bash
set -euo pipefail
ulimit -c 0
batch=/work1/eliasmost/jiaxiwu/st_migration_s10_u0_2500_20261026
export S8_MACHINE=amd ST8X_ROOT="$batch" ST8_CASE=yc0500_control ROOT="$batch/yc0500_control"
export ST8X_BASE=/work1/eliasmost/jiaxiwu/st_reconstruction_s08_production_20261021
export ST8X_START_EPOCH=$(date +%s) S8_MAX_TIME=2501 ST8X_GUARDS_CORRECTED=1
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg TZ=UTC
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export UCX_TLS='^rocm_ipc' UCX_MEMTYPE_CACHE=n UCX_LOG_LEVEL=info
unset UCX_NET_DEVICES  # Use ports present on the actual allocated node.
export S10_CPU_ONLY=1
if [ "$S10_ROLE" = production ]; then export S10_CPU_ONLY=0; fi
record="$batch/evidence/allocations/$SLURM_JOB_ID"
mkdir -p "$record/workflow" "$batch/evidence/locks"
exec 9>"$batch/evidence/locks/yc0500_control.lock"
flock -n 9 || { echo "Another continuation stage is active"; exit 1; }
cp -p "$batch/scripts/"*.py "$batch/scripts/cases.json" "$record/workflow/"
cp -p "$0" "$record/launch_amd.sh"
export PYTHONPATH="$record/workflow:$ST8X_BASE/dependencies/site:$batch/src/athenak/scripts/reconstruction_s05:$batch/src/athenak/vis/python"
finish() {
  status=$?
  if [ -n "${guard_pid:-}" ]; then kill "$guard_pid" 2>/dev/null || true; fi
  python3 - "$record" "$status" <<'END'
from common import *
import sys,time
atomic(Path(sys.argv[1])/'ALLOCATION_END.json',dict(job=job(),utc=now(),nodes=int(os.environ['SLURM_NNODES']),ranks=int(os.environ['SLURM_NTASKS']),elapsed_seconds=time.time()-float(os.environ['ST8X_START_EPOCH']),exit_status=int(sys.argv[2]),scope='Pre-exit worker wall duration. Final allocated cost from Slurm; reserve full request until terminal evidence.'))
END
  if [ "$status" = 0 ]; then python3 "$record/workflow/dispatch.py" "$SLURM_JOB_ID"; fi
}
trap finish EXIT
{ hostname; date -u; module list; hipcc --version; scontrol show job "$SLURM_JOB_ID"; mpirun --version; } > "$record/environment.txt" 2>&1
if [ "$S10_ROLE" = production ]; then
  rocm-smi --showproductname --showdriverversion --json > "$record/gpu_hardware.json"
fi
sha256sum "$record/workflow/"* > "$record/workflow.sha256"
actual_wall_seconds=$(python3 "$record/workflow/allocation_budget.py")
export ST8X_WALL_SECONDS="$actual_wall_seconds"
python3 "$record/workflow/extension_resources.py" > "$record/watchdog.log" 2>&1 &
guard_pid=$!
python3 "$record/workflow/worker.py" > "$record/campaign.log" 2>&1
