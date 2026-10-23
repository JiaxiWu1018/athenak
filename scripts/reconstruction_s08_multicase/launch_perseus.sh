#!/bin/bash
set -euo pipefail
ulimit -c 0
batch=/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023
export S8_MACHINE=perseus ST8X_ROOT="$batch/perseus" ST8_CASE="$1" ROOT="$batch/perseus/$1"
export ST8X_BASE=/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021
export ST8X_U="$2" ST8X_START_EPOCH=$(date +%s)
export PATH=/home/emost/codes/openmpi/bin:/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg TZ=UTC
export OMPI_MCA_hwloc_base_binding_policy=none OMPI_MCA_rmaps_base_mapping_policy=slot OMPI_MCA_opal_cuda_support=true OMPI_MCA_pml=ucx OMPI_MCA_btl_vader_single_copy_mechanism=none
export UCX_TLS=self,sm,cuda_copy,cuda_ipc UCX_MEMTYPE_CACHE=n
record="$ROOT/evidence/jobs/$SLURM_JOB_ID"
mkdir -p "$record/workflow" "$ST8X_ROOT/evidence/locks"
exec 9>"$ST8X_ROOT/evidence/locks/$ST8_CASE.lock"
flock -n 9 || { echo 'Duplicate case allocation'; exit 1; }
cp -p "$batch/src/athenak/scripts/reconstruction_s08_multicase/"*.py "$record/workflow/"
cp -p "$0" "$record/launch_perseus.sh"
export PYTHONPATH="$record/workflow:$ST8X_BASE/src/athenak/scripts/reconstruction_s05:$ST8X_BASE/src/athenak/vis/python"
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
{ hostname; date -u; scontrol show job "$SLURM_JOB_ID"; printf 'CUDA_VISIBLE_DEVICES=%s\nSLURM_JOB_GPUS=%s\n' "${CUDA_VISIBLE_DEVICES:-}" "${SLURM_JOB_GPUS:-}"; mpirun --version; nvidia-smi; } > "$record/environment.txt" 2>&1
sha256sum "$record/workflow/"* > "$record/workflow.sha256"
python3 "$record/workflow/extension_resources.py" > "$record/watchdog.log" 2>&1 &
guard_pid=$!
python3 "$record/workflow/trajectory.py" > "$record/campaign.log" 2>&1
