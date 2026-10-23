#!/bin/bash
set -euo pipefail
ulimit -c 0
export ST8X_ROOT=/work1/eliasmost/jiaxiwu/st_s08_yc0500_extensions_20261023
export ST8X_BASE=/work1/eliasmost/jiaxiwu/st_reconstruction_s08_production_20261021
mode=$1
submitted_mode=$mode
dispatch="$ST8X_ROOT/evidence/pending_dispatch.json"
if [ -f "$dispatch" ]; then
 mode=$(python3 - "$dispatch" "$SLURM_JOB_ID" "$mode" <<'DISPATCH'
import json,sys
record=json.load(open(sys.argv[1])).get(sys.argv[2])
if record:
 assert record['submitted_mode']==sys.argv[3] and record['run_mode']=='recover_long'
 print(record['run_mode'])
else:print(sys.argv[3])
DISPATCH
)
fi
export ROOT="$ST8X_ROOT/$mode"
lock_mode=$mode
if [ "$mode" = recover_long ]; then export ROOT="$ST8X_ROOT/long"; lock_mode=long; fi
export ST8X_U=0
if [ "$mode" = inward0020 ]; then export ST8X_U=-0.02; fi
if [ "$mode" = inward0050 ]; then export ST8X_U=-0.05; fi
export PYTHONPATH="$ST8X_ROOT/scripts:$ST8X_ROOT/dependencies/site:$ST8X_BASE/src/athenak/scripts/reconstruction_s05:$ST8X_BASE/src/athenak/vis/python:$ST8X_BASE/dependencies/site"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TZ=UTC
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
# 475604 aborted in UCX1.18.1 rocm_ipc_cache_map_memhandle. Retain GPU-aware
# rocm_copy and other transports, excluding the failing ROCm IPC transport.
export UCX_TLS='^rocm_ipc' UCX_MEMTYPE_CACHE=n UCX_LOG_LEVEL=info
mkdir -p "$ROOT/evidence/jobs/$SLURM_JOB_ID"
record="$ROOT/evidence/jobs/$SLURM_JOB_ID"
mkdir -p "$ST8X_ROOT/evidence/locks"
exec 9>"$ST8X_ROOT/evidence/locks/$lock_mode.lock"
if ! flock -n 9; then
 echo "duplicate candidate: another allocation owns $mode" > "$record/DUPLICATE_NOT_STARTED.txt"
 exit 0
fi
start_epoch=$(date +%s)
export ST8X_START_EPOCH=$start_epoch
export ST8X_WALL_SECONDS=${ST8X_WALL_SECONDS:-21600}
finish() {
 status=$?
 if [ -n "${guard_pid:-}" ]; then kill "$guard_pid" 2>/dev/null || true; fi
 python3 - "$record" "$start_epoch" "$status" <<'END_RECEIPT'
from common import *
import sys,time
atomic(Path(sys.argv[1])/'ALLOCATION_END.json',dict(utc=now(),job=job(),elapsed=time.time()-int(sys.argv[2]),nodes=int(os.environ['SLURM_NNODES']),ranks=int(os.environ['SLURM_NTASKS']),exit_status=int(sys.argv[3])))
END_RECEIPT
}
trap finish EXIT
{ hostname; date -Is; module list 2>&1; hipcc --version; command -v prun; printf 'UCX_TLS=%s\nUCX_MEMTYPE_CACHE=%s\nUCX_LOG_LEVEL=%s\n' "$UCX_TLS" "$UCX_MEMTYPE_CACHE" "$UCX_LOG_LEVEL"; scontrol show job "$SLURM_JOB_ID"; rocm-smi --showproductname --showmeminfo vram; } > "$record/environment.txt" 2>&1
mkdir -p "$record/workflow"
cp -p "$ST8X_ROOT"/scripts/*.py "$ST8X_ROOT"/scripts/*.sh "$record/workflow/"
cp -p "$ST8X_ROOT/evidence/SOURCE_PIN.json" "$record/NATIVE_SOURCE_PIN.json"
if [ -f "$dispatch" ]; then cp -p "$dispatch" "$record/pending_dispatch.json"; fi
printf 'submitted_mode=%s\nexecution_mode=%s\n' "$submitted_mode" "$mode" > "$record/dispatch.txt"
if [ -f "$ST8X_ROOT/evidence/NATIVE_ACTIVE.json" ]; then cp -p "$ST8X_ROOT/evidence/NATIVE_ACTIVE.json" "$record/NATIVE_ACTIVE.json"; fi
sha256sum "$record"/workflow/* > "$record/workflow.sha256"
export PYTHONPATH="$record/workflow:$PYTHONPATH"
python3 "$record/workflow/extension_resources.py" > "$record/watchdog.log" 2>&1 &
guard_pid=$!
python3 "$record/workflow/campaign.py" "$mode" > "$record/campaign.log" 2>&1
