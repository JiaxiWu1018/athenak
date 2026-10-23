#!/bin/bash
set -euo pipefail
ulimit -c 0
export ST8X_ROOT=/work1/eliasmost/jiaxiwu/st_s08_yc0500_extensions_20261023
export ST8X_BASE=/work1/eliasmost/jiaxiwu/st_reconstruction_s08_production_20261021
mode=$1
export ROOT="$ST8X_ROOT/$mode"
export ST8X_U=0
if [ "$mode" = inward0020 ]; then export ST8X_U=-0.02; fi
if [ "$mode" = inward0050 ]; then export ST8X_U=-0.05; fi
export PYTHONPATH="$ST8X_ROOT/scripts:$ST8X_BASE/src/athenak/scripts/reconstruction_s05:$ST8X_BASE/src/athenak/vis/python:$ST8X_BASE/dependencies/site"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TZ=UTC
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mkdir -p "$ROOT/evidence/jobs/$SLURM_JOB_ID"
record="$ROOT/evidence/jobs/$SLURM_JOB_ID"
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
{ hostname; date -Is; module list 2>&1; hipcc --version; command -v prun; scontrol show job "$SLURM_JOB_ID"; rocm-smi --showproductname --showmeminfo vram; } > "$record/environment.txt" 2>&1
mkdir -p "$record/workflow"
cp -p "$ST8X_ROOT"/scripts/*.py "$ST8X_ROOT"/scripts/*.sh "$record/workflow/"
cp -p "$ST8X_ROOT/evidence/SOURCE_PIN.json" "$record/NATIVE_SOURCE_PIN.json"
sha256sum "$record"/workflow/* > "$record/workflow.sha256"
export PYTHONPATH="$record/workflow:$PYTHONPATH"
python3 "$record/workflow/extension_resources.py" > "$record/watchdog.log" 2>&1 &
guard_pid=$!
python3 "$record/workflow/campaign.py" "$mode" > "$record/campaign.log" 2>&1
