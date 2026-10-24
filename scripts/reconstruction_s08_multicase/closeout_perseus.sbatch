#!/bin/bash
#SBATCH -J st8b_delivery
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 00:15:00
#SBATCH --no-requeue
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023/logs/%x.%j.log
set -euo pipefail
batch=/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023
workflow=${ST8_CLOSEOUT_WORKFLOW_DIR:-$batch/src/athenak/scripts/reconstruction_s08_multicase}
repo=$(realpath "$workflow/../..")
source_commit=$(git -C "$repo" rev-parse HEAD)
if [ -n "${ST8_CLOSEOUT_SOURCE_COMMIT:-}" ]; then
  test "$source_commit" = "$ST8_CLOSEOUT_SOURCE_COMMIT"
fi
export S8_MACHINE=perseus ST8X_ROOT="$batch/perseus" ST8_CASE=yc0500_long700 ROOT="$batch/perseus/final"
export ST8X_BASE=/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 TZ=UTC
export ST8X_START_EPOCH=$(date +%s)
record="$ROOT/evidence/jobs/$SLURM_JOB_ID"
mkdir -p "$record/workflow"
cp -p "$0" "$record/executed.sbatch"
cp -p "$workflow/"*.py "$record/workflow/"
export PYTHONPATH="$record/workflow"
finish() {
  status=$?
  if [ -n "${guard_pid:-}" ]; then kill "$guard_pid" 2>/dev/null || true; fi
  python3 - "$record" "$status" <<'END'
from common import *
import sys,time
atomic(Path(sys.argv[1])/'ALLOCATION_END.json', dict(utc=now(),job=job(),nodes=int(os.environ['SLURM_NNODES']),elapsed_before_exit_seconds=time.time()-float(os.environ['ST8X_START_EPOCH']),exit_status=int(sys.argv[2])))
END
  scontrol show job "$SLURM_JOB_ID" > "$record/scontrol_before_exit.txt" 2>&1 || true
}
trap finish EXIT
{ hostname; date -u; scontrol show job "$SLURM_JOB_ID"; python3 --version; ffmpeg -version; } > "$record/environment.txt" 2>&1
printf '%s\n' "$source_commit" > "$record/source_commit.txt"
sha256sum "$record/workflow/"* > "$record/workflow.sha256"
python3 "$record/workflow/extension_resources.py" > "$record/resource_guard.log" 2>&1 &
guard_pid=$!
python3 "$record/workflow/final_delivery.py" --source-commit "$source_commit" > "$record/delivery.log" 2>&1
