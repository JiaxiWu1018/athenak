#!/bin/bash
#SBATCH -J st8b_review
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 00:15:00
#SBATCH --no-requeue
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023/logs/%x.%j.log
set -euo pipefail
batch=/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023
export S8_MACHINE=perseus ST8X_ROOT="$batch/perseus" ST8_CASE=yc0500_long700
export ST8X_BASE=/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021
mode=$1
export ROOT="$ST8X_ROOT/${ST8_REVIEW_SUBDIR:-final}"
record="$ROOT/evidence/jobs/$SLURM_JOB_ID"
mkdir -p "$record/workflow"
cp -p "$0" "$record/executed.sbatch"
cp -p "$batch/src/athenak/scripts/reconstruction_s08_multicase/"*.py "$record/workflow/"
export PYTHONPATH="$record/workflow:$ST8X_BASE/src/athenak/scripts/reconstruction_s05:$ST8X_BASE/src/athenak/vis/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 TZ=UTC
export ST8X_START_EPOCH=$(date +%s)
finish() {
 status=$?
 if [ -n "${guard_pid:-}" ]; then kill "$guard_pid" 2>/dev/null || true; fi
 python3 - "$record" "$status" <<'END'
from common import *
import sys,time
atomic(Path(sys.argv[1])/'ALLOCATION_END.json',dict(utc=now(),job=job(),elapsed_before_exit_seconds=time.time()-float(os.environ['ST8X_START_EPOCH']),nodes=int(os.environ['SLURM_NNODES']),exit_status=int(sys.argv[2])))
END
 scontrol show job "$SLURM_JOB_ID" > "$record/scontrol_before_exit.txt" 2>&1 || true
}
trap finish EXIT
{ hostname; date -u; scontrol show job "$SLURM_JOB_ID"; python3 --version; ffmpeg -version; } > "$record/environment.txt" 2>&1
sha256sum "$record/workflow/"* > "$record/workflow.sha256"
python3 "$record/workflow/extension_resources.py" > "$record/resource_guard.log" 2>&1 &
guard_pid=$!
case "$mode" in
 import) python3 "$record/workflow/import_packets.py" ;;
 figures) shift; python3 "$record/workflow/products.py" "$@" ;;
 boundary) python3 "$record/workflow/boundary.py" ;;
 movie) python3 "$record/workflow/movies.py" "$2" ;;
 movie_chunk) python3 "$record/workflow/movie_chunks.py" render "$2" "${SLURM_ARRAY_TASK_ID:-${3:-0}}" ;;
 movie_encode) python3 "$record/workflow/movie_chunks.py" encode "$2" ;;
 reports) python3 "$record/workflow/reports.py" ;;
 *) echo "Unknown review operation: $mode"; exit 1 ;;
esac > "$record/review.log" 2>&1
