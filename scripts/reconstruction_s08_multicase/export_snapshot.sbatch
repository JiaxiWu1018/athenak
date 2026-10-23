#!/bin/bash
#SBATCH -J st8b_snapshot
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH --no-requeue
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023/logs/%x.%j.log
set -euo pipefail
batch=/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023
export S8_MACHINE=perseus ST8X_ROOT="$batch/perseus" ST8_CASE=yc0500_long700 ROOT="$batch/perseus/yc0500_long700"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=UTC
record="$ROOT/evidence/jobs/$SLURM_JOB_ID"
mkdir -p "$record/workflow"
cp -p "$0" "$record/executed.sbatch"
cp -p "$batch/src/athenak/scripts/reconstruction_s08_multicase/"*.py "$record/workflow/"
export PYTHONPATH="$record/workflow"
{ hostname; date -u; scontrol show job "$SLURM_JOB_ID"; } > "$record/environment.txt"
python3 "$record/workflow/export_snapshot.py" "$1" > "$record/export.log" 2>&1
scontrol show job "$SLURM_JOB_ID" > "$record/scontrol_before_exit.txt" 2>&1 || true
