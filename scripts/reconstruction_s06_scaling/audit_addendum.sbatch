#!/bin/bash
#SBATCH -J st_s06ps_final_integrity
#SBATCH -p preempt
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH --no-requeue
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/06_yc0819_long_stability_20261016/followups/paired_scaling_20261017/logs/%x.%j.log
set -euo pipefail
export ROOT=/data/jiaxiwu/NRPIC/reconstruction_ST/06_yc0819_long_stability_20261016/followups/paired_scaling_20261017
export PYTHONPATH="$ROOT/scripts:$ROOT/src/athenak/scripts/reconstruction_s05:$ROOT/src/athenak/vis/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
notify_exit() {
  result=$?
  /home/jiaxiwu/.local/bin/codex queue --thread 01a14666-7bb8-74d1-8819-0037c04e9eb6 --message "ST session6 paired-scaling final-integrity job $SLURM_JOB_ID exited $result. Read $ROOT/HANDOFF.md and evidence/state.json; inspect evidence/FINAL_AUDIT_OK.json and separate addendum manifest, preserve terminal accounting, finish followup closeout. Event does not replace task." || true
  exit "$result"
}
trap notify_exit EXIT
mkdir -p "$ROOT/evidence/jobs/$SLURM_JOB_ID"
{ hostname; date -Is; python3 --version; git -C "$ROOT/src/athenak" rev-parse HEAD; } > "$ROOT/evidence/jobs/$SLURM_JOB_ID/environment.txt"
python3 -c 'from cfg import ROOT;from os_gate import require_release;require_release(ROOT)'
python3 "$ROOT/scripts/audit_addendum.py"
