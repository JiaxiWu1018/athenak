#!/bin/bash
#SBATCH -J st9_verify_mirror
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 00:15:00
#SBATCH --no-requeue
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/09_perturbed_production_20261024/logs/%x.%j.log
set -euo pipefail
session=/data/jiaxiwu/NRPIC/reconstruction_ST/09_perturbed_production_20261024
record="$session/evidence/perseus_mirror/$SLURM_JOB_ID"
mkdir -p "$record"
cp -p "$session/src/athenak/scripts/reconstruction_s09_production/verify_mirror.py" "$record/verify_mirror.py"
cp -p "$0" "$record/executed.sbatch"
{ hostname; date -u; python3 --version; scontrol show job "$SLURM_JOB_ID"; } > "$record/environment.txt" 2>&1
sha256sum "$record/verify_mirror.py" > "$record/source.sha256"
python3 "$record/verify_mirror.py" "$session/review_mirror" > "$record/verification.log" 2>&1
scontrol show job "$SLURM_JOB_ID" > "$record/scontrol_before_exit.txt"
