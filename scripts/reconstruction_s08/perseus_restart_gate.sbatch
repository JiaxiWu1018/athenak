#!/bin/bash
#SBATCH -J st_s08_restart_gate
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -c 1
#SBATCH --gres=gpu:3
#SBATCH --mem=192G
#SBATCH -t 00:20:00
#SBATCH --no-requeue
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/perseus/logs/%x.%j.log
set -euo pipefail
source /data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/src/athenak/scripts/reconstruction_s08/perseus_env.sh
record=$ROOT/evidence/jobs/$SLURM_JOB_ID
mkdir -p "$record"
{ date -u; hostname; scontrol show job "$SLURM_JOB_ID"; mpicxx --showme; nvidia-smi --query-gpu=uuid,name,memory.total --format=csv; } > "$record/environment.txt" 2>&1
python3 "$ROOT/scripts/perseus_restart_gate.py"
