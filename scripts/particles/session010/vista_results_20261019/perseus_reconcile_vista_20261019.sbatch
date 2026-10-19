#!/bin/bash
#SBATCH -J jeans10-reconcile
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -t 00:25:00
#SBATCH --mail-user=jiaxiwu@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
SESSION=/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$SESSION/evidence/vista_reconciliation_20261019/matplotlib_cache"
mkdir -p "$SESSION/evidence/vista_reconciliation_20261019"
{ date -u; hostname; scontrol show job "$SLURM_JOB_ID"; python3 --version; df -h /data; } > "$SESSION/evidence/vista_reconciliation_20261019/slurm_${SLURM_JOB_ID}.txt"
python3 "$SESSION/code/scripts/particles/session010/vista_results_20261019/reconcile.py" --session "$SESSION"
