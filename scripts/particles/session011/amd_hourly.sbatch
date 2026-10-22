#!/bin/bash
#SBATCH -J s11-hourly-products
#SBATCH -A eliasmost
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH --no-requeue
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export MPLCONFIGDIR=$SESSION_ROOT/.mpl_cache
python3 "$HOURLY_TOOLING/hourly.py" --root "$SESSION_ROOT"
