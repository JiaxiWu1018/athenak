#!/bin/bash
#SBATCH -J jn9_20261009_stopped_review1
#SBATCH -p long
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=/data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /home/jiaxiwu/miniconda3/bin/python "$ROOT/reviews/stopped_run_20261009/review.py"
