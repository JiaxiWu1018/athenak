#!/bin/bash
#SBATCH -J jeans8-separation
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH -t 00:05:00
set -euo pipefail
ROOT=/data/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
date -u
hostname
python3 "$ROOT/code/scripts/particles/session008/plot_separation_20261005.py" \
  --tables "$ROOT/evidence/separation_20261005" \
  --horizons "$ROOT/analysis/horizon_review_20261005" \
  --output "$ROOT/analysis/separation_20261005"
