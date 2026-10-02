#!/bin/bash
#SBATCH -J jeans8-final-checks
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
set -euo pipefail
ROOT=/data/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/tmp/jeans8_checks_${SLURM_JOB_ID}
date -u
hostname
python3 -m unittest discover -s "$ROOT/code/scripts/particles/session008" -p 'test_*.py'
python3 -m py_compile "$ROOT/code/scripts/particles/session008/"*.py
for file in "$ROOT/code/scripts/particles/session008/"*.sh "$ROOT/code/scripts/particles/session008/"*.sbatch "$ROOT/code/scripts/particles/session008/"*.inc; do bash -n "$file"; done
rm -rf "$PYTHONPYCACHEPREFIX"
printf 'SESSION008_FINAL_CHECKS_PASSED\n'
