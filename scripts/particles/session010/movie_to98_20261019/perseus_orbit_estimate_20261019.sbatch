#!/bin/bash
#SBATCH -J jeans10-orbit-estimate
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH --mail-user=jiaxiwu@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
SESSION=/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$SESSION/movies/particles3d_to98_20261019_v1/evidence/matplotlib_cache"
{ date -u; hostname; scontrol show job "$SLURM_JOB_ID"; python3 --version; } > "$SESSION/movies/particles3d_to98_20261019_v1/evidence/orbit_slurm_${SLURM_JOB_ID}.txt"
python3 "$SESSION/code/scripts/particles/session010/movie_to98_20261019/orbit_estimate.py" --session "$SESSION"
