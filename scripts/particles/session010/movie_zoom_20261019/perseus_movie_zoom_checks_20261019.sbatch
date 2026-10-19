#!/bin/bash
#SBATCH -J jeans10-zoom-check
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
OUT="$SESSION/movies/particles3d_to98_zoom_20261019_v2"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
{ date -u; hostname; scontrol show job "$SLURM_JOB_ID"; } > "$OUT/evidence/checks_slurm_${SLURM_JOB_ID}.txt"
python3 "$SESSION/code/scripts/particles/session010/movie_zoom_20261019/check_zoom.py" --session "$SESSION" --out "$OUT"
