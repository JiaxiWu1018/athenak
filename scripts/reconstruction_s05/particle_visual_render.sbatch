#!/bin/bash
#SBATCH -J st_s05_particle_viz
#SBATCH -p preempt
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014/logs/particle_viz.%j.log
set -euo pipefail
export ROOT=/data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
repo="$ROOT/src/athenak"
mkdir -p "$ROOT/evidence/jobs/$SLURM_JOB_ID"
git -C "$repo" diff --exit-code HEAD
git -C "$repo" rev-parse HEAD > "$ROOT/evidence/jobs/$SLURM_JOB_ID/visualization_source.txt"
{ hostname; date -Is; python3 --version; ffmpeg -version; } > "$ROOT/evidence/jobs/$SLURM_JOB_ID/environment.txt"
python3 "$repo/scripts/reconstruction_s05/particle_visual_render.py" "$@"
