#!/bin/bash
#SBATCH -J st_s07_stage
#SBATCH -p preempt
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH --no-requeue
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/07_mc_particle_convergence_20261019/logs/%x.%j.log
set -euo pipefail
export ROOT=/data/jiaxiwu/NRPIC/reconstruction_ST/07_mc_particle_convergence_20261019
export PYTHONPATH="$ROOT/scripts:$ROOT/src/athenak/scripts/reconstruction_s05:$ROOT/src/athenak/vis/python"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$ROOT/evidence/jobs/$SLURM_JOB_ID"
{ hostname; date -Is; python3 --version; git -C "$ROOT/src/athenak" rev-parse HEAD; } > "$ROOT/evidence/jobs/$SLURM_JOB_ID/environment.txt"
python3 "$ROOT/scripts/stage.py"
