#!/bin/bash
#SBATCH -J st_s07_thin_mirror
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
export PYTHONPATH="$ROOT/scripts:$ROOT/src/athenak/scripts/reconstruction_s05"
python3 "$ROOT/scripts/mirror.py"
