#!/bin/bash
#SBATCH -J st_s06_qa
#SBATCH -p preempt
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/06_yc0819_long_stability_20261016/logs/%x.%j.log
set -euo pipefail
export ROOT=/data/jiaxiwu/NRPIC/reconstruction_ST/06_yc0819_long_stability_20261016
export ATHENAK_VIS_PYTHON="$ROOT/src/athenak/vis/python"
export PYTHONPATH="$ROOT/src/athenak/scripts/reconstruction_s06:$ROOT/src/athenak/scripts/reconstruction_s05:$ATHENAK_VIS_PYTHON"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 "$ROOT/scripts/verify.py"
