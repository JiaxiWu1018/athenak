#!/bin/bash
#SBATCH -J st_s08_stage
#SBATCH -p debug
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH --no-requeue
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/logs/%x.%j.log
set -euo pipefail
export ROOT=/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021
export PYTHONPATH="$ROOT/src/athenak/scripts/reconstruction_s08"
python3 "$ROOT/scripts/stage_build.py"
