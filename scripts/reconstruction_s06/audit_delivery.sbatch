#!/bin/bash
#SBATCH -J st_s06_delivery
#SBATCH -p preempt
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/06_yc0819_long_stability_20261016/logs/%x.%j.log
set -euo pipefail
export ROOT=/data/jiaxiwu/NRPIC/reconstruction_ST/06_yc0819_long_stability_20261016
export PYTHONPATH="$ROOT/src/athenak/scripts/reconstruction_s06:$ROOT/src/athenak/scripts/reconstruction_s05:$ROOT/src/athenak/vis/python"
python3 "$ROOT/scripts/audit_delivery.py"
