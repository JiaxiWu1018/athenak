#!/bin/bash
#SBATCH -J st8b_prefix_repair
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH --no-requeue
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023/logs/%x.%j.log
set -euo pipefail
export ST8X_WALL_SECONDS=900 ST8_RECOVER_ONLY=1
bash /data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023/src/athenak/scripts/reconstruction_s08_multicase/launch_perseus.sh yc0500_long700 0
