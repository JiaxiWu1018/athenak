#!/bin/bash
#SBATCH -J st8b_long700
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 1
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH -t 06:00:00
#SBATCH --no-requeue
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023/logs/%x.%j.log
set -euo pipefail
export ST8X_WALL_SECONDS=21600
bash /data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023/src/athenak/scripts/reconstruction_s08_multicase/launch_perseus.sh yc0500_long700 0
