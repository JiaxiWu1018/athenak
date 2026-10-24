#!/bin/bash
#SBATCH -J st9_yc0605_u0005
#SBATCH -A eliasmost
#SBATCH -p mi2508x
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -c 2
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 08:00:00
#SBATCH -o /work1/eliasmost/jiaxiwu/st_s09_perturbed_production_20261024/logs/%x.%j.log
set -euo pipefail
export ST8X_WALL_SECONDS=28800
bash /work1/eliasmost/jiaxiwu/st_s09_perturbed_production_20261024/scripts/launch_amd.sh yc0605_u0005 -0.005
