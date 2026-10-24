#!/bin/bash
#SBATCH -J st10_production
#SBATCH -A eliasmost
#SBATCH -p mi2508x
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -c 2
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 02:00:00
#SBATCH -o /work1/eliasmost/jiaxiwu/st_migration_s10_20261024/logs/%x.%j.log
set -euo pipefail
export S10_CPU_ONLY=0 ST8X_WALL_SECONDS=7200
bash /work1/eliasmost/jiaxiwu/st_migration_s10_20261024/scripts/launch_amd.sh production "${S10_CASE:-yc0500_outward}" "${S10_ENDPOINT:-20}" "${S10_U:-0.005}"
