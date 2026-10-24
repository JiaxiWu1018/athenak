#!/bin/bash
#SBATCH -J st10_qa
#SBATCH -A eliasmost
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 00:20:00
#SBATCH -o /work1/eliasmost/jiaxiwu/st_migration_s10_20261024/logs/%x.%j.log
set -euo pipefail
export S10_CPU_ONLY=1 ST8X_WALL_SECONDS=1200
bash /work1/eliasmost/jiaxiwu/st_migration_s10_20261024/scripts/launch_amd.sh qa "${S10_CASE:-yc0500_outward}" 20 0.005
