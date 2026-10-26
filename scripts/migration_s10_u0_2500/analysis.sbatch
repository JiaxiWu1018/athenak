#!/bin/bash
#SBATCH -J st10u0_analysis
#SBATCH -A eliasmost
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 00:30:00
#SBATCH -o /work1/eliasmost/jiaxiwu/st_migration_s10_u0_2500_20261026/logs/%x.%j.log
set -euo pipefail
export S10_ROLE=analysis
bash /work1/eliasmost/jiaxiwu/st_migration_s10_u0_2500_20261026/scripts/launch_amd.sh
