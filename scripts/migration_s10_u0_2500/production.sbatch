#!/bin/bash
#SBATCH -J st10u0_production
#SBATCH -A eliasmost
#SBATCH -p mi2508x
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -c 2
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 01:30:00
#SBATCH -o /work1/eliasmost/jiaxiwu/st_migration_s10_u0_2500_20261026/logs/%x.%j.log
set -euo pipefail
export S10_ROLE=production
bash /work1/eliasmost/jiaxiwu/st_migration_s10_u0_2500_20261026/scripts/launch_amd.sh
