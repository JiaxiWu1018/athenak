#!/bin/bash
#SBATCH -A eliasmost
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 00:30:00
#SBATCH --no-requeue
set -euo pipefail
source /work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/scripts/amd_env.inc
python3 "$ROOT/scripts/recovery_20261005/recovery.py" inspect --number "${1:?continuation number}"
