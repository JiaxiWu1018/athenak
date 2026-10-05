#!/bin/bash
#SBATCH -J jeans9-inspect
#SBATCH -A eliasmost
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=/work1/eliasmost/jiaxiwu/gi_s009_amd_20261005
python3 "$ROOT/scripts/workflow.py" inspect --segment "${1:?index}" --predecessor "${2:?job id}"
