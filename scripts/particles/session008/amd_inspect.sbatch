#!/bin/bash
#SBATCH -J jeans8-inspect
#SBATCH -A eliasmost
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=/work1/eliasmost/jiaxiwu/gi_s008_amd_20261002
python3 "$ROOT/scripts/workflow.py" inspect --segment "${1:?index}" --predecessor "${2:?job id}"
