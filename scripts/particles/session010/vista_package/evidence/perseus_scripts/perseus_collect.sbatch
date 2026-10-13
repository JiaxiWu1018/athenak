#!/bin/bash
#SBATCH -J jeans10-collect
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH --mail-user=jiaxiwu@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
SESSION=/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python3 "$SESSION/code/scripts/particles/session010/collect_handoff_evidence.py"
