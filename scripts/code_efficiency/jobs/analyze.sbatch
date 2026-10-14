#!/bin/bash
#SBATCH --job-name=eff-report
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/data/jiaxiwu/NRPIC/paper/session_02_code_efficiency/logs/report.%j.log
set -euo pipefail
SESSION=${EFFICIENCY_SESSION:-/data/jiaxiwu/NRPIC/paper/session_02_code_efficiency}
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 "$SESSION/scripts/analyze.py"
