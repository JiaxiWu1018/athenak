#!/bin/bash
#SBATCH --job-name=eff-numerics
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/data/jiaxiwu/NRPIC/paper/session_02_code_efficiency/logs/numerics.%j.log
set -euo pipefail
SESSION=${EFFICIENCY_SESSION:-/data/jiaxiwu/NRPIC/paper/session_02_code_efficiency}
export PATH=/home/emost/codes/openmpi/bin:/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
python3 "$SESSION/scripts/validate_numerics.py"
python3 "$SESSION/scripts/analyze.py"
