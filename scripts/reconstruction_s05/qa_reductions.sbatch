#!/bin/bash
#SBATCH -J st_s05_reduceqa
#SBATCH -p preempt
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gpus=1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014/logs/reduceqa.%j.log
set -euo pipefail
session=/data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014
export ROOT="$session" PYTHONPATH="$session/src/athenak/scripts/reconstruction_s05"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 "$session/src/athenak/scripts/reconstruction_s05/qa_reductions.py" "$@"
