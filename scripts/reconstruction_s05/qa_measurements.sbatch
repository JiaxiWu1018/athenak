#!/bin/bash
#SBATCH -J st_s05_measureqa
#SBATCH -p preempt
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014/logs/measureqa.%j.log
set -euo pipefail
export ROOT=/data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014/qa_measurements
export PYTHONPATH=/data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014/scripts
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
mkdir -p "$ROOT/samples" "$ROOT/evidence"
python3 "$PYTHONPATH/qa_measurements.py"
