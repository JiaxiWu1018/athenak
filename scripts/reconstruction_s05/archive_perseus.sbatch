#!/bin/bash
#SBATCH -J st_s05_archive_perseus
#SBATCH -p preempt
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014/logs/archive_perseus.%j.log
set -euo pipefail
export ROOT=/data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/src/athenak/scripts/reconstruction_s05:$ROOT/src/athenak/vis/python:${PYTHONPATH:-}"
export ATHENAK_VIS_PYTHON="$ROOT/src/athenak/vis/python"
mkdir -p "$ROOT/evidence/jobs/$SLURM_JOB_ID"
{ hostname; date -Is; python3 --version; } > "$ROOT/evidence/jobs/$SLURM_JOB_ID/archive_environment.txt"
git -C "$ROOT/src/athenak" rev-parse HEAD > "$ROOT/evidence/jobs/$SLURM_JOB_ID/archive_source.txt"
python3 "$ROOT/src/athenak/scripts/reconstruction_s05/archive_closeout.py" "${1:-transfer}"
