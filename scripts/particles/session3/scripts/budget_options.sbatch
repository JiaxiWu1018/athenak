#!/bin/bash
#SBATCH --job-name=pl_s3_budget_options
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005/logs/budget_options_%j.log
set -euo pipefail
session=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005
python3 "$session/analysis/budget_options.py" "$session"
