#!/bin/bash
#SBATCH --job-name=pl_s3_preflight_fig
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005/logs/preflight_fig_%j.log
set -euo pipefail
session=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005
export MPLBACKEND=Agg
python3 "$session/analysis/plot_preflight.py" "$session"
sha256sum "$session/figures/preflight_health.pdf" "$session/figures/preflight_health.png"
