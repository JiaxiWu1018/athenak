#!/bin/bash
#SBATCH --job-name=pl_s3_init_fig
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005/logs/init_fig_%j.log
set -euo pipefail
session=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005
export MPLBACKEND=Agg
hostname
date -u
python3 "$session/analysis/plot_initialization.py" "$session" "$session/runs/cpu_t0_live_11064"
sha256sum "$session/figures/initialization_fidelity.pdf" "$session/figures/initialization_fidelity.png"
