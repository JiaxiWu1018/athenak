#!/bin/bash
#SBATCH --job-name=pl_s3_cpu_fix
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005/logs/cpu_fix_%j.log
set -euo pipefail
session=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005
git -C "$session/src/athenak" rev-parse HEAD
cmake --build "$session/initial_data/build_cpu" -j 2
sha256sum "$session/initial_data/build_cpu/src/athena"
