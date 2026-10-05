#!/bin/bash
#SBATCH --job-name=pl_s3_contract
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005/logs/contract_%j.log
set -euo pipefail
session=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005
python3 "$session/analysis/test_health_contract.py"
python3 -m compileall -q "$session/analysis" "$session/scripts"
for script in "$session"/scripts/*.sh "$session"/scripts/*.sbatch; do bash -n "$script"; done
