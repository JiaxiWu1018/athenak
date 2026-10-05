#!/bin/bash
#SBATCH --job-name=pl_s3_profile
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005/logs/profile_%j.log
set -euo pipefail
session=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005
cd "$session"
hostname
date -u
g++ --version
python3 --version
git -C src/athenak rev-parse HEAD
git -C src/athenak submodule status
g++ -std=c++17 -O2 -I src/athenak/src/pgen/particles analysis/profile_probe.cpp -o initial_data/profile_probe
sha256sum initial_data/profile_probe src/athenak/src/pgen/particles/plummer_isotropic_profile.hpp
./initial_data/profile_probe > initial_data/cpp_profile.csv
python3 analysis/validate_profile.py initial_data/cpp_profile.csv evidence/profile_agreement.json
