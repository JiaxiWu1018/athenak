#!/bin/bash
#SBATCH --job-name=pl_s3_cpu_build
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005/logs/cpu_build_%j.log
set -euo pipefail
session=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005
cd "$session"
hostname
date -u
git -C src/athenak diff --stat
cmake -S src/athenak -B initial_data/build_cpu -DKokkos_ENABLE_SERIAL=ON -DAthena_ENABLE_MPI=ON -DPROBLEM=particles/nr_pic_plummer -DCMAKE_CXX_FLAGS=-O2
cmake --build initial_data/build_cpu -j 2
sha256sum initial_data/build_cpu/src/athena
