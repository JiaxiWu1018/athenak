#!/bin/bash
#SBATCH --job-name=st10_cpu_build
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/data/jiaxiwu/NRPIC/reconstruction_ST/10_rk3_predictor_20261102/logs/build_cpu.%j.log
set -euo pipefail
session=/data/jiaxiwu/NRPIC/reconstruction_ST/10_rk3_predictor_20261102
repo=$session/src/athenak
build=$session/build_cpu
mkdir -p "$session/evidence/build_cpu_$SLURM_JOB_ID"
record=$session/evidence/build_cpu_$SLURM_JOB_ID
hostname > "$record/host.txt"
date -u +%FT%TZ > "$record/start.txt"
git -C "$repo" rev-parse HEAD > "$record/source.txt"
git -C "$repo" diff --binary > "$record/working.patch"
git -C "$repo" submodule status > "$record/submodules.txt"
cmake --version > "$record/cmake.txt"
mpicxx --version > "$record/compiler.txt"
cmake -S "$repo" -B "$build" -DKokkos_ENABLE_SERIAL=ON -DAthena_ENABLE_MPI=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/mpicxx -DCMAKE_C_COMPILER=/usr/bin/mpicc \
  -DCMAKE_BUILD_TYPE=Release -DPROBLEM=particles/relativistic_cluster
cmake --build "$build" -j "$SLURM_CPUS_PER_TASK"
sha256sum "$build/src/athena" > "$record/executable.sha256"
date -u +%FT%TZ > "$record/completed.txt"
