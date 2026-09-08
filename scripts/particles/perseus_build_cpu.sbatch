#!/bin/bash
# Perseus serial/OpenMP CPU build of project/Plummer-cluster (fast compile-error shakeout
# and a small-N validation binary).  Heavy work must not run on the login node.
#SBATCH -J plm_build_cpu
#SBATCH -o /data/jiaxiwu/NRPIC/Plummer-cluster/logs/perseus_build_cpu.%j.log
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -p debug

set -euo pipefail
repo=/data/jiaxiwu/NRPIC/Plummer-cluster/code
build=/data/jiaxiwu/NRPIC/Plummer-cluster/build/cpu
export PATH=/home/emost/codes/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:${LD_LIBRARY_PATH:-}

hostname; date -Is
mpicxx --version | head -1
cmake --version | head -1
git -C "$repo" log --oneline -1
git -C "$repo" status --porcelain | head -20

rm -rf "$build"
cmake -S "$repo" -B "$build" \
  -DKokkos_ENABLE_OPENMP=On \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_C_COMPILER=/usr/bin/cc \
  -DAthena_ENABLE_MPI=ON \
  -DCMAKE_CXX_FLAGS="-O2" \
  -DCMAKE_C_FLAGS="-O2" \
  -DPROBLEM=particles/nr_pic_plummer
cmake --build "$build" -j 8 2>&1 | tail -80

grep -E "PROBLEM:STRING|Kokkos_ENABLE_OPENMP:|Athena_ENABLE_MPI:" "$build/CMakeCache.txt"
sha256sum "$build/src/athena" | tee "$build/exe.sha256"
git -C "$repo" rev-parse HEAD > "$build/git-head.txt"
date -Is
echo "PERSEUS CPU BUILD OK"
