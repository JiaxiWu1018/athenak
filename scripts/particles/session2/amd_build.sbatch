#!/bin/bash
# ROCm-6.4.1 MI210 build of project/Plummer-cluster (NRPIC Plummer session 1).
# Recipe inherited from ST_migration/session_06 scripts/amd_build.sbatch; only the
# ROOT and PROBLEM change.  Do NOT substitute the default ROCm 7.x: it builds but
# faults at runtime in particle stress-energy deposition (AGENTS.md).
#SBATCH -J plm2_build
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -o /work1/eliasmost/jiaxiwu/plummer_s02_20260910/logs/plm_build.%j.log

set -euo pipefail
ROOT=/work1/eliasmost/jiaxiwu/plummer_s02_20260910
repo=$ROOT/src/athenak
build=$ROOT/build_rocm641

hostname; date -Is
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
module list 2>&1
hipcc --version | head -3
mpicxx --version | head -1
cmake --version | head -1
git -C "$repo" log --oneline -1
git -C "$repo" status --porcelain | head
git -C "$repo" submodule status

rm -rf "$build"
cmake -S "$repo" -B "$build" \
  -DKokkos_ENABLE_HIP=On -DKokkos_ARCH_AMD_GFX90A=On \
  -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=hipcc \
  -DAthena_ENABLE_MPI=ON -DCMAKE_CXX_FLAGS=-O3 -DCMAKE_C_FLAGS=-O3 \
  -DPROBLEM=particles/nr_pic_plummer
cmake --build "$build" -j 16

grep -E "PROBLEM:STRING|Kokkos_ENABLE_HIP:|Kokkos_ARCH_AMD_GFX90A:|Athena_ENABLE_MPI:" "$build/CMakeCache.txt"
sha256sum "$build/src/athena" | tee "$build/exe.sha256"
sha256sum "$repo/src/pgen/particles/nr_pic_plummer.cpp" | tee "$build/pgen.sha256"
sha256sum "$repo/src/pgen/particles/plummer_profile.hpp" | tee "$build/profile.sha256"
sha256sum "$repo/src/z4c/z4c_calcrhs.cpp" | tee "$build/z4c_calcrhs.sha256"
git -C "$repo" rev-parse HEAD | tee "$build/git-head.txt"
date -Is
echo "AMD BUILD OK"
