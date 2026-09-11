#!/bin/bash
# CPU (Serial) Kokkos build WITH bounds checking, for isolating device-side memory
# faults in the Plummer pgen.  Not for production.
#SBATCH -J plm2_build_dbg
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -o /work1/eliasmost/jiaxiwu/plummer_s02_20260910/logs/plm_build_dbg.%j.log

set -euo pipefail
ROOT=/work1/eliasmost/jiaxiwu/plummer_s02_20260910
repo=$ROOT/src/athenak
build=$ROOT/build_cpu_debug

hostname; date -Is
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3
git -C "$repo" log --oneline -1

rm -rf "$build"
cmake -S "$repo" -B "$build" \
  -DKokkos_ENABLE_SERIAL=On \
  -DKokkos_ENABLE_DEBUG=On \
  -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=On \
  -DKokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK=On \
  -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=mpicxx \
  -DAthena_ENABLE_MPI=ON -DCMAKE_CXX_FLAGS="-O1 -g" -DCMAKE_C_FLAGS="-O1 -g" \
  -DPROBLEM=particles/nr_pic_plummer
cmake --build "$build" -j 16
sha256sum "$build/src/athena"
date -Is
echo "CPU DEBUG BUILD OK"
