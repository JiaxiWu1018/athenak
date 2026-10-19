#!/bin/bash
# Host-only diagnostic build on the already allocated GG node; no simulation/submission.
set -euo pipefail
: "${SLURM_JOB_ID:?compute allocation required}"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=$ROOT/runtime_s10/build/source/athenak
BUILD=$ROOT/runtime_s10/build_parser_check
mkdir -p "$BUILD/evidence"
module reset
module load nvidia/24.7 openmpi/5.0.5 cmake/4.1.1
{
  date -u
  hostname
  module list
  gcc --version
  git -C "$SRC" rev-parse HEAD
  git -C "$SRC/kokkos" rev-parse HEAD
} > "$BUILD/evidence/provenance.txt" 2>&1
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_PROJECT_INCLUDE="$ROOT/scripts/kokkos_4_7_2_arm_sve_compat.cmake" \
  -DKokkos_ENABLE_CUDA=OFF -DKokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_ARMV9_GRACE=ON \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DAthena_ENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release -DPROBLEM=particles/gi_cluster \
  > "$BUILD/evidence/configure.log" 2>&1
cmake --build "$BUILD" -j 16 > "$BUILD/evidence/build.log" 2>&1
sha256sum "$BUILD/src/athena"
