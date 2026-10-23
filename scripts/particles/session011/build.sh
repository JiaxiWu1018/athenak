#!/bin/bash
# Compilation only. Never executes Athena or submits jobs.
set -euo pipefail
: "${SLURM_JOB_ID:?Allocated compute required}"
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
SESSION=$(dirname "$ROOT")
test "$(git -C "$ROOT/kokkos" rev-parse HEAD)" = 6739bc623081648af9e752b616d9671527922cbf
module reset
module load nvidia/24.7 openmpi/5.0.5 cmake/4.1.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$SESSION/build" "$SESSION/evidence/build"
for KIND in ${1:-gi os spin}; do
 case "$KIND" in
 gi) PROBLEM=particles/gi_cluster ;;
 os) PROBLEM=particles/nr_pic_os ;;
 spin) PROBLEM=z4c/kerr_schild ;;
 *) exit 2 ;;
 esac
 BDIR="$SESSION/build/$KIND"
 { date -u; hostname; module list; git -C "$ROOT" rev-parse HEAD; nvcc --version; mpicxx --showme; } > "$SESSION/evidence/build/$KIND.provenance.txt" 2>&1
 cmake -S "$ROOT" -B "$BDIR" \
 -DCMAKE_PROJECT_INCLUDE="$ROOT/scripts/particles/session011/kokkos_4_7_2_arm_sve_compat.cmake" \
 -DKokkos_ENABLE_CUDA=On -DKokkos_ARCH_HOPPER90=On -DKokkos_ARCH_ARMV9_GRACE=On \
 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER="$ROOT/kokkos/bin/nvcc_wrapper" \
 -DAthena_ENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release -DPROBLEM="$PROBLEM" \
 > "$SESSION/evidence/build/$KIND.configure.log" 2>&1
 cmake --build "$BDIR" -j "${S11_BUILD_JOBS:-16}" > "$SESSION/evidence/build/$KIND.build.log" 2>&1
 python3 "$ROOT/scripts/particles/session011/prepare.py" build-receipt "$KIND"
done
