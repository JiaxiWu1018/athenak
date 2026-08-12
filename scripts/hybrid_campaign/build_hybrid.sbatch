#!/usr/bin/env bash
#SBATCH --job-name=hyb_build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
# Build inside the same module environment the solver runs in (a login-node
# build mixes ROCm 7.2 device libs into the 6.4.1 hipcc and aborts at the
# first kernel launch -- root-caused in the 2026-08-01 campaign).
set -euo pipefail
module purge
module load autotools cmake/3.25.2 prun/2.3 rocm/6.4.1 gnu12/12.2.0 openmpi4/4.1.8
echo "ROCM_PATH=${ROCM_PATH}"
REPO=/work1/eliasmost/jiaxiwu/athenak_hybrid_20260802
BUILD=${REPO}/build_hybrid_gfx90a_rocm6_4_1
rm -rf "${BUILD}"; mkdir -p "${BUILD}"; cd "${BUILD}"
cmake -DPROBLEM=nr_pic_homogeneous_cluster \
      -DAthena_ENABLE_MPI=ON \
      -DKokkos_ENABLE_HIP=On \
      -DKokkos_ARCH_AMD_GFX90A=On \
      -DCMAKE_CXX_COMPILER=/opt/rocm-6.4.1/bin/hipcc \
      -DCMAKE_CXX_FLAGS=-O3 \
      -DCMAKE_C_FLAGS=-O3 \
      .. > cmake_configure.log 2>&1
make -j32 > build.log 2>&1
ls -la src/athena && sha256sum src/athena
echo BUILD_OK
