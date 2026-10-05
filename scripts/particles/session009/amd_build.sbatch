#!/bin/bash
#SBATCH -J jeans9-build
#SBATCH -A eliasmost
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:30:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=/work1/eliasmost/jiaxiwu/gi_s009_amd_20261005
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
mkdir -p "$ROOT/build"
{
 date -u; hostname; echo "job=$SLURM_JOB_ID"; module list
 git -C "$ROOT/athenak" rev-parse HEAD
 git -C "$ROOT/athenak" submodule status
 git -C "$ROOT/athenak" status --short
 cmake --version; hipcc --version; mpicxx --showme
} > "$ROOT/evidence/build_provenance.txt" 2>&1
cmake -S "$ROOT/athenak" -B "$ROOT/build" -DKokkos_ENABLE_HIP=On -DKokkos_ARCH_AMD_GFX90A=On -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=hipcc -DAthena_ENABLE_MPI=ON -DCMAKE_CXX_FLAGS=-O3 -DCMAKE_C_FLAGS=-O3 -DPROBLEM=particles/gi_cluster > "$ROOT/logs/configure.log" 2>&1
cmake --build "$ROOT/build" -j 16 > "$ROOT/logs/make.log" 2>&1
sha256sum "$ROOT/build/src/athena" > "$ROOT/evidence/executable.sha256"
printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/control/build_job.txt"

cmake -S "$ROOT/athenak" -B "$ROOT/build_wave" -DKokkos_ENABLE_HIP=On -DKokkos_ARCH_AMD_GFX90A=On -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=hipcc -DAthena_ENABLE_MPI=ON -DCMAKE_CXX_FLAGS=-O3 -DCMAKE_C_FLAGS=-O3 -DPROBLEM=built_in_pgens > "$ROOT/logs/configure_wave.log" 2>&1
cmake --build "$ROOT/build_wave" -j 16 > "$ROOT/logs/make_wave.log" 2>&1
sha256sum "$ROOT/build_wave/src/athena" > "$ROOT/evidence/wave_executable.sha256"
mpicc -O2 "$ROOT/scripts/mpi_probe.c" -o "$ROOT/build/mpi_probe"
sha256sum "$ROOT/build/mpi_probe" > "$ROOT/evidence/mpi_probe.sha256"
