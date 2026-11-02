#!/bin/bash
#SBATCH --job-name=st10_build
#SBATCH --partition=mi2101x
#SBATCH --account=eliasmost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102/logs/build.%j.log
set -euo pipefail
root=/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102
repo=$root/src/athenak
record=$root/evidence/build_$SLURM_JOB_ID
mkdir -p "$record"
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
{ hostname; date -u +%FT%TZ; module list 2>&1; hipcc --version; mpicxx --version; } > "$record/environment.txt"
cp "$root/evidence/WORKING_SOURCE.json" "$record/WORKING_SOURCE.json"
g++ -std=c++17 -O2 "$repo/scripts/reconstruction_s10/test_midpoint.cpp" -o "$record/test_midpoint"
"$record/test_midpoint" > "$record/manufactured.txt"
du -s -B1 /work1/eliasmost/jiaxiwu > "$record/whole_user_storage.txt"
cmake -S "$repo" -B "$root/build_cpu" -DKokkos_ENABLE_SERIAL=ON \
  -DAthena_ENABLE_MPI=ON -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_BUILD_TYPE=Release -DPROBLEM=particles/relativistic_cluster
cmake --build "$root/build_cpu" -j 16
sha256sum "$root/build_cpu/src/athena" > "$record/cpu_executable.sha256"
cmake -S "$repo" -B "$root/build_hip" -DKokkos_ENABLE_HIP=ON \
  -DKokkos_ARCH_AMD_GFX90A=ON -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=hipcc \
  -DAthena_ENABLE_MPI=ON -DCMAKE_CXX_FLAGS=-O3 -DCMAKE_C_FLAGS=-O3 \
  -DPROBLEM=particles/relativistic_cluster
cmake --build "$root/build_hip" -j 16
sha256sum "$root/build_hip/src/athena" > "$record/hip_executable.sha256"
date -u +%FT%TZ > "$record/completed.txt"
