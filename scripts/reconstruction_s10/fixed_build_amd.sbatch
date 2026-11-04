#!/bin/bash
#SBATCH --job-name=st10_fixed_gate
#SBATCH --partition=mi2101x
#SBATCH --account=eliasmost
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102/logs/fixed_gate.%j.log
set -euo pipefail
root=/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102
repo=$root/src/athenak
record=$root/evidence/fixed_gate_$SLURM_JOB_ID
mkdir -p "$record"
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export PYTHONPATH=$repo/scripts/reconstruction_s10 ST10_MIDPOINT_STENCIL=predicted_fixed
{ hostname; date -u; module list 2>&1; hipcc --version; mpicxx --version; } > "$record/environment.txt"
cp "$root/evidence/FIXED_V1_SOURCE.json" "$record/SOURCE.json"
python3 "$repo/scripts/reconstruction_s10/qualify_fixed.py" verify
python3 -m py_compile "$repo"/scripts/reconstruction_s10/*.py
g++ -std=c++17 -O2 "$repo/scripts/reconstruction_s10/test_midpoint.cpp" -o "$record/test_midpoint"
"$record/test_midpoint" > "$record/manufactured.txt"
cmake -S "$repo" -B "$root/build_cpu" -DKokkos_ENABLE_SERIAL=ON \
  -DAthena_ENABLE_MPI=ON -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_BUILD_TYPE=Release -DPROBLEM=particles/relativistic_cluster
cmake --build "$root/build_cpu" -j 16
cmake -S "$repo" -B "$root/build_hip" -DKokkos_ENABLE_HIP=ON \
  -DKokkos_ARCH_AMD_GFX90A=ON -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=hipcc \
  -DAthena_ENABLE_MPI=ON -DCMAKE_CXX_FLAGS=-O3 -DCMAKE_C_FLAGS=-O3 \
  -DPROBLEM=particles/relativistic_cluster
cmake --build "$root/build_hip" -j 16
python3 "$repo/scripts/reconstruction_s10/native_gates.py" --exe "$root/build_cpu/src/athena" --root "$root/tests/fixed_cpu_$SLURM_JOB_ID"
python3 "$repo/scripts/reconstruction_s10/gpu_smoke.py" --exe "$root/build_hip/src/athena" --root "$root/tests/fixed_hip_$SLURM_JOB_ID"
cmake -S "$repo" -B "$root/build_fixture_cpu" -DKokkos_ENABLE_SERIAL=ON \
  -DAthena_ENABLE_MPI=ON -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_BUILD_TYPE=Release -DPROBLEM=particles/part_tmunu_test
cmake --build "$root/build_fixture_cpu" -j 16
python3 "$repo/scripts/reconstruction_s10/seam_gates.py" --exe "$root/build_fixture_cpu/src/athena" --root "$root/tests/fixed_seams_$SLURM_JOB_ID"
python3 "$repo/scripts/reconstruction_s10/qualify_fixed.py" promote
date -u +%FT%TZ > "$record/completed.txt"
