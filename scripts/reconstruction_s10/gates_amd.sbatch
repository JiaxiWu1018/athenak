#!/bin/bash
#SBATCH --job-name=st10_gates
#SBATCH --partition=mi2101x
#SBATCH --account=eliasmost
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102/logs/gates.%j.log
set -euo pipefail
root=/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102
repo=$root/src/athenak
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export PYTHONPATH=$repo/scripts/reconstruction_s10
record=$root/evidence/gates_$SLURM_JOB_ID
mkdir -p "$record"
{ hostname; date -u; module list 2>&1; } > "$record/environment.txt"
python3 "$repo/scripts/reconstruction_s10/native_gates.py" --exe "$root/build_cpu/src/athena" --root "$root/tests/cpu_$SLURM_JOB_ID"
# The first GPU gate uses one rank on the single-MI210 support node.
python3 "$repo/scripts/reconstruction_s10/gpu_smoke.py" --exe "$root/build_hip/src/athena" --root "$root/tests/hip_$SLURM_JOB_ID"
cmake -S "$repo" -B "$root/build_fixture_cpu" -DKokkos_ENABLE_SERIAL=ON \
  -DAthena_ENABLE_MPI=ON -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_BUILD_TYPE=Release -DPROBLEM=particles/part_tmunu_test
cmake --build "$root/build_fixture_cpu" -j 16
python3 "$repo/scripts/reconstruction_s10/seam_gates.py" --exe "$root/build_fixture_cpu/src/athena" --root "$root/tests/seams_$SLURM_JOB_ID"
date -u +%FT%TZ > "$record/completed.txt"
