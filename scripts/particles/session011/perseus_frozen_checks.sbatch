#!/bin/bash
#SBATCH -J jeans11-frozen
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 02:00:00
set -euo pipefail
SESSION=/data/jiaxiwu/NRPIC/GI_in_cluster/session_011_amd_ah_orbit_20261021
export PATH=/home/emost/codes/openmpi/bin:/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TEST_ROOT=$SESSION/evidence/frozen_${SLURM_JOB_ID}
mkdir -p "$TEST_ROOT"
git clone --no-hardlinks "$SESSION/code" "$TEST_ROOT/source" > "$TEST_ROOT/clone.log" 2>&1
git -C "$TEST_ROOT/source" checkout --detach "$SOURCE_COMMIT" >> "$TEST_ROOT/clone.log" 2>&1
git clone --no-hardlinks "$SESSION/code/kokkos" "$TEST_ROOT/source/kokkos" >> "$TEST_ROOT/clone.log" 2>&1
cp "$SESSION/inputs/frozen_ah.athinput" "$TEST_ROOT/input.athinput"
{
  date -u
  hostname
  scontrol show job "$SLURM_JOB_ID"
  git -C "$TEST_ROOT/source" rev-parse HEAD
  git -C "$TEST_ROOT/source/kokkos" rev-parse HEAD
  nvcc --version
  mpicxx --showme
  nvidia-smi --query-gpu=name,memory.total --format=csv
} > "$TEST_ROOT/provenance.txt" 2>&1
cmake -S "$TEST_ROOT/source" -B "$TEST_ROOT/build" \
  -DKokkos_ENABLE_CUDA=On -DKokkos_ARCH_AMPERE80=On \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER="$TEST_ROOT/source/kokkos/bin/nvcc_wrapper" \
  -DAthena_ENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release \
  -DPROBLEM=z4c/z4c_one_puncture > "$TEST_ROOT/configure.log" 2>&1
cmake --build "$TEST_ROOT/build" -j "$SLURM_CPUS_PER_TASK" > "$TEST_ROOT/build.log" 2>&1
sha256sum "$TEST_ROOT/build/src/athena" "$TEST_ROOT/input.athinput" > "$TEST_ROOT/SHA256SUMS"
mkdir "$TEST_ROOT/fresh"
cd "$TEST_ROOT/fresh"
"$TEST_ROOT/build/src/athena" -i "$TEST_ROOT/input.athinput" -d "$TEST_ROOT/fresh" \
  > "$TEST_ROOT/fresh.log" 2>&1
rg 'FROZEN_AH_CHECKS_PASS.*evolution_advances=0' "$TEST_ROOT/fresh.log"
printf '{"job_id":"%s","source":"%s","stage":"fresh_frozen_pass"}\n' \
  "$SLURM_JOB_ID" "$SOURCE_COMMIT" > "$TEST_ROOT/RESULT.json"
