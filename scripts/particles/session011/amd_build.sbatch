#!/bin/bash
#SBATCH -J jeans11-build
#SBATCH -A eliasmost
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=${SESSION_ROOT:?}
SOURCE_COMMIT=${SOURCE_COMMIT:?}
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TEST_ROOT=$ROOT/evidence/build_${SLURM_JOB_ID}
mkdir -p "$TEST_ROOT"
test "$(git -C "$ROOT/code" rev-parse HEAD)" = "$SOURCE_COMMIT"
python3 "$ROOT/code/scripts/particles/session011/test_runtime.py" > "$TEST_ROOT/controller_checks.log" 2>&1
{
  date -u
  hostname
  scontrol show job "$SLURM_JOB_ID"
  module list
  git -C "$ROOT/code" rev-parse HEAD
  git -C "$ROOT/code" submodule status
  git -C "$ROOT/code" status --short
  hipcc --version
  mpicxx --showme
  rocm-smi --showproductname --showmeminfo vram
} > "$TEST_ROOT/provenance.txt" 2>&1
cmake -S "$ROOT/code" -B "$ROOT/build" \
  -DKokkos_ENABLE_HIP=On -DKokkos_ARCH_AMD_GFX90A=On \
  -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=hipcc -DAthena_ENABLE_MPI=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-O3 -DCMAKE_C_FLAGS=-O3 \
  -DPROBLEM=z4c/z4c_one_puncture > "$TEST_ROOT/configure.log" 2>&1
cmake --build "$ROOT/build" -j 16 > "$TEST_ROOT/build.log" 2>&1
cp "$ROOT/build/src/athena" "$TEST_ROOT/athena_frozen"
cp "$ROOT/inputs/frozen_ah.athinput" "$TEST_ROOT/input.athinput"
mkdir "$TEST_ROOT/fresh"
cd "$TEST_ROOT/fresh"
"$TEST_ROOT/athena_frozen" -i "$TEST_ROOT/input.athinput" -d "$TEST_ROOT/fresh" \
  > "$TEST_ROOT/fresh.log" 2>&1
grep 'FROZEN_AH_CHECKS_PASS.*evolution_advances=0' "$TEST_ROOT/fresh.log"
python3 "$ROOT/code/scripts/particles/session011/check_frozen_outputs.py" "$TEST_ROOT" > "$TEST_ROOT/readback.log" 2>&1
RST=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoints"][-1]["path"])' "$TEST_ROOT/readback.json")
mkdir "$TEST_ROOT/restarted"
"$TEST_ROOT/athena_frozen" -r "$RST" -d "$TEST_ROOT/restarted" \
  diagnostic_checks/frozen_ah_searches=6 time/nlim=0 > "$TEST_ROOT/restarted.log" 2>&1
grep 'FROZEN_AH_CHECKS_PASS.*evolution_advances=0' "$TEST_ROOT/restarted.log"
cmake -S "$ROOT/code" -B "$ROOT/build" -DPROBLEM=particles/gi_cluster \
  > "$TEST_ROOT/configure_production.log" 2>&1
cmake --build "$ROOT/build" -j 16 > "$TEST_ROOT/build_production.log" 2>&1
cp "$ROOT/build/src/athena" "$TEST_ROOT/athena_production"
sha256sum "$TEST_ROOT/athena_frozen" "$TEST_ROOT/athena_production" "$TEST_ROOT/input.athinput" > "$TEST_ROOT/SHA256SUMS"
printf '{"job_id":"%s","source":"%s","frozen_check":"pass","production_build":"pass"}\n' \
  "$SLURM_JOB_ID" "$SOURCE_COMMIT" > "$TEST_ROOT/RESULT.json"
