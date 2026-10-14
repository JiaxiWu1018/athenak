#!/bin/bash
#SBATCH -J st_s05_build
#SBATCH -N 1
#SBATCH -p mi2104x
#SBATCH -A eliasmost
#SBATCH -n 4
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --exclude=k003-002,k003-009,k003-010
#SBATCH -t 02:00:00
#SBATCH -o /work1/eliasmost/jiaxiwu/st_reconstruction_s05_20261014/logs/%x.%j.log
source /work1/eliasmost/jiaxiwu/st_reconstruction_s05_20261014/scripts/amd_common.sh
repo="$ROOT/src/athenak"
git -C "$repo" status --short > "$ROOT/evidence/build_source_state.txt"
git -C "$repo" rev-parse HEAD > "$ROOT/evidence/source_commit.txt"
git -C "$repo" submodule status > "$ROOT/evidence/submodules.txt"
cmake -S "$repo" -B "$ROOT/build" \
  -DKokkos_ENABLE_HIP=On -DKokkos_ARCH_AMD_GFX90A=On \
  -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=hipcc \
  -DAthena_ENABLE_MPI=ON -DCMAKE_CXX_FLAGS=-O3 -DCMAKE_C_FLAGS=-O3 \
  -DPROBLEM=particles/relativistic_cluster
cmake --build "$ROOT/build" -j32
mkdir -p "$ROOT/protected"
cp -p "$ROOT/build/src/athena" "$ROOT/protected/athena"
sha256sum "$ROOT/protected/athena" > "$ROOT/evidence/executable_sha256.txt"
cp "$ROOT/build/CMakeCache.txt" "$ROOT/evidence/CMakeCache.txt"
python3 "$ROOT/scripts/generate.py"
printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/evidence/BUILD_OK"
