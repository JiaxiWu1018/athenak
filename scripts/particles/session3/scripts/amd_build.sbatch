#!/bin/bash
#SBATCH --job-name=pl_s3_build
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/work1/eliasmost/jiaxiwu/plummer_s03_20261005/logs/build_%j.log
set -euo pipefail
root=${PLUMMER_S3_ROOT:-/work1/eliasmost/jiaxiwu/plummer_s03_20261005}
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
repo=$root/src/athenak
commit=$(git -C "$repo" rev-parse HEAD)
build=$root/builds/$commit
mkdir -p "$build"
hostname
date -u
module list 2>&1
hipcc --version
mpicxx --version
cmake --version
git -C "$repo" status --porcelain
git -C "$repo" submodule status
test -z "$(git -C "$repo" status --porcelain)"
cmake -S "$repo" -B "$build" -DKokkos_ENABLE_HIP=On -DKokkos_ARCH_AMD_GFX90A=On \
  -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=hipcc -DAthena_ENABLE_MPI=ON \
  -DCMAKE_CXX_FLAGS=-O3 -DCMAKE_C_FLAGS=-O3 -DPROBLEM=particles/nr_pic_plummer
cmake --build "$build" -j 16
sha256sum "$build/src/athena" > "$build/exe.sha256"
printf '%s\n' "$commit" > "$build/git-head.txt"
printf '%s\n' "$build" > "$root/state/build_path.txt"
cat "$build/exe.sha256"
date -u
