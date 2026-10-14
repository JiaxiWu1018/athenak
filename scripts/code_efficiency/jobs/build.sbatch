#!/bin/bash
#SBATCH --job-name=eff-build-v2
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/jiaxiwu/NRPIC/paper/session_02_code_efficiency/logs/build.%j.log
set -euo pipefail
SESSION=${EFFICIENCY_SESSION:-/data/jiaxiwu/NRPIC/paper/session_02_code_efficiency}
REPO=$SESSION/src/athenak
export PATH=/home/emost/codes/openmpi/bin:/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
REC=$SESSION/provenance/build_$SLURM_JOB_ID
mkdir -p "$REC" "$SESSION/binaries/release" "$SESSION/binaries/profile"
{ date -u; hostname; scontrol show job "$SLURM_JOB_ID"; git -C "$REPO" rev-parse HEAD; git -C "$REPO" status --short; git -C "$REPO" submodule status; nvcc --version; mpicxx --showme; nvidia-smi --query-gpu=index,name,memory.total,uuid,driver_version --format=csv; gcc --version; cmake --version; } > "$REC/environment.txt" 2>&1
if [ -f "$REPO/.git/modules/kokkos/objects/info/alternates" ]; then
  git -C "$REPO/kokkos" repack -a -d > "$REC/kokkos_repack.log" 2>&1
  mv "$REPO/.git/modules/kokkos/objects/info/alternates" "$REC/kokkos_alternates.before"
  git -C "$REPO/kokkos" fsck --no-dangling > "$REC/kokkos_fsck.log" 2>&1
fi
cmake -S "$REPO" -B "$SESSION/build_release" -DKokkos_ENABLE_CUDA=On -DKokkos_ARCH_AMPERE80=On -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER="$REPO/kokkos/bin/nvcc_wrapper" -DAthena_ENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release -DPROBLEM=particles/code_efficiency -DAthena_EFFICIENCY_DEVICE_TIMING=OFF > "$REC/release.configure.log" 2>&1
cmake --build "$SESSION/build_release" -j 4 > "$REC/release.build.log" 2>&1
gmake -C "$SESSION/build_release" -f src/CMakeFiles/athena.dir/build.make src/CMakeFiles/athena.dir/particles/gr_boris.cpp.o -B > "$REC/release.pusher.log" 2>&1
gmake -C "$SESSION/build_release" -f src/CMakeFiles/athena.dir/build.make src/CMakeFiles/athena.dir/build -j 4 -o src/CMakeFiles/athena.dir/flags.make > "$REC/release.link.log" 2>&1
cp "$SESSION/build_release/src/athena" "$SESSION/binaries/release/athena.$SLURM_JOB_ID"
mv "$SESSION/binaries/release/athena.$SLURM_JOB_ID" "$SESSION/binaries/release/athena"
cp "$SESSION/build_release/CMakeCache.txt" "$REC/release.CMakeCache.txt"
cmake -S "$REPO" -B "$SESSION/build_release" -DAthena_EFFICIENCY_DEVICE_TIMING=ON > "$REC/profile.configure.log" 2>&1
# Other compile flags are unchanged. Invoke the direct target; -o is not inherited by
# recursive GNU Make, so the top-level cmake --build would recompile the whole target.
gmake -C "$SESSION/build_release" -f src/CMakeFiles/athena.dir/build.make src/CMakeFiles/athena.dir/particles/gr_boris.cpp.o -B > "$REC/profile_object.log" 2>&1
gmake -C "$SESSION/build_release" -f src/CMakeFiles/athena.dir/build.make src/CMakeFiles/athena.dir/build -j 4 -o src/CMakeFiles/athena.dir/flags.make > "$REC/profile_link.log" 2>&1
cp "$SESSION/build_release/src/athena" "$SESSION/binaries/profile/athena.$SLURM_JOB_ID"
mv "$SESSION/binaries/profile/athena.$SLURM_JOB_ID" "$SESSION/binaries/profile/athena"
cp "$SESSION/build_release/CMakeCache.txt" "$REC/profile.CMakeCache.txt"
sha256sum "$SESSION/binaries/release/athena" "$SESSION/binaries/profile/athena" > "$REC/executables.sha256"
printf '{"job_id":"%s","status":"built"}\n' "$SLURM_JOB_ID" > "$SESSION/provenance/build_receipt.json"
