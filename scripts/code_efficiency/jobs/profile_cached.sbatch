#!/bin/bash
#SBATCH --job-name=eff-profile-build
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/data/jiaxiwu/NRPIC/paper/session_02_code_efficiency/logs/build.%j.log
set -euo pipefail
SESSION=${EFFICIENCY_SESSION:-/data/jiaxiwu/NRPIC/paper/session_02_code_efficiency}
BUILD=$SESSION/build_release
REC=$SESSION/provenance/build_$SLURM_JOB_ID
mkdir -p "$REC"
export PATH=/home/emost/codes/openmpi/bin:/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
{ hostname; date -u; scontrol show job "$SLURM_JOB_ID"; git -C "$SESSION/src/athenak" status --short; } > "$REC/environment.txt"
cp "$BUILD/src/CMakeFiles/athena.dir/flags.make" "$REC/flags.make"
# CMake's Makefiles share one flags dependency. Only the gr_boris custom define changed;
# the release core objects use exactly the same CXX_DEFINES, CXX_INCLUDES and CXX_FLAGS.
# Force this object, then rebuild missing objects/link while treating that common file old.
gmake -C "$BUILD" -f src/CMakeFiles/athena.dir/build.make src/CMakeFiles/athena.dir/particles/gr_boris.cpp.o -B > "$REC/profile_object.log" 2>&1
gmake -C "$BUILD" -f src/CMakeFiles/athena.dir/build.make src/CMakeFiles/athena.dir/build -j 2 -o src/CMakeFiles/athena.dir/flags.make > "$REC/profile_link.log" 2>&1
cp "$BUILD/src/athena" "$SESSION/binaries/profile/athena"
sha256sum "$SESSION/binaries/release/athena" "$SESSION/binaries/profile/athena" > "$REC/executables.sha256"
printf '{"job_id":"%s","status":"built","release":"%s","profile":"%s"}\n' "$SLURM_JOB_ID" "$SESSION/binaries/release/athena" "$SESSION/binaries/profile/athena" > "$SESSION/provenance/build_receipt.json"
