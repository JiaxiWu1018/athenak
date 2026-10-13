#!/bin/bash
# Run only inside an allocated Vista compute job, after plan-mode discussion/approval.
set -euo pipefail
: "${SLURM_JOB_ID:?native build requires allocated compute}"
: "${SCRATCH:?Vista SCRATCH required}"
HANDOFF=${1:?absolute extracted handoff path}
WORK_ROOT=${2:?unique absolute build/work root beneath SCRATCH}
case "$WORK_ROOT/" in "$SCRATCH/"*) ;; *) echo 'Work root must be beneath SCRATCH' >&2;exit 2;;esac
SRC=$WORK_ROOT/source/athenak
BUILD=$WORK_ROOT/build_cuda_gi
mkdir -p "$WORK_ROOT/source" "$WORK_ROOT/evidence"
readarray -t PINS < <(python3 - "$HANDOFF/PROVENANCE.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]));print(p['source_commit']);print(p['kokkos_commit'])
PY
)
test ! -e "$SRC" || { echo 'Refuse existing source root; choose an isolated work root' >&2;exit 2; }
git clone "$HANDOFF/source/athenak.bundle" "$SRC"
git -C "$SRC" checkout --detach "${PINS[0]}"
git clone "$HANDOFF/source/kokkos.bundle" "$SRC/kokkos"
git -C "$SRC/kokkos" checkout --detach "${PINS[1]}"
module reset
# Historical successful Session007 stack; verify availability before changing any version.
module load nvidia/24.7 openmpi/5.0.5 cmake/4.1.1
{
 date -u;hostname;scontrol show job "$SLURM_JOB_ID";module list
 nvcc --version;gcc --version;mpicxx --showme;python3 --version
 git -C "$SRC" rev-parse HEAD;git -C "$SRC" submodule status
} > "$WORK_ROOT/evidence/native_build_provenance.txt" 2>&1
cmake -S "$SRC" -B "$BUILD" \
 -DCMAKE_PROJECT_INCLUDE="$HANDOFF/scripts/kokkos_4_7_2_arm_sve_compat.cmake" \
 -DKokkos_ENABLE_CUDA=On -DKokkos_ARCH_HOPPER90=On -DKokkos_ARCH_ARMV9_GRACE=On \
 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER="$SRC/kokkos/bin/nvcc_wrapper" \
 -DAthena_ENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release -DPROBLEM=particles/gi_cluster \
 > "$WORK_ROOT/evidence/native_build.configure.log" 2>&1
cmake --build "$BUILD" -j "${SLURM_CPUS_PER_TASK:-2}" > "$WORK_ROOT/evidence/native_build.make.log" 2>&1
sha256sum "$BUILD/src/athena" > "$WORK_ROOT/evidence/native_executable.sha256"
