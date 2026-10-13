#!/bin/bash
set -euo pipefail
CONFIG=${1:?usage: vista_build.sbatch SITE_CONFIG.json}
HANDOFF=$(python3 -c 'import json, os, sys; print(os.path.expanduser(os.path.expandvars(json.load(open(sys.argv[1]))["handoff_dir"])))' "$CONFIG")
SCRIPT_DIR=$HANDOFF/scripts
WF=$SCRIPT_DIR/workflow.py
test -f "$WF"
get() { python3 "$WF" --config "$CONFIG" config-get "$1"; }
test "$(get handoff_dir)" = "$HANDOFF"
WORK_ROOT=$(get work_root)
EXPECT_HEAD=$(get source_commit)
EXPECT_KOKKOS=$(get kokkos_commit)
MODULES=$(get modules)
BUILD_JOBS=$(get build_jobs)
readarray -t SOURCE_BUNDLE_CONFIG < <(python3 - "$CONFIG" <<'PY'
import json, sys
cfg=json.load(open(sys.argv[1]))
print(cfg.get("source_bundle_relpath", "source/athenak.bundle"))
print(cfg.get("source_bundle_sha256", ""))
PY
)
SOURCE_BUNDLE=$HANDOFF/${SOURCE_BUNDLE_CONFIG[0]}
EXPECTED_SOURCE_BUNDLE_SHA256=${SOURCE_BUNDLE_CONFIG[1]}
SRC=$WORK_ROOT/source/athenak
BUILD=$WORK_ROOT/build_cuda_gi
DETAILS=$WORK_ROOT/control/build_details.${SLURM_JOB_ID:-manual}.txt
CMAKE_COMPAT=$SCRIPT_DIR/kokkos_4_7_2_arm_sve_compat.cmake
mkdir -p "$WORK_ROOT/source" "$WORK_ROOT/control"
test -f "$CMAKE_COMPAT"
test -f "$SOURCE_BUNDLE"
if [ -n "$EXPECTED_SOURCE_BUNDLE_SHA256" ]; then
  test "$(sha256sum "$SOURCE_BUNDLE" | awk '{print $1}')" = \
    "$EXPECTED_SOURCE_BUNDLE_SHA256"
fi
module reset >/dev/null 2>&1
module load $MODULES
if [ ! -d "$SRC/.git" ]; then
  test ! -e "$SRC" || { echo "FATAL: conflicting non-git source path $SRC"; exit 3; }
  git clone "$SOURCE_BUNDLE" "$SRC"
  git -C "$SRC" checkout --detach "$EXPECT_HEAD"
  git clone "$HANDOFF/source/kokkos.bundle" "$SRC/kokkos"
  git -C "$SRC/kokkos" checkout --detach "$EXPECT_KOKKOS"
fi
test "$(git -C "$SRC" rev-parse HEAD)" = "$EXPECT_HEAD"
test "$(git -C "$SRC/kokkos" rev-parse HEAD)" = "$EXPECT_KOKKOS"
test -z "$(git -C "$SRC" status --short)"
{
  echo "host=$(hostname)"
  echo "job_id=${SLURM_JOB_ID:-none}"
  echo "started=$(date -Is)"
  module list 2>&1
  nvcc --version
  gcc --version | head -3
  mpicxx --showme
  echo "source_commit=$EXPECT_HEAD"
  echo "source_bundle=$SOURCE_BUNDLE"
  sha256sum "$SOURCE_BUNDLE"
  echo "kokkos_commit=$EXPECT_KOKKOS"
  echo "cmake_flags=CUDA,HOPPER90,ARMV9_GRACE,MPI,Release,particles/gi_cluster"
  echo "cmake_compat=$CMAKE_COMPAT"
  sha256sum "$CMAKE_COMPAT"
} > "$DETAILS"
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_PROJECT_INCLUDE="$CMAKE_COMPAT" \
  -DKokkos_ENABLE_CUDA=On \
  -DKokkos_ARCH_HOPPER90=On \
  -DKokkos_ARCH_ARMV9_GRACE=On \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER="$SRC/kokkos/bin/nvcc_wrapper" \
  -DAthena_ENABLE_MPI=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DPROBLEM=particles/gi_cluster >> "$DETAILS" 2>&1
cmake --build "$BUILD" -j "$BUILD_JOBS" >> "$DETAILS" 2>&1
sha256sum "$BUILD/src/athena" | tee "$WORK_ROOT/control/executable.sha256" >> "$DETAILS"
echo "completed=$(date -Is)" >> "$DETAILS"
python3 "$WF" --config "$CONFIG" mark build pass --details "$DETAILS"
