#!/bin/bash
# Native build only; never submits jobs.
set -euo pipefail
: "${SLURM_JOB_ID:?allocated compute required}"
HANDOFF=$(cd "$(dirname "$0")/.." && pwd)
WORK=$HANDOFF/runtime_s10/build
mkdir -p "$WORK/evidence"
readarray -t PINS < <(python3 - "$HANDOFF/PROVENANCE.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]));print(p['source_commit']);print(p['kokkos_commit'])
PY
)
SRC=$WORK/source/athenak
BUILD=$WORK/build_cuda_gi
if [ ! -d "$SRC/.git" ]; then
  test ! -e "$SRC"
  mkdir -p "$WORK/source"
  git clone "$HANDOFF/source/athenak.bundle" "$SRC"
  git -C "$SRC" checkout --detach "${PINS[0]}"
  git clone "$HANDOFF/source/kokkos.bundle" "$SRC/kokkos"
  git -C "$SRC/kokkos" checkout --detach "${PINS[1]}"
fi
test "$(git -C "$SRC" rev-parse HEAD)" = "${PINS[0]}"
test "$(git -C "$SRC/kokkos" rev-parse HEAD)" = "${PINS[1]}"
test -z "$(git -C "$SRC" status --porcelain --untracked-files=no)"
test -z "$(git -C "$SRC/kokkos" status --porcelain --untracked-files=no)"
module reset
module load nvidia/24.7 openmpi/5.0.5 cmake/4.1.1
{
  date -u
  hostname
  uname -m
  scontrol show job "$SLURM_JOB_ID"
  module list
  nvcc --version
  gcc --version
  mpicxx --showme
  ucx_info -v
  git -C "$SRC" rev-parse HEAD
  git -C "$SRC" submodule status
} > "$WORK/evidence/native_build_provenance.txt" 2>&1
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_PROJECT_INCLUDE="$HANDOFF/scripts/kokkos_4_7_2_arm_sve_compat.cmake" \
  -DKokkos_ENABLE_CUDA=On -DKokkos_ARCH_HOPPER90=On -DKokkos_ARCH_ARMV9_GRACE=On \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER="$SRC/kokkos/bin/nvcc_wrapper" \
  -DAthena_ENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release -DPROBLEM=particles/gi_cluster \
  > "$WORK/evidence/configure.log" 2>&1
cmake --build "$BUILD" -j 16 > "$WORK/evidence/build.log" 2>&1
python3 - "$HANDOFF" "$WORK" <<'PY'
import hashlib,json,os,pathlib,platform,subprocess,sys,time
root,work=map(pathlib.Path,sys.argv[1:]);pins=json.loads((root/'PROVENANCE.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
exe=work/'build_cuda_gi/src/athena'
files=[exe,work/'evidence/native_build_provenance.txt',work/'evidence/configure.log',work/'evidence/build.log',work/'build_cuda_gi/CMakeCache.txt']
receipt=dict(source_commit=pins['source_commit'],kokkos_commit=pins['kokkos_commit'],host=platform.node(),architecture=platform.machine(),job_id=os.environ['SLURM_JOB_ID'],created_epoch=time.time(),executable=str(exe),executable_sha256=sha(exe),files=[dict(path=str(p),sha256=sha(p)) for p in files],build_script_sha256=sha(root/'launch_s10/build.sh'))
p=work/'BUILD_VERIFIED.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(receipt,indent=2)+'\n');os.replace(tmp,p)
print('Native build verified:',exe,receipt['executable_sha256'],flush=True)
PY
