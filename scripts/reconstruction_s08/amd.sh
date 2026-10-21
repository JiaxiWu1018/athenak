#!/bin/bash
set -euo pipefail
export ROOT=/work1/eliasmost/jiaxiwu/st_reconstruction_s08_production_20261021
export PYTHONPATH="$ROOT/scripts:$ROOT/src/athenak/scripts/reconstruction_s05:$ROOT/src/athenak/vis/python:$ROOT/dependencies/site"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TZ=UTC
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
record="$ROOT/evidence/jobs/$SLURM_JOB_ID"
mkdir -p "$record"
{ hostname; date -Is; module list 2>&1; hipcc --version; python3 --version; } > "$record/environment.txt"
cp "$ROOT/evidence/SOURCE_PIN.json" "$record/SOURCE_PIN.json"
python3 "$ROOT/scripts/watchdog.py" > "$record/watchdog.log" 2>&1 &
watchdog_pid=$!
trap 'kill "$watchdog_pid" 2>/dev/null || true' EXIT
case "$1" in
 build)
  sha256sum "$ROOT/source.tar" "$ROOT/kokkos.tar" > "$record/source_archives.sha256"
  cmake -S "$ROOT/src/athenak" -B "$ROOT/build" -DKokkos_ENABLE_HIP=On -DKokkos_ARCH_AMD_GFX90A=On -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=hipcc -DAthena_ENABLE_MPI=ON -DCMAKE_CXX_FLAGS=-O3 -DCMAKE_C_FLAGS=-O3 -DPROBLEM=particles/relativistic_cluster
  cmake --build "$ROOT/build" -j32
  mkdir -p "$ROOT/protected"
  cp -p "$ROOT/build/src/athena" "$ROOT/protected/athena"
  cp "$ROOT/build/CMakeCache.txt" "$record/CMakeCache.txt"
  sha256sum "$ROOT/protected/athena" > "$ROOT/evidence/executable.sha256"
  python3 "$ROOT/scripts/prepare.py"
  mkdir -p "$ROOT/tests/frozen_ah"
  python3 - <<'PYCODE'
from common import *
atomic(ROOT/'evidence/active.json',dict(work=str(ROOT/'tests/frozen_ah'),phase='frozen_ah',job=job()))
PYCODE
  prun "$ROOT/protected/athena" --kokkos-map-device-id-by=mpi_rank -i "$ROOT/inputs/frozen_ah.athinput" -d "$ROOT/tests/frozen_ah" > "$record/frozen_ah.log" 2>&1
  test -f "$ROOT/tests/frozen_ah/FROZEN_AH_OK"
  python3 - <<'PYCODE'
from common import *
atomic(ROOT/'evidence/BUILD_AND_FROZEN_AH_OK.json',dict(utc=now(),job=job(),executable=receipt(ROOT/'protected/athena'),frozen_check=receipt(ROOT/'tests/frozen_ah/FROZEN_AH_OK')))
PYCODE
 ;;
 *) echo "Unimplemented job action $1" >&2; exit 2 ;;
esac
