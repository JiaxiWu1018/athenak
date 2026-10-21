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
  if [ ! -f "$ROOT/evidence/SAMPLES_READY.json" ]; then python3 "$ROOT/scripts/prepare.py"; fi
  export FROZEN_WORK="$ROOT/tests/frozen_ah_$SLURM_JOB_ID"
  mkdir "$FROZEN_WORK"
  python3 - <<'PYCODE'
from common import *
atomic(ROOT/'evidence/active.json',dict(work=os.environ['FROZEN_WORK'],phase='frozen_ah',job=job()))
PYCODE
  prun "$ROOT/protected/athena" --kokkos-map-device-id-by=mpi_rank -i "$ROOT/inputs/frozen_ah.athinput" -d "$FROZEN_WORK" fastflow/verbose=true "problem/cluster_t0_dump=$FROZEN_WORK/initial_particles.bin" > "$record/frozen_ah.log" 2>&1
  test -f "$FROZEN_WORK/FROZEN_AH_OK"
  python3 - <<'PYCODE'
from common import *
atomic(ROOT/'evidence/BUILD_AND_FROZEN_AH_OK.json',dict(utc=now(),job=job(),executable=receipt(ROOT/'protected/athena'),frozen_check=receipt(Path(os.environ['FROZEN_WORK'])/'FROZEN_AH_OK')))
PYCODE
 ;;
 checks)
  test -f "$ROOT/evidence/BUILD_AND_FROZEN_AH_OK.json"
  python3 "$ROOT/scripts/native_checks.py"
 ;;
 initialize)
  test -f "$ROOT/evidence/NATIVE_CHECKS_OK.json"
  for run_case in yc0283 yc0500 yc0605 yc0819 yc0900; do
    python3 "$ROOT/scripts/run_segment.py" "$run_case" --initialize --wall-seconds 900
  done
 ;;
 run)
  test -f "$ROOT/evidence/NATIVE_CHECKS_OK.json"
  shift
  python3 "$ROOT/scripts/run_segment.py" "$@"
 ;;
 postprocess)
  if [ ! -f "$ROOT/evidence/PLOTTING_RUNTIME.json" ]; then
    python3 -m pip install --target "$ROOT/dependencies/site" numpy==1.26.4 matplotlib==3.9.4 imageio-ffmpeg==0.6.0 pypdfium2==4.30.0 > "$record/pip.log" 2>&1
    python3 - <<'PYCODE'
from common import *
import numpy,matplotlib,imageio_ffmpeg,pypdfium2
atomic(ROOT/'evidence/PLOTTING_RUNTIME.json',dict(job=job(),utc=now(),numpy=numpy.__version__,matplotlib=matplotlib.__version__,imageio_ffmpeg=imageio_ffmpeg.__version__,ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()))
cmd(['python3','-m','pip','freeze','--path',ROOT/'dependencies/site'],stdout=(ROOT/'evidence/plotting_requirements.txt').open('w'))
PYCODE
  fi
  for run_case in yc0283 yc0500 yc0605 yc0819 yc0900; do python3 "$ROOT/scripts/reduce.py" "$run_case"; done
  python3 "$ROOT/scripts/plots.py"
  python3 "$ROOT/scripts/movies.py"
 ;;
 audit)
  shift
  python3 "$ROOT/scripts/reduce.py" "$@"
 ;;
 *) echo "Unimplemented job action $1" >&2; exit 2 ;;
esac
