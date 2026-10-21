#!/bin/bash
#SBATCH -J jeans11-final-check
#SBATCH -A eliasmost
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=${SESSION_ROOT:?}
BASE_CHECK=${BASE_CHECK:?}
SOURCE_COMMIT=${SOURCE_COMMIT:?}
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset OMPI_MCA_pml OMPI_MCA_btl UCX_TLS UCX_MEMTYPE_CACHE
TEST_ROOT=$ROOT/evidence/final_${SLURM_JOB_ID}
mkdir -p "$TEST_ROOT"
test "$(git -C "$ROOT/code" rev-parse HEAD)" = "$SOURCE_COMMIT"
test -f "$BASE_CHECK/RESULT.json"
{
  date -u
  hostname
  module list
  scontrol show job "$SLURM_JOB_ID"
  git -C "$ROOT/code" rev-parse HEAD
  git -C "$ROOT/code" submodule status
  hipcc --version
  mpicxx --showme
} > "$TEST_ROOT/provenance.txt" 2>&1
cmake -S "$ROOT/code" -B "$ROOT/build" -DPROBLEM=particles/gi_cluster > "$TEST_ROOT/configure.log" 2>&1
cmake --build "$ROOT/build" -j 16 > "$TEST_ROOT/build.log" 2>&1
cp "$ROOT/build/src/athena" "$TEST_ROOT/athena_production"
cp "$ROOT/inputs/production.athinput" "$TEST_ROOT/production.athinput"
python3 "$ROOT/code/scripts/particles/session011/test_runtime.py" > "$TEST_ROOT/controller_checks.log" 2>&1
RST=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoints"][-1]["path"])' "$BASE_CHECK/readback.json")
export RUN=$TEST_ROOT/restarted
mkdir "$RUN"
ln -s "$BASE_CHECK/fresh" "$TEST_ROOT/fresh"
prun "$ROOT/scripts/amd_rank_wrapper.sh" "$TEST_ROOT/athena_production" \
  --kokkos-map-device-id-by=mpi_rank -r "$RST" -d "$RUN" \
  diagnostic_checks/frozen_ah_searches=6 time/nlim=0 particles/feedback=true \
  > "$TEST_ROOT/restarted.log" 2>&1
grep 'FROZEN_AH_CHECKS_PASS.*evolution_advances=0' "$TEST_ROOT/restarted.log"
grep 'Number of parallel ranks = 4' "$TEST_ROOT/restarted.log"
if grep -q 'conservation VIOLATED' "$TEST_ROOT/restarted.log"; then exit 1; fi
python3 "$ROOT/code/scripts/particles/session011/check_restart.py" "$TEST_ROOT" > "$TEST_ROOT/restart_comparison.log" 2>&1
python3 - "$TEST_ROOT" <<'PY'
import json,math,re,struct,sys
from pathlib import Path
r=Path(sys.argv[1])
frames=list((r/'restarted/trk').glob('*.trk2'))
assert len(frames)==1
data=frames[0].read_bytes();header=data[:512].rstrip(b'\0').decode()
assert 'count=8 ' in header and len(data)==512+72*8
records=list(struct.iter_unpack('=q8d',data[512:]))
assert sorted(row[0] for row in records)==list(range(8))
assert all(math.isfinite(x) for row in records for x in row[1:])
telemetry=list((r/'restarted').glob('telemetry_*.json'))
assert telemetry and len(json.loads(telemetry[0].read_text())['devices'])==4
(r/'mpi_readback.json').write_text(json.dumps(dict(ranks=4,exact_ids=True,finite=True,telemetry_devices=4),indent=2)+'\n')
PY
sha256sum "$TEST_ROOT/athena_production" "$TEST_ROOT/production.athinput" > "$TEST_ROOT/SHA256SUMS"
printf '{"job_id":"%s","source":"%s","mpi_ranks":4,"evolution_advances":0,"status":"pass"}\n' \
  "$SLURM_JOB_ID" "$SOURCE_COMMIT" > "$TEST_ROOT/RESULT.json"
