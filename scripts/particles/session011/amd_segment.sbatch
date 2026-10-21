#!/bin/bash
#SBATCH -J jeans11-production
#SBATCH -A eliasmost
#SBATCH -p mi2104x
#SBATCH --exclusive
#SBATCH -t 04:00:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=${SESSION_ROOT:?}
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset OMPI_MCA_pml OMPI_MCA_btl UCX_TLS UCX_MEMTYPE_CACHE
export RUN=$ROOT/runs/segment_$(printf '%04d' "${SEGMENT:?}")
test ! -e "$ROOT/control/PAUSED.json"
test ! -e "$ROOT/control/ENDPOINT_75.json"
test ! -e "$RUN"
mkdir -p "$RUN"
cp "$ROOT/inputs/production.athinput" "$RUN/canonical.athinput"
python3 - "$ROOT" "$RUN" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,run=map(Path,sys.argv[1:])
gate=json.loads((root/'control/VALIDATION_PASS.json').read_text())
exe=Path(gate['executable'])
assert hashlib.sha256(exe.read_bytes()).hexdigest()==gate['executable_sha256']
assert hashlib.sha256((run/'canonical.athinput').read_bytes()).hexdigest()==gate['input_sha256']
for relative,expected in gate['runtime_scripts'].items():
    assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==expected,relative
(run/'executable_path').write_text(str(exe)+'\n')
(run/'validation_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
PY
EXE=$(cat "$RUN/executable_path")
START_ARGS=(-i "$RUN/canonical.athinput")
if [[ -e "$ROOT/control/latest_checkpoint.json" ]]; then
  RST=$(python3 -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["SESSION_ROOT"])/"control/latest_checkpoint.json").read_text())["path"])')
  python3 "$ROOT/scripts/runtime.py" checkpoint --path "$RST" > "$RUN/restart_readback.json"
  python3 - "$ROOT/control/latest_checkpoint.json" "$RUN/restart_readback.json" <<'PY'
import json,sys
a,b=[json.load(open(p)) for p in sys.argv[1:]]
assert a['sha256']==b['sha256'] and a['cycle']==b['cycle'] and b['time']<75
PY
  # Preserve the checkpoint's warm state, output counters, and milestones. Only
  # operational overrides below are applied; do not reload the fresh input.
  START_ARGS=(-r "$RST")
else
  test "$SEGMENT" = 0
fi
{
  date -u
  hostname
  module list
  scontrol show job "$SLURM_JOB_ID"
  git -C "$ROOT/code" rev-parse HEAD
  git -C "$ROOT/code" submodule status
  sha256sum "$EXE" "$RUN/canonical.athinput"
  printf 'command: prun amd_rank_wrapper.sh %q ' "$EXE"
  printf '%q ' "${START_ARGS[@]}"
  printf '%s\n' "--kokkos-map-device-id-by=mpi_rank -d $RUN -t 02:00:00 time/tlim=75 time/nlim=-1 mesh_refinement/max_nmb_per_rank=$BLOCK_CAPACITY time/stop_file=$RUN/REQUEST_STOP time/stop_check_every=1"
} > "$RUN/provenance.txt" 2>&1
python3 "$ROOT/scripts/runtime.py" watch --root "$ROOT" --run "$RUN" --nodes "$SLURM_JOB_NUM_NODES" &
monitor=$!
cleanup(){ kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true; }
trap cleanup EXIT
rc=0
date -u +%s > "$RUN/START_EPOCH"
(cd "$RUN"; prun "$ROOT/scripts/amd_rank_wrapper.sh" "$EXE" \
  --kokkos-map-device-id-by=mpi_rank "${START_ARGS[@]}" -d "$RUN" -t 02:00:00 \
  time/tlim=75 time/nlim=-1 mesh_refinement/max_nmb_per_rank="$BLOCK_CAPACITY" \
  time/stop_file="$RUN/REQUEST_STOP" time/stop_check_every=1) > "$RUN/run.log" 2>&1 || rc=$?
date -u +%s > "$RUN/END_EPOCH"
printf '%s\n' "$rc" > "$RUN/EXIT_CODE"
cleanup
trap - EXIT
python3 "$ROOT/scripts/runtime.py" finish --root "$ROOT" --run "$RUN" --returncode "$rc"
exit "$rc"
