#!/bin/bash
#SBATCH -J jeans8-segment
#SBATCH -A eliasmost
#SBATCH -p mi2104x
#SBATCH -N 3
#SBATCH -n 12
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH -t 04:00:00
#SBATCH --no-requeue
set -euo pipefail
SEGMENT=${1:?segment number}
source /work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/scripts/amd_env.inc
python3 "$ROOT/scripts/workflow.py" permit --segment "$SEGMENT"
RST=$(python3 -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["ROOT"])/"control/latest_checkpoint.json").read_text())["path"])')
export RUN=$ROOT/runs/segment_$(printf '%02d' "$SEGMENT")
test ! -e "$RUN"
mkdir -p "$RUN"
cp "$DECK" "$RUN/canonical.athinput"
{
 date -u; hostname; module list; scontrol show job "$SLURM_JOB_ID"
 sha256sum "$EXE" "$DECK"; echo "restart=$RST"
 cat "$ROOT/control/latest_checkpoint.json"
} > "$RUN/provenance.txt" 2>&1
python3 "$ROOT/scripts/runtime.py" monitor --run "$RUN" &
monitor=$!
cleanup() { kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true; }
trap cleanup EXIT
rc=0
(cd "$RUN"; prun "$ROOT/scripts/amd_rank_wrapper.sh" "$EXE" --kokkos-map-device-id-by=mpi_rank -r "$RST" -i "$DECK" -d "$RUN" -t 03:20:00 time/stop_file="$ROOT/control/REQUEST_STOP") > "$RUN/run.log" 2>&1 || rc=$?
cleanup; trap - EXIT
printf '%s\n' "$rc" > "$RUN/EXIT_CODE"
python3 "$ROOT/scripts/runtime.py" retain --run "$RUN"
exit "$rc"
