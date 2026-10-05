#!/bin/bash
#SBATCH -A eliasmost
#SBATCH -p mi2104x
#SBATCH -N 3
#SBATCH -n 12
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --exclude=k002-[005-006],k003-[001-006,008-010],k005-[006,008-010]
#SBATCH -t 04:00:00
#SBATCH --no-requeue
set -euo pipefail
NUMBER=${1:?continuation number}
source /work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/scripts/amd_env.inc
RECOVERY=$ROOT/scripts/recovery_20261005/recovery.py
python3 "$RECOVERY" permit --number "$NUMBER"
RST=$(python3 -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["ROOT"])/"control/latest_checkpoint.json").read_text())["path"])')
export RUN=$ROOT/runs/segment_$(printf '%02d' "$((NUMBER+2))")_recovery_20261005
test ! -e "$RUN"
mkdir -p "$RUN"
cp "$DECK" "$RUN/canonical.athinput"
{
 date -u; hostname; module list; scontrol show job "$SLURM_JOB_ID"
 sha256sum "$EXE" "$DECK"; cat "$ROOT/control/latest_checkpoint.json"
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
