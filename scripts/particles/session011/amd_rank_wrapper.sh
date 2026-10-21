#!/bin/bash
set -uo pipefail
monitor=
if [[ ${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-0}} == 0 ]]; then
  python3 "$SESSION_ROOT/scripts/runtime.py" telemetry --run "$RUN" &
  monitor=$!
fi
cleanup() {
  if [[ -n $monitor ]]; then kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true; fi
}
trap cleanup EXIT
"$@"
rc=$?
exit "$rc"
