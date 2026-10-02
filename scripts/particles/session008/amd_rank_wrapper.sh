#!/bin/bash
set -uo pipefail
sampler=
if [[ ${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-0}} -eq 0 ]]; then
 python3 "$ROOT/scripts/runtime.py" vram --root "$ROOT" --run "$RUN" &
 sampler=$!
fi
cleanup() { if [[ -n "$sampler" ]]; then kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true; fi; }
trap cleanup EXIT
rc=0
"$@" || rc=$?
exit "$rc"
