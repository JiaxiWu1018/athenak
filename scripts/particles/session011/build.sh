#!/bin/bash
# Compilation only: fresh build trees; preserve previous binaries and receipts.
set -euo pipefail
: "${SLURM_JOB_ID:?Allocated compute required}"
test "${SLURM_JOB_PARTITION:-}" = gg
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
module reset
module load nvidia/24.7 openmpi/5.0.5 cmake/4.1.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if (( $# == 0 )); then set -- gi os spin; fi
python3 -B "$ROOT/scripts/particles/session011/clean_build.py" "$@"
