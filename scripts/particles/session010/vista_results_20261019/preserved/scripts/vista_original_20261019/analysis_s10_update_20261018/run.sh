#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$BASE_DIR/runtime_s10/analysis/matplotlib_cache"
exec "$BASE_DIR/runtime_s10/analysis_env/bin/python" -u "$SCRIPT_DIR/run.py" "$@"
