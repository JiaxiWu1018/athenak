#!/usr/bin/env bash
# Complete machine checks and review sheets; never submits or changes production.
set -euo pipefail
S10_ANALYSIS_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
S10_ANALYSIS_BASE=$(cd -- "$S10_ANALYSIS_SCRIPT_DIR/.." && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$S10_ANALYSIS_BASE/runtime_s10/analysis/matplotlib_cache"
test "${SLURM_JOB_PARTITION:-}" = gg
while ! test -f "$S10_ANALYSIS_BASE/runtime_s10/analysis/through_segment002_20261018/SEALED.json"; do
    if (( $(date +%s) > ${SLURM_JOB_END_TIME:-0} - 1200 )); then
        echo 'Insufficient allocated time to begin the render; resume on GG.' >&2
        exit 75
    fi
    sleep 5
done
bash "$S10_ANALYSIS_SCRIPT_DIR/run.sh"
"$S10_ANALYSIS_BASE/runtime_s10/analysis_env/bin/python" "$S10_ANALYSIS_SCRIPT_DIR/qa.py"
echo 'Machine verification complete. Inspect the actual PDF and decoded movie review sheets before finalizing.'
