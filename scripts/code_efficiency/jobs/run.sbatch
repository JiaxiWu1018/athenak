#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
set -euo pipefail
ulimit -c 0
SESSION=${EFFICIENCY_SESSION:-/data/jiaxiwu/NRPIC/paper/session_02_code_efficiency}
: "${CASE:?}" "${KIND:=throughput}" "${TRACE:=0}"
export PATH=/home/emost/codes/openmpi/bin:/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
args=()
[ "$TRACE" != 1 ] || args+=(--trace)
status=0
python3 "$SESSION/scripts/run_case.py" "$CASE" --kind "$KIND" "${args[@]}" || status=$?
python3 "$SESSION/scripts/advance.py" || true
python3 "$SESSION/scripts/analyze.py" || printf 'REPORT GENERATION FAILED; see job log\n'
exit "$status"
