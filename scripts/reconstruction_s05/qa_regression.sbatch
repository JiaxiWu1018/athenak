#!/bin/bash
#SBATCH -J st_s05_regqa
#SBATCH -p preempt
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gpus=1
#SBATCH --mem=2G
#SBATCH -t 00:10:00
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014/logs/regqa.%j.log
set -euo pipefail
session=/data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014
export ROOT="$session/qa_regression_463500"
export PYTHONPATH="$session/scripts"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$ROOT/evidence"
git -C "$session/src/athenak" rev-parse HEAD > "$ROOT/evidence/source_commit.txt"
python3 - <<'PY'
import os
from preflight import compare_snapshots
from common import ROOT,atomic,sha,require_job
require_job()
a=ROOT/'snapshots/baseline.part.vtk';b=ROOT/'snapshots/default.part.vtk'
r=compare_snapshots(a,b,4096)
r.update(job=os.environ['SLURM_JOB_ID'],snapshot_sha256=[sha(a),sha(b)],original_amd_job=463500)
atomic(ROOT/'evidence/QA_REGRESSION_OK.json',r)
print(r,flush=True)
PY
