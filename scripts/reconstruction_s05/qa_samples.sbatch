#!/bin/bash
#SBATCH -J st_s05_samplesqa
#SBATCH -p preempt
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014/logs/samplesqa.%j.log
set -euo pipefail
session=/data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014
export ROOT="$session/qa_samples"
export PYTHONPATH="$session/scripts"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
mkdir -p "$ROOT/samples" "$ROOT/evidence"
git -C "$session/src/athenak" rev-parse HEAD > "$ROOT/evidence/source_commit.txt"
{ hostname; python3 --version; date -Is; } > "$ROOT/evidence/environment.txt"
python3 "$session/scripts/generate.py"
python3 - <<'PY'
import os,numpy as np
from common import ROOT,CASES,atomic,sha,sample
rows=[]
for case,rad,method,seed in CASES:
    path=ROOT/'samples'/(case+'.bin');_,a=sample(path)
    if not np.array_equal(a[:,0],np.arange(1500000)):raise ValueError('Sample ID set')
    rows.append({'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)})
atomic(ROOT/'evidence/QA_SAMPLES_OK.json',{'job':os.environ['SLURM_JOB_ID'],'manifest':rows,'sample_bytes':sum(r['bytes'] for r in rows),
       'retention':'Retained on Perseus as initialization verification evidence; no deletion authorized'})
PY
