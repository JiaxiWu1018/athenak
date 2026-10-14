#!/bin/bash
#SBATCH -J st_s05_refqa
#SBATCH -p preempt
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014/logs/refqa.%j.log
set -euo pipefail
session=/data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014
export ROOT="$session/qa_reference_v2"
export PYTHONPATH="$session/scripts"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
mkdir -p "$ROOT/evidence"
git -C "$session/src/athenak" rev-parse HEAD > "$ROOT/evidence/source_commit.txt"
sha256sum "$session/scripts/reference.py" "$session/scripts/analyze.py" > "$ROOT/evidence/script_sha256.txt"
python3 - <<'PY'
import json,numpy as np
from generate import model
from reference import interpolation_matrix,tensor_interpolate,quadrature,ensure
from common import ROOT,atomic
n=3;order=4
b=interpolation_matrix(order,n)
x=np.arange(-n-4,n+4)+.5
field=x[:,None,None]**2+2*x[None,:,None]**3+.5*x[None,None,:]**4
u=quadrature(order)[0]
q=(np.arange(-n,n)[:,None]+u).ravel()
expected=q[:,None,None]**2+2*q[None,:,None]**3+.5*q[None,None,:]**4
actual=tensor_interpolate(field,b)
if not np.allclose(actual,expected,rtol=1e-12,atol=1e-10):raise ValueError('Tensor interpolation does not reproduce polynomials')
from analyze import window_measurements
t=np.r_[0,np.arange(2,302,2)+.017];q0=np.arange(1,10,dtype=float)
rows=window_measurements(t,q0[None,:]*(1+.001*t[:,None]),300)
if any(abs(r['detrended_half_range_fraction'])>1e-12 or abs(r['slope_fraction_per_M']-.001)>1e-14 for r in rows):
    raise ValueError('Linear drift leaked into detrended amplitude')
ensure(model())
atomic(ROOT/'evidence/QA_REFERENCE_OK.json',{rad:json.loads((ROOT/'reference'/(rad+'.json')).read_text()) for rad in ('legacy','proper')})
PY
