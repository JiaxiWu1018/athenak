#!/bin/bash
#SBATCH -J st_s05_pyqa
#SBATCH -p preempt
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014/logs/pyqa.%j.log
set -euo pipefail
export ROOT=/data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014/qa_python
export PYTHONPATH=/data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014/scripts
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$ROOT/samples" "$ROOT/evidence/samples"
python3 - <<'PY'
import numpy as np
from generate import model,density,integrate,octant_cells,generate_one,lhs
from common import ROOT,atomic
p=model(); r=p['riso_over_m']; qa={}
for radial in ('legacy','proper'):
    mass=np.trapz(4*np.pi*r*r*p['D_'+radial],r)
    if abs(mass/p['M0_over_M']-1)>1e-5:raise ValueError('Density/CDF normalization mismatch')
    if np.max(np.diff(p['D_'+radial]))>1e-10*p['D_'+radial].max():raise ValueError('Non-monotone density')
    lo,hi=octant_cells(p['Riso_over_M']); ix=np.linspace(0,len(lo)-1,20).astype(int)
    m16=integrate(p,radial,lo[ix],hi[ix],16);m32=integrate(p,radial,lo[ix],hi[ix],32)
    qa[radial]={'density_mass':float(mass),'probe_cell_difference':float(np.max(abs(m32-m16)))}
    s=generate_one(p,radial,'stratified',1,n=8192)
    a=generate_one(p,radial,'ablation',1,n=8192)
    if not np.array_equal(s[:,1:4],a[:,1:4]) or not np.array_equal(s[:,7],a[:,7]):raise ValueError('Ablation identity')
u=lhs(8192,5,1001)
for d in range(5):
    if not np.array_equal(np.sort((u[:,d]*8192).astype(int)),np.arange(8192)):raise ValueError('LHS marginal strata')
atomic(ROOT/'evidence/QA_OK.json',qa)
print(qa,flush=True)
PY
