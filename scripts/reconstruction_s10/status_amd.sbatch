#!/bin/bash
#SBATCH --job-name=st10_status
#SBATCH --partition=mi2101x
#SBATCH --account=eliasmost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102/logs/status.%j.log
set -euo pipefail
root=/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102
export PYTHONPATH=$root/src/athenak/scripts/reconstruction_s10
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
python3 - <<'PY'
import os, socket
from pathlib import Path
import numpy, matplotlib
from campaign import ROOT, resource_snapshot
from state_io import write_json
from plot import plt
r=resource_snapshot();r.update(host=socket.gethostname(),numpy=numpy.__version__,matplotlib=matplotlib.__version__,retention='All raw/test/build/reference data retained; no cleanup authorized')
write_json(ROOT/f'evidence/STATUS_{os.environ["SLURM_JOB_ID"]}.json',r)
# This renders actual manufactured-test errors, not scientific ST results.
a=numpy.loadtxt(ROOT/'evidence/build_495009/manufactured.txt',max_rows=5)
fig,ax=plt.subplots(figsize=(6,4));h=1/a[:,0]
ax.loglog(h,a[:,1],'o-',label='Metric variable U')
ax.loglog(h,a[:,2],'s-',label='Particle variable Z')
ax.loglog(h,a[-1,1]*(h/h[-1])**2,':',color='.3',label='Second-order reference')
ax.set_xlabel('Time step h');ax.set_ylabel('Absolute error at t = 1');ax.set_title('Manufactured coupled-system test');ax.legend(frameon=False);ax.grid(alpha=.2)
fig.tight_layout();out=ROOT/'evidence/manufactured_figure';out.mkdir(exist_ok=False)
for ext in ('png','pdf'):fig.savefig(out/('manufactured_order.'+ext),dpi=180)
PY
