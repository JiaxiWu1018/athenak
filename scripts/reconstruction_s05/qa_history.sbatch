#!/bin/bash
#SBATCH -J st_s05_histqa
#SBATCH -p preempt
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gpus=1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014/logs/histqa.%j.log
set -euo pipefail
session=/data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014
export ROOT="$session/qa_history"
export PYTHONPATH="$session/scripts"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$ROOT/evidence"
python3 - <<'PY'
import os,numpy as np
from common import ROOT,atomic
from analyze import history
# Software fixture only: the next step is ordinary while the completed final step
# is tiny. Such a row must be masked; genuine initial constraints must remain usable.
a=np.zeros((3,19));a[:,0]=[0,2,300];a[:,1]=.2
a[:,[2,8,14,16]]=1;a[:,[3,9]]=4;a[:,[4,10]]=9
a[:,15]=.85;a[:,17]=.995;a[:,18]=[0,.125,1e-12]
path=ROOT/'completed_step_fixture.hst';np.savetxt(path,a)
rows=history(path)
if [r['constraint_usable'] for r in rows]!=[True,True,False]:raise ValueError('Terminal-step mask uses the next dt')
if any(r['H_rms_r10']!=2 or r['M_rms_r16']!=3 for r in rows):raise ValueError('Norm column indexing changed')
atomic(ROOT/'evidence/QA_HISTORY_OK.json',{'job':os.environ['SLURM_JOB_ID'],'source':'software fixture; no simulation result',
       'initial_kept':True,'normal_step_kept':True,'tiny_terminal_step_masked':True})
PY
