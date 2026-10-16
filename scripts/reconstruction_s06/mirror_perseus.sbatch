#!/bin/bash
#SBATCH -J st_s06_mirror
#SBATCH -p preempt
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/06_yc0819_long_stability_20261016/logs/%x.%j.log
set -euo pipefail
export ROOT=/data/jiaxiwu/NRPIC/reconstruction_ST/06_yc0819_long_stability_20261016
export PYTHONPATH="$ROOT/src/athenak/scripts/reconstruction_s06:$ROOT/src/athenak/scripts/reconstruction_s05:$ROOT/src/athenak/vis/python"
amd_root=/work1/eliasmost/jiaxiwu/st_reconstruction_s06_20261016
mkdir -p "$ROOT/evidence/amd" "$ROOT/checkpoints" "$ROOT/runs"
rsync -a hpcfund.amd.com:"$amd_root/evidence/" "$ROOT/evidence/amd/"
rsync -a hpcfund.amd.com:"$amd_root/reduced/" "$ROOT/reduced/"
rsync -a hpcfund.amd.com:"$amd_root/plots/" "$ROOT/plots/"
rsync -a hpcfund.amd.com:"$amd_root/reports/" "$ROOT/reports/"
rsync -a --include='*/' --include='*.json' --include='*.jsonl' --include='*.log' --include='*.hst' --exclude='*' \
  hpcfund.amd.com:"$amd_root/runs/" "$ROOT/runs/"
rsync -a hpcfund.amd.com:"$amd_root/runs/production/out/rst/proper_stratified_s1.00010.rst" "$ROOT/checkpoints/"
python3 - <<'PYCODE'
import json,os,subprocess
from pathlib import Path
from session import ROOT,atomic,sha,now
import run_case as native
completion=json.loads((ROOT/'runs/production/completion.json').read_text())
final=max(completion['checkpoints'],key=lambda r:r['time'])
path=ROOT/'checkpoints'/Path(final['path']).name
if sha(path)!=final['sha256']:raise ValueError('Final checkpoint mirror checksum differs')
# No sample input is required for this repeat payload check; AMD checked per-ID masses.
native.checkpoint(path,expected=1000)
checks=[]
for row in json.loads((ROOT/'evidence/amd/DELIVERY_MANIFEST.json').read_text())['files']:
    p=ROOT/row['path']
    if sha(p)!=row['sha256']:raise ValueError('Compact delivery checksum differs:'+str(p))
for row in json.loads((ROOT/'evidence/amd/DESCRIPTIVE_METRICS.json').read_text())['figures']:
    p=ROOT/row['path']
    if sha(p)!=row['sha256']:raise ValueError('Figure mirror checksum differs')
    checks.append(row)
manifest=json.loads((ROOT/'evidence/amd/OUTPUT_MANIFEST.json').read_text())
for row in manifest['files']:
    p=ROOT/row['path']
    if p.exists() and sha(p)!=row['sha256']:raise ValueError('Mirrored run metadata differs:'+str(p))
atomic(ROOT/'evidence/MIRROR_OK.json',{'utc':now(),'job':os.environ['SLURM_JOB_ID'],'final_checkpoint':final,'figures':checks,
       'retention':'AMD raw retained; Perseus final checkpoint and review products verified; no deletion'})
PYCODE
mkdir -p "$ROOT/plots/pdf_previews"
pdftoppm -f 1 -singlefile -scale-to 1800 -png "$ROOT/plots/f1_long_baseline_histories.pdf" "$ROOT/plots/pdf_previews/f1"
pdftoppm -f 1 -singlefile -scale-to 1800 -png "$ROOT/plots/f2_shell_amplitude_and_drift.pdf" "$ROOT/plots/pdf_previews/f2"
