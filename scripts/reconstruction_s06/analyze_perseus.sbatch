#!/bin/bash
#SBATCH -J st_s06_cpu_analysis
#SBATCH -p preempt
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/06_yc0819_long_stability_20261016/logs/%x.%j.log
set -euo pipefail
export ROOT=/data/jiaxiwu/NRPIC/reconstruction_ST/06_yc0819_long_stability_20261016
export ATHENAK_VIS_PYTHON="$ROOT/src/athenak/vis/python"
export PYTHONPATH="$ROOT/src/athenak/scripts/reconstruction_s06:$ROOT/src/athenak/scripts/reconstruction_s05:$ATHENAK_VIS_PYTHON"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
notify_exit() {
  result=$?
  /home/jiaxiwu/.local/bin/codex queue --thread 01a14666-7bb8-74d1-8819-0037c04e9eb6 \
    --message "ST reconstruction session6 event perseus_analysis_${SLURM_JOB_ID}_exit. Allocated Perseus analysis job $SLURM_JOB_ID exited with status $result. Read /data/jiaxiwu/NRPIC/reconstruction_ST/06_yc0819_long_stability_20261016/HANDOFF.md and evidence/state.json; inspect logs/receipts, acknowledge this event and continue the already-authorized results, visual review and report. This event does not replace the task." || true
  exit "$result"
}
trap notify_exit EXIT
amd_root=/work1/eliasmost/jiaxiwu/st_reconstruction_s06_20261016
previous_root=/data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014
mkdir -p "$ROOT/evidence/jobs/$SLURM_JOB_ID" "$ROOT/evidence/amd" "$ROOT/samples" "$ROOT/reference"
{ hostname; date -Is; python3 --version; git -C "$ROOT/src/athenak" rev-parse HEAD; } > "$ROOT/evidence/jobs/$SLURM_JOB_ID/environment.txt"
git -C "$ROOT/src/athenak" diff --exit-code HEAD -- scripts/reconstruction_s06 scripts/reconstruction_s05
cp "$previous_root/archive/amd_closeout_20261016/samples/proper_stratified_s1.bin" "$ROOT/samples/"
cp "$previous_root/samples/model.npz" "$ROOT/samples/"
cp "$previous_root/reference/proper.npz" "$ROOT/reference/"
ssh hpcfund.amd.com python3 - <<'REMOTE' > "$ROOT/evidence/AMD_SOURCE_INVENTORY.json"
import json,datetime
from pathlib import Path
root=Path('/work1/eliasmost/jiaxiwu/st_reconstruction_s06_20261016')
rows=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
      for role in ('smoke','production') for p in sorted((root/'runs'/role).rglob('*')) if p.is_file()]
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),'files':rows},indent=2))
REMOTE
rsync -a hpcfund.amd.com:"$amd_root/evidence/" "$ROOT/evidence/amd/"
mkdir -p "$ROOT/runs/production" "$ROOT/runs/smoke"
# All particle/field diagnostics, histories/logs, and only the final continuation
# checkpoint; complete AMD originals and every intermediate checkpoint stay put.
rsync -a --exclude='/out/rst/proper_stratified_s1.0000*.rst' hpcfund.amd.com:"$amd_root/runs/production/" "$ROOT/runs/production/"
rsync -a --exclude='rst/' hpcfund.amd.com:"$amd_root/runs/smoke/" "$ROOT/runs/smoke/"
python3 - <<'PYCODE'
from pathlib import Path
from session import *
import run_case as native
inventory=load(ROOT/'evidence/AMD_SOURCE_INVENTORY.json')
copied=[]
for row in inventory['files']:
    path=ROOT/row['path']
    if path.exists():
        if path.stat().st_size!=row['bytes']:raise ValueError('Transfer size differs:'+str(path))
        copied.append({'path':row['path'],'bytes':row['bytes'],'sha256':sha(path)})
summary=load(ROOT/'baseline'/CASE/'summary.json')
if sha(ROOT/'samples'/(CASE+'.bin'))!=summary['initial_sample_sha256']:raise ValueError('Initial sample differs')
state=max(load(ROOT/'runs/production/completion.json')['checkpoints'],key=lambda r:r['time'])
checkpoint=ROOT/'runs/production/out/rst'/Path(state['path']).name
if sha(checkpoint)!=state['sha256']:raise ValueError('Final native checkpoint differs')
native.checkpoint(checkpoint,expected=1000,initial=ROOT/'samples'/(CASE+'.bin'))
atomic(ROOT/'evidence/PERSEUS_ANALYSIS_STAGE.json',{'utc':now(),'job':os.environ['SLURM_JOB_ID'],
       'transport':'rsync over SSH; source byte sizes checked; copied SHA256 manifest retained',
       'source':'AMD native outputs after successful production, retained unchanged',
       'final_checkpoint_source_sha256_verified':state['sha256'],'files':copied})
PYCODE
python3 "$ROOT/scripts/analyze_extension.py"
python3 "$ROOT/scripts/verify.py" --final
python3 "$ROOT/scripts/plot_results.py"
python3 "$ROOT/scripts/finalize_evidence.py"
mkdir -p "$ROOT/plots/pdf_previews"
pdftoppm -f 1 -singlefile -scale-to 1800 -png "$ROOT/plots/f1_long_baseline_histories.pdf" "$ROOT/plots/pdf_previews/f1"
pdftoppm -f 1 -singlefile -scale-to 1800 -png "$ROOT/plots/f2_shell_amplitude_and_drift.pdf" "$ROOT/plots/pdf_previews/f2"
python3 - <<'PYCODE'
from session import *
from PIL import Image
for p in sorted((ROOT/'plots').rglob('*.png')):
    with Image.open(p) as im:im.verify()
atomic(ROOT/'evidence/PERSEUS_ANALYSIS_COMPLETE.json',{'utc':now(),'job':os.environ['SLURM_JOB_ID'],
       'status':'All reductions, independent QA, figures, PDF previews and PNG decoding passed; visual/scientific review still required'})
PYCODE
