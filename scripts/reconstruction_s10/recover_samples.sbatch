#!/bin/bash
#SBATCH --job-name=st10_transfer
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/data/jiaxiwu/NRPIC/reconstruction_ST/10_rk3_predictor_20261102/logs/transfer.%j.log
set -euo pipefail
session=/data/jiaxiwu/NRPIC/reconstruction_ST/10_rk3_predictor_20261102
stage=$session/staging/session7
mkdir -p "$stage" "$session/evidence/transfer_$SLURM_JOB_ID"
record=$session/evidence/transfer_$SLURM_JOB_ID
{ hostname; date -u; scontrol show job "$SLURM_JOB_ID"; } > "$record/environment.txt"
for n in 1 2 4 8; do
  printf '%s\n' "samples/mc_${n}m_s1.bin" "evidence/samples/mc_${n}m_s1.json" "reduced/mc_${n}m_s1/health.csv" "reduced/mc_${n}m_s1/global_constraints.csv" "reduced/mc_${n}m_s1/shells.csv"
done > "$record/files.txt"
printf '%s\n' samples/model.npz protected/athena >> "$record/files.txt"
rsync -a --files-from="$record/files.txt" jiaxiwu@anta.caltech.edu:/data3/jiaxiwu/NRPIC/reconstruction_ST/07_mc_particle_convergence_20261019/ "$stage/"
python3 - "$stage" "$record" <<'PY'
import sys,json,hashlib,os
from pathlib import Path
root,record=map(Path,sys.argv[1:]);rows=[]
for p in sorted(root.rglob('*')):
    if not p.is_file():continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    row=dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=h.hexdigest());rows.append(row)
    if p.suffix=='.bin' and p.parent.name=='samples':
        expected=json.loads((root/'evidence/samples'/p.with_suffix('.json').name).read_text())
        assert row['sha256']==expected['sha256'] and row['bytes']==expected['bytes'],row
    if row['path']=='protected/athena':
        assert row['sha256']=='5fb4fb9f2131984854b3dd2eef455e8acc6808152e800698fcf5349e5bc35482',row
(record/'TRANSFER_MANIFEST.json').write_text(json.dumps(dict(job=os.environ['SLURM_JOB_ID'],files=rows),indent=2)+'\n')
PY
rsync -a "$stage/" hpcfund.amd.com:/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102/reference/session7/
rsync -a "$record/TRANSFER_MANIFEST.json" hpcfund.amd.com:/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102/evidence/
date -u +%FT%TZ > "$record/completed.txt"
