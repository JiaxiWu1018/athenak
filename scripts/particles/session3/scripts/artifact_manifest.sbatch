#!/bin/bash
#SBATCH --job-name=pl_s3_artifacts
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005/logs/artifacts_%j.log
set -euo pipefail
session=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005
cd "$session"
sha256sum -c evidence/reduced_transfer.sha256
python3 - <<'PY'
import csv,hashlib,json,os,socket,subprocess
from pathlib import Path
root=Path.cwd();paths=set(root.glob('*.md'))|set(root.glob('*.pdf'))
for directory in ('inputs','reports','figures'):
    paths.update(p for p in (root/directory).rglob('*') if p.is_file())
with (root/'evidence/review_artifacts.tsv').open('w') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['relative_path','bytes','sha256'])
    for p in sorted(paths):w.writerow([str(p.relative_to(root)),p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest()])
provenance=dict(host=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],
    commit=subprocess.check_output(['git','-C','src/athenak','rev-parse','HEAD'],text=True).strip(),
    scientific_status='initialization/preflight only; production stopped for budget discussion',
    raw_retention='AMD and Perseus retained; Anta full; no deletion authorized')
(root/'evidence/artifact_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
PY
