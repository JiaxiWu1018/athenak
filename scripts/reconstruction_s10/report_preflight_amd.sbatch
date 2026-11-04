#!/bin/bash
#SBATCH --job-name=st10_report_preflight
#SBATCH --partition=mi2101x
#SBATCH --account=eliasmost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102/logs/report_preflight.%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg ST10_SCHEME=fixed_fast_v1
root=/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102
export PYTHONPATH=$root/src/athenak/scripts/reconstruction_s10
python3 - <<'PY'
import ast,os
from pathlib import Path
from campaign import ROOT,resource_snapshot
from analyze import completed
from final_report import segment_sources
from state_io import require_job,sha,write_json
require_job()
scripts=ROOT/'src/athenak/scripts/reconstruction_s10'
source=[]
for name in ('analyze.py','plot.py','final_report.py'):
    p=scripts/name;ast.parse(p.read_text());source.append(dict(path=str(p),sha256=sha(p)))
records=[]
for n in (1,4,8):
    r=completed(f'fixed_fast_v1_{n}m_h010')[-1]
    files=segment_sources(r)
    assert all(p.is_file() for p in files)
    records.append(dict(case=r['case'],target=r['target'],sources=[dict(path=str(p),sha256=sha(p)) for p in files]))
r=completed('fixed_fast_v1_4m_h010')[0]
assert r['target']==100
checkpoint=r['checkpoint'];assert sha(checkpoint['path'])==checkpoint['sha256']
write_json(ROOT/f'evidence/REPORT_PREFLIGHT_{os.environ["SLURM_JOB_ID"]}.json',dict(passed=True,scope='Current completed segment provenance, reporting syntax and4M100M checkpoint hash. This does not replace final validation of the future500M data and short controls.',sources=source,records=records,checkpoint=checkpoint,resources=resource_snapshot()))
PY
