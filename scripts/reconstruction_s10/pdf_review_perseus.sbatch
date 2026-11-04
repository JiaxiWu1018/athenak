#!/bin/bash
#SBATCH --job-name=st10_pdf_review
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/data/jiaxiwu/NRPIC/reconstruction_ST/10_rk3_predictor_20261102/logs/pdf_review.%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 - "${1:?pass the copied final figures directory}" <<'PY'
import hashlib,json,os,re,socket,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path

job=os.environ['SLURM_JOB_ID']
root=Path('/data/jiaxiwu/NRPIC/reconstruction_ST/10_rk3_predictor_20261102')
source=Path(sys.argv[1]).resolve()
assert source.is_relative_to(root),source
out=root/'evidence'/f'pdf_review_{job}'
out.mkdir(exist_ok=False)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
manifest=source/'FIGURES.json'
entries=json.loads(manifest.read_text())['files']
expected={'constraint_evolution','constraint_ratios','particle_number_scaling',
          'temporal_order','temporal_differences','short_controls'}
assert {e['path'] for e in entries}=={n+ext for n in expected for ext in ('.png','.pdf')}
assert len(entries)==12
for entry in entries:
    path=source/entry['path']
    assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],path
records=[]
for name in sorted(expected):
    pdf=source/(name+'.pdf')
    info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    assert re.search(r'^Pages:\s+1\s*$',info,re.M),info
    subprocess.run(['pdftoppm','-scale-to','1800','-png','-singlefile',str(pdf),str(out/name)],check=True)
    preview=out/(name+'.png')
    assert preview.stat().st_size>0
    records.append(dict(pdf=str(pdf),pdf_sha256=sha(pdf),preview=preview.name,
                        preview_sha256=sha(preview),preview_bytes=preview.stat().st_size))
version=subprocess.run(['pdftoppm','-v'],text=True,capture_output=True,check=True)
repo=root/'src/athenak'
script=repo/'scripts/reconstruction_s10/pdf_review_perseus.sbatch'
record=dict(utc=datetime.now(timezone.utc).isoformat(),job_id=job,host=socket.gethostname(),
            source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
            script=str(script),script_sha256=sha(script),
            source_manifest=str(manifest),source_manifest_sha256=sha(manifest),
            renderer=(version.stderr or version.stdout).splitlines()[0],
            source_hashes_passed=True,rendered_pdf_count=len(records),
            manual_visual_review_complete=False,records=records)
(out/'RENDER.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
PY
