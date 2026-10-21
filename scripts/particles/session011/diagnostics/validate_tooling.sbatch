#!/bin/bash
#SBATCH -A eliasmost
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:20:00
#SBATCH -J s11-energy-QA
#SBATCH --no-requeue
set -euo pipefail
ROOT=/work1/eliasmost/jiaxiwu/gi_s011_amd_20261021
FIX=$ROOT/evidence/nonfinite_20261021
OUT=$FIX/qa_$SLURM_JOB_ID
mkdir "$OUT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=$ROOT/.mpl_cache PRODUCT_QA=$OUT/figures
python3 "$FIX/tooling/test_runtime.py" > "$OUT/runtime_tests.log" 2>&1
python3 "$FIX/tooling/test_products.py" > "$OUT/product_tests.log" 2>&1
python3 - "$ROOT" "$FIX" "$OUT" <<'PY'
import json,sys,shutil,hashlib,os
from pathlib import Path
root,fix,out=map(Path,sys.argv[1:]);sys.path.insert(0,str(fix/'tooling'))
import runtime,products
qa=out/'population';(qa/'control').mkdir(parents=True);run=qa/'segment_0003';run.mkdir()
for name in ('initial_mass.npy','dead_tags.npy'):
    shutil.copy2(root/'control'/name,qa/'control'/name)
for name in ('s11_amd.prtcl_destroy.csv','s11_amd.gi_particles.csv'):
    (run/name).symlink_to(root/'runs/segment_0003'/name)
latest=runtime.checkpoint(root/'runs/segment_0003/rst/s11_amd.00005.rst',hashing=False)
runtime.audit_population(qa,run,[latest],latest)
quality=[]
for frame in sorted((root/'runs/segment_0003/trk').glob('*.trk2')):
    cycle,time,records=products.trk(frame)
    import numpy as np
    quality.append(dict(path=str(frame),cycle=cycle,time=time,count=len(records),unknown_energy=int((~np.isfinite(records['q'][:,1])).sum())))
assert quality,'missing cohort frames'
(out/'cohort_quality.json').write_text(json.dumps(quality,indent=2)+'\n')
scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (fix/'tooling').glob('*.py')}
(out/'RESULT.json').write_text(json.dumps(dict(status='pass',job=os.environ['SLURM_JOB_ID'],evolution_steps=0,tooling_sha256=scripts,audit=json.loads((run/'population_audit.json').read_text()),cohort_frames=len(quality)),indent=2)+'\n')
PY
du -sb /work1/eliasmost/jiaxiwu > "$OUT/whole_user_bytes.txt"
