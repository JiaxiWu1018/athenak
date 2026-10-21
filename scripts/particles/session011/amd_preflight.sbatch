#!/bin/bash
#SBATCH -J jeans11-preflight
#SBATCH -A eliasmost
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=${SESSION_ROOT:?}
OUT=$ROOT/evidence/preflight_${SLURM_JOB_ID}
mkdir -p "$OUT"
export MPLBACKEND=Agg MPLCONFIGDIR=$ROOT/.mpl_cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PRODUCT_QA=$OUT/qa
python3 "$ROOT/scripts/test_runtime.py" > "$OUT/controller_checks.log" 2>&1
python3 "$ROOT/scripts/test_products.py" > "$OUT/product_checks.log" 2>&1
python3 - <<'PY' > "$OUT/encoder.txt"
import imageio_ffmpeg
print(imageio_ffmpeg.get_ffmpeg_exe())
PY
du -sb /work1/eliasmost/jiaxiwu > "$OUT/whole_user_bytes.txt"
df -B1 /work1 > "$OUT/filesystem.txt"
python3 - "$ROOT" "$OUT" <<'PY'
import hashlib,json,sys,os,subprocess
from pathlib import Path
root,out=map(Path,sys.argv[1:])
used=int((out/'whole_user_bytes.txt').read_text().split()[0])
assert used+128*1024**3 < 1.7*1024**4,'insufficient initialization and recovery reserve'
inputs=root/'inputs/production.athinput'
sys.path.insert(0,str(root/'scripts'))
from make_input import parse
p=parse(inputs.read_text())
assert float(p['time']['tlim'])==75 and p['time']['integrator']=='rk4'
assert float(p['time']['cfl_number'])==.4 and p['mesh_refinement']['max_nmb_per_rank']=='256'
assert p['particles']['excise_ah']=='false' and float(p['particles']['excise_lapse'])==.05
assert p['problem']['gi_seed']=='4001'
source=subprocess.check_output(['git','-C',str(root/'code'),'rev-parse','HEAD'],text=True).strip()
scripts={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'scripts').iterdir()) if p.is_file()}
result=dict(status='pass',job_id=os.environ['SLURM_JOB_ID'],source=source,runtime_scripts=scripts,whole_user_bytes=used,input_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest())
(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
PY
if [[ -n ${FINAL_CHECK:-} ]]; then
  python3 "$ROOT/scripts/release_gate.py" --root "$ROOT" --final "$FINAL_CHECK" --preflight "$OUT" > "$OUT/validation_gate.json"
fi
