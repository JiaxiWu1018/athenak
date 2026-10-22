#!/bin/bash
#SBATCH -A eliasmost
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:05:00
#SBATCH -J s11-render-check
#SBATCH --no-requeue
set -euo pipefail
ROOT=${SESSION_ROOT:?}
SOURCE_COMMIT=${SOURCE_COMMIT:?}
OUT=$ROOT/evidence/render_qa_$SLURM_JOB_ID
mkdir -p "$OUT/source"
git -C "$ROOT/code" archive "$SOURCE_COMMIT" scripts/particles/session011 | tar -x -C "$OUT/source"
export TOOLING_DIR=$OUT/source/scripts/particles/session011
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=$ROOT/.mpl_cache PRODUCT_QA=$OUT/qa
export RENDERER_SHA256=$(sha256sum "$TOOLING_DIR/products.py" | cut -d ' ' -f 1)
{
 date -u
 hostname
 scontrol show job "$SLURM_JOB_ID"
 git -C "$ROOT/code" rev-parse "$SOURCE_COMMIT"
} > "$OUT/provenance.txt"
python3 "$TOOLING_DIR/test_runtime.py" > "$OUT/runtime_tests.log" 2>&1
python3 "$TOOLING_DIR/test_products.py" > "$OUT/product_tests.log" 2>&1
if [[ -f "$TOOLING_DIR/diagnostics/test_product_regressions.py" ]]; then
 python3 "$TOOLING_DIR/diagnostics/test_product_regressions.py" > "$OUT/product_regressions.log" 2>&1
fi
python3 "$TOOLING_DIR/diagnostics/check_real_render.py" > "$OUT/real_render.log" 2>&1
python3 - "$OUT" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
out=Path(sys.argv[1]);scripts=Path(os.environ['TOOLING_DIR'])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=Path(os.environ['SESSION_ROOT'])/'evidence'/f'real_render_{os.environ["SLURM_JOB_ID"]}/RESULT.json'
assert json.loads(receipt.read_text())['status']=='pass'
record=dict(status='pass',source=os.environ['SOURCE_COMMIT'],job_id=os.environ['SLURM_JOB_ID'],
 real_render_receipt=str(receipt),real_render_sha256=sha(receipt),
 tooling_sha256={n:sha(scripts/n) for n in ('endpoint.py','products.py','test_runtime.py','test_products.py')})
(out/'RESULT.json').write_text(json.dumps(record,indent=2)+'\n')
PY
if [[ ${RUN_REAL_REPORT:-0} == 1 ]]; then
 export REPORT_RENDER_QA=$OUT/RESULT.json
 python3 "$TOOLING_DIR/diagnostics/check_real_report.py" > "$OUT/real_report.log" 2>&1
fi
