#!/bin/bash
#SBATCH -J jeans11-products
#SBATCH -A eliasmost
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=${SESSION_ROOT:?}
export MPLBACKEND=Agg MPLCONFIGDIR=$ROOT/.mpl_cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT=$ROOT/products/endpoint75_${SLURM_JOB_ID}
python3 "$ROOT/scripts/products.py" --root "$ROOT" --out "$OUT" --movies
python3 - "$ROOT" "$OUT" <<'PY'
import json,os,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:])
(root/'control/PRODUCTS_COMPLETE.json').write_text(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],path=str(out),manifest=str(out/'manifest.json')),indent=2)+'\n')
PY
