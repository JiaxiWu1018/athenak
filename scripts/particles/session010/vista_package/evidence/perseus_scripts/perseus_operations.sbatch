#!/bin/bash
#SBATCH -J jeans10-operations
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH -t 00:20:00
#SBATCH --mail-user=jiaxiwu@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
SESSION=/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013
PACKAGE=$SESSION/code/scripts/particles/session010/vista_package
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export S7_SOURCE_TREE=$SESSION/code
cd "$PACKAGE"
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 - "$PACKAGE" <<'PY'
import ast,sys
from pathlib import Path
for p in Path(sys.argv[1]).rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
PY
for f in scripts/*.sbatch scripts/*.sh; do test ! -f "$f" || bash -n "$f"; done
printf '{"passed":true,"job_id":%s}\n' "$SLURM_JOB_ID" > "$SESSION/evidence/operations_receipt.json"
