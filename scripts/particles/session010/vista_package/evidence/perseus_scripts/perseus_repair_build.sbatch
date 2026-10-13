#!/bin/bash
#SBATCH -J jeans10-repair
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 00:20:00
#SBATCH --mail-user=jiaxiwu@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
SESSION=/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013
export PATH=/home/emost/codes/openmpi/bin:/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ROOT=$SESSION/evidence/build_repair_${SLURM_JOB_ID}
mkdir -p "$ROOT"
scontrol show job "$SLURM_JOB_ID" > "$ROOT/slurm.txt"
cp "$SESSION/evidence/build_12753/patched/src/athena" "$SESSION/evidence/build_12753/patched/src/athena.0b7c630b"
cmake --build "$SESSION/evidence/build_12753/patched" -j 2 > "$ROOT/build.log" 2>&1
cp "$SESSION/evidence/build_12753/patched/src/athena" "$ROOT/athena"
git -C "$SESSION/code" rev-parse HEAD > "$ROOT/source.txt"
sha256sum "$ROOT/athena" > "$ROOT/athena.sha256"
python3 - "$SESSION" "$ROOT" <<'INNER'
import json,sys
from pathlib import Path
s,r=map(Path,sys.argv[1:]);p=s/'evidence/build_receipt.json'
d=json.loads(p.read_text());(r/'prior_receipt.json').write_text(p.read_text());d.update(patched=str(r/'athena'),repair_root=str(r));p.write_text(json.dumps(d,indent=2)+'\n')
INNER
