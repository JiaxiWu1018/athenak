#!/bin/bash
#SBATCH --job-name=pl_s3_inventory
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005/logs/inventory_%j.log
set -euo pipefail
session=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005
hostname
date -u
du -sb "$session/runs"/* > "$session/evidence/perseus_raw_sizes.tsv"
python3 - "$session" <<'PY'
import hashlib,sys,csv
from pathlib import Path
root=Path(sys.argv[1])
with (root/'evidence/perseus_raw_manifest.tsv').open('w') as out:
    writer=csv.writer(out,delimiter='\t');writer.writerow(['path','bytes','sha256','retention'])
    for path in sorted((root/'runs').rglob('*')):
        if not path.is_file():continue
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
        writer.writerow([str(path),path.stat().st_size,digest.hexdigest(),'retained on Perseus; no deletion authorized; Anta destination currently full'])
PY
date -u
