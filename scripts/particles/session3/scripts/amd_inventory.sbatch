#!/bin/bash
#SBATCH --job-name=pl_s3_raw_inventory
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/work1/eliasmost/jiaxiwu/plummer_s03_20261005/logs/inventory_%j.log
set -euo pipefail
root=${PLUMMER_S3_ROOT:-/work1/eliasmost/jiaxiwu/plummer_s03_20261005}
source "$root/scripts/amd_watchdog.sh"
start_storage_watchdog "$root"
trap 'kill "$STORAGE_PID" 2>/dev/null || true' EXIT
hostname
date -u
du -sb "$root/runs"/* > "$root/evidence/amd_raw_sizes.tsv"
python3 - "$root" <<'PY'
import hashlib,sys,csv
from pathlib import Path
root=Path(sys.argv[1])
with (root/'evidence/amd_raw_manifest.tsv').open('w') as out:
    writer=csv.writer(out,delimiter='\t');writer.writerow(['path','bytes','sha256','retention'])
    for path in sorted((root/'runs').rglob('*')):
        if not path.is_file():continue
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
        writer.writerow([str(path),path.stat().st_size,digest.hexdigest(),'retained on AMD; no deletion authorized; Anta destination full'])
with (root/'evidence/reduced_transfer.sha256').open('w') as out:
    for path in sorted((root/'reduced').rglob('*')):
        if path.is_file():
            out.write(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+str(path.relative_to(root))+'\n')
PY
date -u
