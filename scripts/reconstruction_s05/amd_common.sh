set -euo pipefail
export ROOT=/work1/eliasmost/jiaxiwu/st_reconstruction_s05_20261014
export PYTHONPATH="$ROOT/scripts:$ROOT/src/athenak/vis/python:${PYTHONPATH:-}"
export ATHENAK_VIS_PYTHON="$ROOT/src/athenak/vis/python"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mkdir -p "$ROOT/evidence/jobs/$SLURM_JOB_ID"
{ hostname; date -Is; module list 2>&1; hipcc --version; python3 --version; } > "$ROOT/evidence/jobs/$SLURM_JOB_ID/environment.txt"

# Use the versioned scripts directly; reject detached, stale deployment copies.
python3 - <<'PYCODE'
import hashlib,json,os,subprocess
from pathlib import Path
root=Path(os.environ['ROOT']);repo=root/'src/athenak'
if (root/'scripts').resolve()!=(repo/'scripts/reconstruction_s05').resolve():
    raise RuntimeError('AMD scripts must resolve to the versioned source directory')
subprocess.run(['git','-C',str(repo),'diff','--quiet','HEAD','--','scripts/reconstruction_s05'],check=True)
paths=subprocess.check_output(['git','-C',str(repo),'ls-files','scripts/reconstruction_s05'],text=True).splitlines()
manifest=[{'path':s,'sha256':hashlib.sha256((repo/s).read_bytes()).hexdigest()} for s in paths]
evidence=root/'evidence/jobs'/os.environ['SLURM_JOB_ID']/'scripts.json'
evidence.write_text(json.dumps({'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
    'resolved_scripts':str((root/'scripts').resolve()),'manifest':manifest},indent=2)+'\n')
PYCODE

# Monitor the whole user root; transient du failures preserve the watchdog.
storage_watchdog() {
  while true; do
    if used=$(du -s -B1 /work1/eliasmost/jiaxiwu 2>/dev/null | awk '{print $1}'); then
      date -Is >> "$ROOT/evidence/jobs/$SLURM_JOB_ID/storage.txt"
      printf '%s\n' "$used" >> "$ROOT/evidence/jobs/$SLURM_JOB_ID/storage.txt"
      if (( used > 1869169767219 )); then
        printf 'Storage stop at %s bytes\n' "$used" >&2
        touch "$ROOT/evidence/STORAGE_STOP"
        scancel "$SLURM_JOB_ID"
      elif (( used > 1649267441664 )); then
        printf 'Storage warning at %s bytes\n' "$used" >&2
      fi
    fi
    sleep 300
  done
}
storage_watchdog & watchdog_pid=$!
trap 'kill "$watchdog_pid" 2>/dev/null || true' EXIT
