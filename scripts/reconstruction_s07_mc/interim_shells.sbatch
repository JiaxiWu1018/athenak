#!/bin/bash
#SBATCH -J st_s07_interim400
#SBATCH -p mi2104x
#SBATCH -A eliasmost
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --exclude=k003-002,k003-009,k003-010
#SBATCH -t 00:30:00
#SBATCH -o /work1/eliasmost/jiaxiwu/st_reconstruction_s07_mc_20261019/logs/%x.%j.log
set -euo pipefail
export ROOT=/work1/eliasmost/jiaxiwu/st_reconstruction_s07_mc_20261019
export PYTHONPATH="$ROOT/src/athenak/scripts/reconstruction_s07_mc:$ROOT/src/athenak/scripts/reconstruction_s06:$ROOT/src/athenak/scripts/reconstruction_s05:$ROOT/src/athenak/vis/python:$ROOT/dependencies/site"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
record="$ROOT/evidence/jobs/${SLURM_JOB_ID}_single"
mkdir -p "$record"
{ hostname; date -Is; module list 2>&1; python3 --version; } > "$record/environment.txt"
watch_storage() {
  while true; do
    if used=$(du -s -B1 /work1/eliasmost/jiaxiwu 2>/dev/null | awk '{print $1}'); then
      printf '%s %s\n' "$(date -Is)" "$used" >> "$record/storage.txt"
      if (( used > 1869169767219 )); then
        touch "$ROOT/evidence/STORAGE_STOP"
        scancel "$SLURM_JOB_ID"
      elif (( used > 1649267441664 )); then
        printf 'Whole-user storage warning: %s bytes\n' "$used" >&2
      fi
    fi
    sleep 300
  done
}
watch_storage & storage_pid=$!
trap 'kill "$storage_pid" 2>/dev/null || true' EXIT
test ! -e "$ROOT/evidence/STORAGE_STOP"
python3 "$ROOT/src/athenak/scripts/reconstruction_s07_mc/verify_runtime.py"
audit_dir=${1:?expected isolated source directory}
python3 "$audit_dir/interim_shells.py"
