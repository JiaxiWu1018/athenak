#!/bin/bash
#SBATCH -J st_s06ps
#SBATCH -p mi2104x
#SBATCH -A eliasmost
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --exclude=k003-002,k003-009,k003-010
#SBATCH -t 16:00:00
#SBATCH -o /work1/eliasmost/jiaxiwu/st_reconstruction_s06_paired_scaling_20261017/logs/%x.%A_%a.log
set -euo pipefail
export ROOT=/work1/eliasmost/jiaxiwu/st_reconstruction_s06_paired_scaling_20261017
export PYTHONPATH="$ROOT/src/athenak/scripts/reconstruction_s06_scaling:$ROOT/src/athenak/scripts/reconstruction_s05:$ROOT/src/athenak/vis/python"
export ATHENAK_VIS_PYTHON="$ROOT/src/athenak/vis/python"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mkdir -p "$ROOT/evidence/jobs/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-single}"
{ hostname; date -Is; module list 2>&1; hipcc --version; python3 --version; } > "$ROOT/evidence/jobs/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-single}/environment.txt"
watch_storage() {
  while true; do
    if used=$(du -s -B1 /work1/eliasmost/jiaxiwu 2>/dev/null | awk '{print $1}'); then
      printf '%s %s\n' "$(date -Is)" "$used" >> "$ROOT/evidence/jobs/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-single}/storage.txt"
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
case "$1" in
  prepare) python3 "$ROOT/src/athenak/scripts/reconstruction_s06_scaling/prepare.py" ;;
  preflight) python3 "$ROOT/src/athenak/scripts/reconstruction_s06_scaling/run.py" preflight "$SLURM_ARRAY_TASK_ID" ;;
  production) python3 "$ROOT/src/athenak/scripts/reconstruction_s06_scaling/run.py" production "$2" ;;
  *) exit 2 ;;
esac
