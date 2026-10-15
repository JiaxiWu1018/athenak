#!/bin/bash
#SBATCH -J st_s05_vizdata
#SBATCH -N 1
#SBATCH -p mi2104x
#SBATCH -A eliasmost
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --exclude=k003-002,k003-009,k003-010
#SBATCH -t 00:15:00
#SBATCH -o /work1/eliasmost/jiaxiwu/st_reconstruction_s05_20261014/logs/%x.%j.log
set -euo pipefail
export ROOT=/work1/eliasmost/jiaxiwu/st_reconstruction_s05_20261014
review_repo=$1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
mkdir -p "$ROOT/evidence/jobs/$SLURM_JOB_ID"
git -C "$review_repo" diff --exit-code HEAD
git -C "$review_repo" rev-parse HEAD > "$ROOT/evidence/jobs/$SLURM_JOB_ID/visualization_source.txt"
{ hostname; date -Is; module list 2>&1; python3 --version; } > "$ROOT/evidence/jobs/$SLURM_JOB_ID/environment.txt"
storage_watchdog() {
  while true; do
    if used=$(du -s -B1 /work1/eliasmost/jiaxiwu 2>/dev/null | awk '{print $1}'); then
      printf '%s %s\n' "$(date -Is)" "$used" >> "$ROOT/evidence/jobs/$SLURM_JOB_ID/storage.txt"
      if (( used > 1869169767219 )); then scancel "$SLURM_JOB_ID"; fi
      if (( used > 1649267441664 )); then printf 'Storage warning: %s bytes\n' "$used" >&2; fi
    fi
    sleep 300
  done
}
storage_watchdog & watchdog_pid=$!
trap 'kill "$watchdog_pid" 2>/dev/null || true' EXIT
python3 "$review_repo/scripts/reconstruction_s05/particle_visual_data.py"
