set -euo pipefail
export ROOT=/work1/eliasmost/jiaxiwu/st_reconstruction_s06_20261016
export PYTHONPATH="$ROOT/src/athenak/scripts/reconstruction_s06:$ROOT/src/athenak/scripts/reconstruction_s05:$ROOT/src/athenak/vis/python:${PYTHONPATH:-}"
export ATHENAK_VIS_PYTHON="$ROOT/src/athenak/vis/python"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mkdir -p "$ROOT/evidence/jobs/$SLURM_JOB_ID"
{ hostname; date -Is; module list 2>&1; hipcc --version; python3 --version; } > "$ROOT/evidence/jobs/$SLURM_JOB_ID/environment.txt"
git -C "$ROOT/src/athenak" diff --exit-code HEAD -- scripts/reconstruction_s06 scripts/reconstruction_s05
git -C "$ROOT/src/athenak" rev-parse HEAD > "$ROOT/evidence/jobs/$SLURM_JOB_ID/script_commit.txt"
git -C "$ROOT/src/athenak/kokkos" rev-parse HEAD > "$ROOT/evidence/jobs/$SLURM_JOB_ID/kokkos.txt"
find "$ROOT/src/athenak/scripts/reconstruction_s06" -type f ! -path '*/__pycache__/*' -exec sha256sum {} + > "$ROOT/evidence/jobs/$SLURM_JOB_ID/script_hashes.txt"
storage_watchdog() {
  while true; do
    if used=$(du -s -B1 /work1/eliasmost/jiaxiwu 2>/dev/null | awk '{print $1}'); then
      printf '%s %s\n' "$(date -Is)" "$used" >> "$ROOT/evidence/jobs/$SLURM_JOB_ID/storage.txt"
      if (( used > 1869169767219 )); then
        printf 'Storage stop: %s bytes\n' "$used" >&2
        touch "$ROOT/evidence/STORAGE_STOP"
        scancel "$SLURM_JOB_ID"
      elif (( used > 1649267441664 )); then
        printf 'Storage warning: %s bytes\n' "$used" >&2
      fi
    fi
    sleep 300
  done
}
storage_watchdog & watchdog_pid=$!
trap 'kill "$watchdog_pid" 2>/dev/null || true' EXIT
test ! -e "$ROOT/evidence/STORAGE_STOP"
