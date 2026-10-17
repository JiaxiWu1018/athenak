#!/bin/bash
#SBATCH -J st_s06_movie
#SBATCH -p preempt
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/06_yc0819_long_stability_20261016/logs/%x.%j.log
set -euo pipefail
export ROOT=/data/jiaxiwu/NRPIC/reconstruction_ST/06_yc0819_long_stability_20261016
export ATHENAK_VIS_PYTHON="$ROOT/src/athenak/vis/python"
export PYTHONPATH="$ROOT/scripts:$ROOT/src/athenak/scripts/reconstruction_s05:$ATHENAK_VIS_PYTHON"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
notify_exit() {
  result=$?
  /home/jiaxiwu/.local/bin/codex queue --thread 01a14666-7bb8-74d1-8819-0037c04e9eb6 \
    --message "ST reconstruction session6 movie event ${SLURM_JOB_ID}_exit. Allocated movie job $SLURM_JOB_ID exited status $result. Read /data/jiaxiwu/NRPIC/reconstruction_ST/06_yc0819_long_stability_20261016/HANDOFF.md and evidence/MOVIE_STATE.json; inspect movie receipts, visually review and complete the already-authorized particle movie addendum. This event does not replace the task." || true
  exit "$result"
}
trap notify_exit EXIT
mkdir -p "$ROOT/evidence/jobs/$SLURM_JOB_ID"
{ hostname; date -Is; python3 --version; command -v python3; if type module >/dev/null 2>&1; then module list 2>&1; else echo 'No environment modules loaded'; fi; ffmpeg -version; git -C "$ROOT/src/athenak" rev-parse HEAD; } > "$ROOT/evidence/jobs/$SLURM_JOB_ID/environment.txt"
git -C "$ROOT/src/athenak" diff --exit-code HEAD -- scripts/reconstruction_s06 scripts/reconstruction_s05
case "$1" in
  prepare)
    python3 "$ROOT/scripts/particle_movie.py" extract
    python3 "$ROOT/scripts/particle_movie.py" preview
    ;;
  render)
    python3 "$ROOT/scripts/particle_movie.py" render
    ;;
  audit)
    python3 "$ROOT/scripts/audit_particle_movie.py"
    ;;
  *) exit 2 ;;
esac
