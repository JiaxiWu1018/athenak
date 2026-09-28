#!/bin/bash
# Movie rendering for one case.  Slurm, never a login node (AGENTS.md rule 1).
# Perseus now REJECTS CPU-only jobs and caps 2 Slurm CPUs per GPU, so a GPU is
# requested even though matplotlib and ffmpeg are CPU-only.
#   sbatch -J mov_<case> make_movies_s2.sbatch CASE [FPS]
#SBATCH -o /data/jiaxiwu/NRPIC/Plummer-cluster/session_02_compactness_scan_20260910/logs/%x.%j.log
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -p debug
set -uo pipefail
CASE=${1:?usage: CASE [FPS]}
FPS=${2:-12}
S=/data/jiaxiwu/NRPIC/Plummer-cluster/session_02_compactness_scan_20260910
cd "$S"
export MPLBACKEND=Agg
REF=initial_data/reference_values_$CASE.json
RUN=runs/prod_$CASE
OUT=movies/$CASE
mkdir -p "$OUT"
hostname; date -Is
rc_all=0
echo "=== Movie B: equatorial slices on the FIXED UNIFORM cart grid ==="
for panel in core wide; do
  mp4=$OUT/slices_H_and_E_equatorial_${panel}.mp4
  if python3 analysis/render_slice_movie_s2.py --rundir "$RUN" --ref "$REF" \
       --panel "$panel" --out "$mp4" --fps "$FPS"; then
    # a zero exit is not proof of a movie: require the file, non-empty
    if [ ! -s "$mp4" ]; then
      echo "FAIL: $panel renderer exited 0 but wrote no movie"; rc_all=1
    fi
  else
    echo "FAIL: $panel renderer exited non-zero"; rc_all=1
  fi
done
echo "=== Movie A: particles, core and halo ==="
NPZ=$OUT/${CASE}_particles.npz
if [ ! -s "$NPZ" ]; then
  echo "NOTE: $NPZ absent; the particle frames must be extracted on AMD first with"
  echo "      analysis/extract_movie_frames.py --rundir runs/prod_$CASE --out ... --npair 1056768"
else
  python3 analysis/render_particle_movie.py --npz "$NPZ" --ref "$REF" \
    --out "$OUT/particles_core_and_halo.mp4" --fps "$FPS"
fi
date -Is
if [ "$rc_all" -ne 0 ]; then
  echo "MOVIES INCOMPLETE ($CASE): at least one renderer failed above"
  exit 1
fi
echo "MOVIES DONE ($CASE)"
