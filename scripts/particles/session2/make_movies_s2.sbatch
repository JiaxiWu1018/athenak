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
echo "=== Movie B: equatorial slices on the FIXED UNIFORM cart grid ==="
for panel in core wide; do
  python3 analysis/render_slice_movie_s2.py --rundir "$RUN" --ref "$REF" \
    --panel "$panel" --out "$OUT/slices_H_and_E_equatorial_${panel}.mp4" --fps "$FPS"
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
echo "MOVIES DONE ($CASE)"
