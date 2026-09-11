#!/bin/bash
# Movie pipeline for the Plummer campaign.
#   usage: make_movies.sh LABEL TAG [FPS]
# Frame extraction runs on the AMD side beside the pvtk dumps; only the small npz and the
# slice .bin files come back, and the rendering happens here.
set -uo pipefail
LABEL=${1:?usage: LABEL TAG [FPS]}
TAG=${2:?usage: LABEL TAG [FPS]}
FPS=${3:-12}
H=hpcfund.amd.com
R=/work1/eliasmost/jiaxiwu/plummer_s02_20260910
C=/data/jiaxiwu/NRPIC/Plummer-cluster/session_02_compactness_scan_20260910
P=1192.496781
S() { ssh -o BatchMode=yes -o ConnectTimeout=30 $H "$@" 2>/dev/null; }

base=$(S "grep -m1 '^basename' $R/runs/$LABEL/deck.athinput | awk '{print \$3}'")
rsync -a "$C/analysis/"{extract_movie_frames.py,pvtk_reader.py} "$H:$R/analysis/" || exit 1
S "cd $R && mkdir -p movies && python3 analysis/extract_movie_frames.py \
     --rundir runs/$LABEL --out movies/${LABEL}_particles.npz --nsub 200000 \
     --ncohort 32 --npair 1056768 2>&1 | tail -5"
mkdir -p "$C/movies/$TAG"
rsync -a "$H:$R/movies/${LABEL}_particles.npz" "$C/movies/$TAG/" || exit 1

export MPLBACKEND=Agg
python3 "$C/analysis/render_particle_movie.py" --npz "$C/movies/$TAG/${LABEL}_particles.npz" \
  --out "$C/movies/$TAG" --period $P --fps "$FPS" 2>&1 | tail -4
if [ -d "$C/runs/$LABEL/bin" ]; then
  python3 "$C/analysis/render_slice_movie.py" --dir "$C/runs/$LABEL/bin" --case "$base" \
    --out "$C/movies/$TAG" --period $P --fps "$FPS" 2>&1 | tail -4
else
  echo "MOVIES: no local slice directory; run analyze_production.sh first"
fi
ls -la "$C/movies/$TAG"/*.mp4 2>/dev/null
echo "MOVIES COMPLETE ($LABEL -> movies/$TAG)"
