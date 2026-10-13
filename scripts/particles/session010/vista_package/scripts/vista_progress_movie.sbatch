#!/bin/bash
set -euo pipefail
umask 077

HANDOFF=/scratch/10222/jiaxiwu1018/nrpic_s7/vista_handoff
RUN=/scratch/10222/jiaxiwu1018/nrpic_s7_runs/s7_two_clump_ccw_20260909T0617Z/segments/segment_001/out
OUTPUT=/scratch/10222/jiaxiwu1018/nrpic_s7_movies/s7_progress_20260909T2015Z
HOME_MOVIE=/home1/10222/jiaxiwu1018/s7_two_clump_ccw_progress_t0_t15_20260909.mp4

test -f "$HANDOFF/analysis_support/render_progress_movie.py"
test -d "$RUN/pvtk"
test -f "$HANDOFF/s7_two_clump_ccw.horizon_consumer_0.csv"
test -f "$HANDOFF/s7_two_clump_ccw.horizon_consumer_1.csv"
mkdir -p "$OUTPUT/tmp" "$OUTPUT/mplconfig"
export TMPDIR=$OUTPUT/tmp
export MPLCONFIGDIR=$OUTPUT/mplconfig
module reset >/dev/null 2>&1
module load gcc/13.2.0 python3/3.11.8
export LD_LIBRARY_PATH=/opt/apps/gcc14/openmpi5/tau/2.33.2/arm64_linux/libunwind-1.8.1/lib:${LD_LIBRARY_PATH:-}
python3 -c 'import matplotlib, numpy, PIL'
ffmpeg -version >/dev/null
ffprobe -version >/dev/null

{
  echo "job_id=${SLURM_JOB_ID:-none}"
  echo "host=$(hostname)"
  echo "started=$(date -Is)"
  echo "particle_dir=$RUN/pvtk"
  echo "output=$OUTPUT"
  echo "home_movie=$HOME_MOVIE"
  module list 2>&1
} > "$OUTPUT/provenance.txt"

python3 "$HANDOFF/analysis_support/render_progress_movie.py" \
  --particle-dir "$RUN/pvtk" \
  --horizon "$HANDOFF/s7_two_clump_ccw.horizon_consumer_0.csv" \
  --horizon "$HANDOFF/s7_two_clump_ccw.horizon_consumer_1.csv" \
  --output-dir "$OUTPUT" \
  --movie "$OUTPUT/s7_two_clump_ccw_progress_t0_t15.mp4" \
  --home-movie "$HOME_MOVIE" \
  --limit 32 --cells 512 --fps 10

echo "completed=$(date -Is)" >> "$OUTPUT/provenance.txt"
