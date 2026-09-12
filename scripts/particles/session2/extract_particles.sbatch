#!/bin/bash
# Extract the Movie-A particle subset on AMD, where the pvtk dumps live and stay.
# Only the resulting few-tens-of-MB npz crosses to Perseus, which is where ffmpeg is.
#   sbatch -J pext_<case> extract_particles.sbatch CASE
#SBATCH -o /work1/eliasmost/jiaxiwu/plummer_s02_20260910/logs/%x.%j.log
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -p mi2101x
#SBATCH -t 4:00:00
set -uo pipefail
CASE=${1:?usage: CASE}
R=/work1/eliasmost/jiaxiwu/plummer_s02_20260910
cd "$R"
mkdir -p movies
python3 analysis/extract_movie_frames.py --rundir "runs/prod_$CASE" \
  --out "movies/${CASE}_particles.npz" --nsub 200000 --ncohort 32 --npair 1056768
rc=$?
ls -la "movies/${CASE}_particles.npz" 2>/dev/null
echo "PARTICLE EXTRACT DONE ($CASE) rc=$rc"
exit $rc
