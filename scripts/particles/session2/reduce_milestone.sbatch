#!/bin/bash
# Slurm-side reduction of one case up to one milestone.  Bulk pvtk pass: never on a
# login node.  usage: sbatch -J red_<case>_P<k> reduce_milestone.sbatch CASE K PERIOD
#SBATCH -o /work1/eliasmost/jiaxiwu/plummer_s02_20260910/logs/%x.%j.log
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -p mi2101x
#SBATCH -t 6:00:00
set -uo pipefail
CASE=${1:?usage: CASE K PERIOD}
K=${2:?usage: CASE K PERIOD}
P=${3:?usage: CASE K PERIOD}
R=/work1/eliasmost/jiaxiwu/plummer_s02_20260910
cd "$R"
TMAX=$(python3 -c "print(repr($K*$P))")
OUT=reduced/${CASE}_P${K}
mkdir -p "$OUT"
echo "=== reduce $CASE through milestone $K : tmax = $TMAX M  ($(date -Is)) ==="
# Each milestone reduces into its OWN directory, so a milestone's products are never
# overwritten by a later one and a partial run cannot truncate an earlier result.
# reduce_run.py opens its CSVs with mode 'w', so it is not append-safe; a per-milestone
# directory plus --tmax is the zero-code-change way to get cumulative products.
python3 analysis/reduce_run.py --rundir "runs/prod_$CASE" --out "$OUT" \
  --period "$P" --lmax 4 --macro 4 --ncohort 32 --tmax "$TMAX"
rc=$?
echo "REDUCE DONE ($CASE P$K) rc=$rc  $(date -Is)"
ls -la "$OUT"
exit $rc
