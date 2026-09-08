#!/bin/bash
# One-command reduction and figure set for the Plummer production run.
#
#   usage: analyze_production.sh LABEL TAG [FIT_LO] [FIT_HI]
#     LABEL   run subdirectory on AMD (e.g. prod_plummer)
#     TAG     output subdirectory name here (e.g. Phalf, 3Phalf)
#     FIT_LO/FIT_HI   growth-rate fit window in t/P_1/2 (default 0.25 to the endpoint)
#
# Heavy work runs on the AMD side beside the data (the pvtk dumps are 81 MB each and
# there are hundreds); only the small CSVs, the history file and the figures come back.
set -uo pipefail
LABEL=${1:?usage: LABEL TAG [FIT_LO] [FIT_HI]}
TAG=${2:?usage: LABEL TAG [FIT_LO] [FIT_HI]}
FLO=${3:-0.25}
FHI=${4:-99}
H=hpcfund.amd.com
R=/work1/eliasmost/jiaxiwu/plummer_s01_20260906
C=/data/jiaxiwu/NRPIC/Plummer-cluster
P=1192.496781
S() { ssh -o BatchMode=yes -o ConnectTimeout=30 $H "$@" 2>/dev/null; }

# Serialise on the OUTPUT directory. reduce_run.py opens its CSVs with 'w', so two
# concurrent runs truncate files the other is still writing and leave a hole of NUL bytes
# rather than a clean overwrite. That happened once: a relaunched watcher has a fresh
# in-memory "already fired" flag, so it fired a second interim analysis on top of a running
# one. An in-memory guard cannot survive a restart; a lock on disk can.
LOCK="$C/reduced/.$LABEL.analyze.lock"
mkdir -p "$C/reduced"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "ANALYZE: another analysis of $LABEL is already running (lock $LOCK); refusing to"
  echo "ANALYZE: run concurrently, because both would write the same CSVs."
  exit 0
fi

base=$(S "grep -m1 '^basename' $R/runs/$LABEL/deck.athinput | awk '{print \$3}'")
[ -n "$base" ] || { echo "ANALYZE: cannot read basename from $LABEL/deck.athinput"; exit 1; }
nf=$(S "ls -1 $R/runs/$LABEL/out/pvtk/*.part.vtk 2>/dev/null | wc -l")
echo "ANALYZE: $LABEL basename=$base, $nf pvtk frames"

# ---- reduce over ALL particles, on the AMD side --------------------------------
rsync -a "$C/analysis/"{reduce_run.py,pvtk_reader.py,sph_real.py} "$H:$R/analysis/" || exit 1
S "mkdir -p $R/reduced/$LABEL && cd $R && python3 analysis/reduce_run.py \
     --rundir runs/$LABEL --out reduced/$LABEL --period $P --lmax 4 --macro 4 \
     --ncohort 32 2>&1 | tail -8"

# ---- pull the small products ---------------------------------------------------
mkdir -p "$C/reduced/$LABEL" "$C/runs/$LABEL" "$C/figures/$TAG"
rsync -a "$H:$R/reduced/$LABEL/" "$C/reduced/$LABEL/"
for f in "$base.user.hst" "$base.z4c.user.hst" "$base.plummer_shells.csv" \
         "$base.plummer_cohorts.csv" "$base.plummer_fields.csv"; do
  rsync -a "$H:$R/runs/$LABEL/out/$f" "$C/runs/$LABEL/" 2>/dev/null
done
rsync -a "$H:$R/runs/$LABEL/deck.athinput" "$C/runs/$LABEL/"
rsync -a --include='*/' --include="$base.con.*.bin" --include="$base.tmunu.*.bin" \
      --exclude='*' "$H:$R/runs/$LABEL/out/bin/" "$C/runs/$LABEL/bin/" 2>/dev/null

# ---- figures -------------------------------------------------------------------
export MPLBACKEND=Agg
python3 "$C/analysis/figs_production.py" --hst "$C/runs/$LABEL/$base.user.hst" \
  --out "$C/figures/$TAG" --modes "$C/reduced/$LABEL/modes.csv" \
  --scalars "$C/reduced/$LABEL/scalars.csv" --cohorts "$C/reduced/$LABEL/cohorts.csv" \
  --period $P --fit-lo "$FLO" --fit-hi "$FHI" 2>&1 | tail -6
python3 "$C/analysis/figs_t0.py" "$C/runs/$LABEL/$base.plummer_shells.csv" \
  "$C/figures/$TAG/state" "$C/runs/$LABEL/$base.plummer_fields.csv" 2>&1 | tail -2

echo "ANALYZE: fit summary"
cat "$C/figures/$TAG/FIT_SUMMARY.txt"
echo "ANALYZE COMPLETE ($LABEL -> figures/$TAG, reduced/$LABEL)"
