#!/usr/bin/env bash
# Per-milestone analysis, run automatically by drive_milestones.sh after every completed
# P_1/2.  Reduces everything accumulated so far, pulls it to Perseus, regenerates the
# CUMULATIVE diagnostic figures into a milestone-specific directory, and writes an interim
# note.  Prior milestones are never overwritten.
#
#   analyze_milestone.sh CASE K
set -uo pipefail
CASE=${1:?usage: CASE K}
K=${2:?usage: CASE K}

S=/data/jiaxiwu/NRPIC/Plummer-cluster/session_02_compactness_scan_20260910
H=hpcfund.amd.com
R=/work1/eliasmost/jiaxiwu/plummer_s02_20260910
LABEL=prod_$CASE
FIG=$S/figures/$CASE/after_${K}P
RED=$S/reduced/${CASE}_P${K}
mkdir -p "$FIG" "$RED" "$S/notes" "$S/logs"
Sx() { ssh -o BatchMode=yes -o ConnectTimeout=30 "$H" "$@" 2>/dev/null; }

P=$(python3 -c "import json;print(repr(json.load(open('$S/initial_data/reference_values_$CASE.json'))['P_half']))")
BASE=$(python3 -c "import json;print(json.load(open('$S/initial_data/reference_values_$CASE.json'))['basename'])")
echo "ANALYZE $CASE P$K: P_1/2 = $P, basename $BASE  ($(date -u +%FT%TZ))"

# --- 1. reduce on AMD (bulk pvtk pass, Slurm, never a login node) -----------------
rsync -az "$S/analysis/" "$H:$R/analysis/" || true
j=$(Sx "cd $R && sbatch --parsable -J red_${CASE}_P${K} scripts/reduce_milestone.sbatch $CASE $K $P" | tr -dc '0-9')
if [ -z "$j" ]; then
  echo "ANALYZE $CASE P$K: reduction submit FAILED; skipping this milestone's figures"
  exit 1
fi
echo "ANALYZE $CASE P$K: reduction job $j"
absent=0
while true; do
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=30 "$H" "squeue -h -j $j -o %T" 2>/dev/null)
  if [ $? -ne 0 ]; then sleep 60; continue; fi
  if [ -z "$out" ]; then absent=$((absent+1)); [ "$absent" -ge 3 ] && break; else absent=0; fi
  sleep 30
done
Sx "grep -hoE 'REDUCE DONE .* rc=[0-9]+' $R/logs/red_${CASE}_P${K}.$j.log | tail -1"

# --- 2. pull the reduced products, the in-code ledgers, and the movie frames -------
rsync -az "$H:$R/reduced/${CASE}_P${K}/" "$RED/" || true
mkdir -p "$S/runs/$LABEL/out"
for f in "$BASE.user.hst" "$BASE.z4c.user.hst" "$BASE.plummer_shells.csv" \
         "$BASE.plummer_cohorts.csv" "$BASE.plummer_fields.csv" \
         "$BASE.plummer_admmom.csv"; do
  rsync -az "$H:$R/runs/$LABEL/out/$f" "$S/runs/$LABEL/out/" 2>/dev/null || true
done
rsync -az "$H:$R/runs/$LABEL/deck.athinput" "$S/runs/$LABEL/" 2>/dev/null || true
rsync -az --include='*/' --include='cart/***' --include='bin/***' --exclude='*' \
      "$H:$R/runs/$LABEL/" "$S/runs/$LABEL/" 2>/dev/null || true

# --- 3. cumulative figures, milestone-specific directory --------------------------
cd "$S"
MPLBACKEND=Agg python3 analysis/figs_session2.py \
  --case "$CASE" --milestone "$K" \
  --reduced "$RED" --rundir "runs/$LABEL" --ref "initial_data/reference_values_$CASE.json" \
  --out "$FIG" 2>&1 | tail -40

# --- 4. interim note --------------------------------------------------------------
MPLBACKEND=Agg python3 analysis/interim_note.py \
  --case "$CASE" --milestone "$K" \
  --reduced "$RED" --rundir "runs/$LABEL" --ref "initial_data/reference_values_$CASE.json" \
  --figdir "$FIG" --out "$S/notes/INTERIM_${CASE}_P${K}.md" 2>&1 | tail -30

echo "ANALYZE $CASE P$K: done. figures -> $FIG ; note -> notes/INTERIM_${CASE}_P${K}.md"
