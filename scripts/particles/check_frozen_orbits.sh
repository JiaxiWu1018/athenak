#!/bin/bash
# Early orbit-integrity gate on the frozen-metric preflight (PF-B).
#
# PF-B evolves the production particle set for a full reference period in the frozen
# analytic metric on the production grid. Its purpose is to certify the pusher, gather,
# migration and cross-refinement machinery BEFORE production node-hours are committed.
# Waiting for the whole 5 h run is not necessary for that decision: a quarter period is
# already many core orbits and covers every refinement face the stratified sample crosses.
#
#   usage: check_frozen_orbits.sh MIN_FRAMES
#
# Reduces whatever pvtk frames exist on the AMD side (beside the data, so nothing large
# moves), pulls back only the CSVs, and prints the E = -u_t and |L| = |x x u| drift.
set -uo pipefail
MINF=${1:-25}
H=hpcfund.amd.com
R=/work1/eliasmost/jiaxiwu/plummer_s01_20260906
C=/data/jiaxiwu/NRPIC/Plummer-cluster
S() { ssh -o BatchMode=yes -o ConnectTimeout=30 $H "$@" 2>/dev/null; }

for i in $(seq 1 400); do
  n=$(S "ls -1 $R/runs/pf_frozen/out/pvtk/*.part.vtk 2>/dev/null | wc -l")
  n=${n:-0}
  [ "$n" -ge "$MINF" ] && { echo "FROZEN: $n pvtk frames available, reducing"; break; }
  sleep 60
done
[ "${n:-0}" -ge "$MINF" ] || { echo "FROZEN: only $n frames after waiting; aborting gate"; exit 1; }

S "mkdir -p $R/analysis && test -d $R/analysis/.ok || true"
rsync -a "$C/analysis/"{reduce_run.py,pvtk_reader.py,sph_real.py} "$H:$R/analysis/" || exit 1
S "cd $R && python3 analysis/reduce_run.py --rundir runs/pf_frozen --out reduced/pf_frozen \
     --period 1192.496781 --ncohort 32 --macro 4 2>&1 | tail -25"
mkdir -p "$C/reduced/pf_frozen"
rsync -a "$H:$R/reduced/pf_frozen/" "$C/reduced/pf_frozen/" || exit 1
python3 - "$C/reduced/pf_frozen/scalars.csv" << 'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("FROZEN: no rows in scalars.csv"); sys.exit(1)
print("FROZEN ORBIT INTEGRITY (frozen analytic metric, production grid and N)")
print(f"  frames {len(rows)}, t/P from {float(rows[0]['t_over_P']):.4f} to "
      f"{float(rows[-1]['t_over_P']):.4f}")
print(f"  {'t/P':>8} {'dE_rms':>10} {'dE_max':>10} {'dL_rms':>10} {'dL_max':>10} "
      f"{'dr_rms':>10} {'N':>9} {'nonfin':>7}")
step = max(1, len(rows)//10)
for r in rows[::step] + [rows[-1]]:
    print(f"  {float(r['t_over_P']):8.4f} {float(r['dE_rms']):10.3e} "
          f"{float(r['dE_max']):10.3e} {float(r['dL_rms']):10.3e} "
          f"{float(r['dL_max']):10.3e} {float(r['dr_rms']):10.3e} "
          f"{int(r['N_alive']):9d} {int(r['n_nonfinite']):7d}")
last = rows[-1]
ok = (float(last['dE_rms']) < 1e-3 and float(last['dL_rms']) < 1e-3
      and int(last['n_nonfinite']) == 0)
print("FROZEN GATE: %s  (target: rms drift in E and |L| below 1e-3, no non-finite)"
      % ("PASS" if ok else "REVIEW"))
PY
echo "FROZEN ORBIT CHECK COMPLETE"
