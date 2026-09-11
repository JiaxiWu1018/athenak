#!/usr/bin/env bash
# Restart-continuity test (specification §5), in the rigorous Session-1 form.
#
# Takes a MID-RUN checkpoint from the production run, restarts it in a SEPARATE run
# directory, evolves it to a tlim the production run has already passed, and compares the
# two states at that same time.  Both paths reach the endpoint through the identical
# number of identical steps, so agreement should be at or near roundoff; a difference
# larger than ~1e-10 relative in any history column is a real discontinuity, i.e. some
# part of the state was not captured in the checkpoint.
#
# This is the only test a restart can be given that is not circular.  The duplicate
# ledger row at a segment boundary comes from Driver::Finalize inside a single process
# and says nothing about restarting.
#
#   restart_test.sh CASE CHECKPOINT_INDEX TARGET_PERIODS
# e.g. restart_test.sh R6p5 00003 1.0   -> restart t = 0.75 P, evolve to t = 1 P
set -uo pipefail
CASE=${1:?usage: CASE CHECKPOINT_INDEX TARGET_PERIODS}
IDX=${2:?usage: CASE CHECKPOINT_INDEX TARGET_PERIODS}
NP=${3:?usage: CASE CHECKPOINT_INDEX TARGET_PERIODS}

S=/data/jiaxiwu/NRPIC/Plummer-cluster/session_02_compactness_scan_20260910
H=hpcfund.amd.com
R=/work1/eliasmost/jiaxiwu/plummer_s02_20260910
LABEL=rst_test_$CASE
Sx() { ssh -o BatchMode=yes -o ConnectTimeout=30 "$H" "$@" 2>/dev/null; }

P=$(python3 -c "import json;print(repr(json.load(open('$S/initial_data/reference_values_$CASE.json'))['P_half']))")
TLIM=$(python3 -c "print(repr($NP*$P))")
echo "RESTART TEST $CASE: seeding from checkpoint $IDX, evolving to tlim = $TLIM ($NP P_1/2)"

src=$(Sx "ls -1 $R/runs/prod_$CASE/out/rst/*.$IDX.rst 2>/dev/null | head -1")
[ -n "$src" ] || { echo "RESTART TEST: no checkpoint $IDX under runs/prod_$CASE/out/rst"; exit 1; }
echo "RESTART TEST: source $src"
Sx "rm -rf $R/runs/$LABEL && mkdir -p $R/runs/$LABEL/out/rst &&
    cp '$src' $R/runs/$LABEL/out/rst/ && ls -la $R/runs/$LABEL/out/rst/"

j=$(Sx "cd $R && sbatch --parsable -t 03:00:00 -J $LABEL scripts/amd_run.sbatch $LABEL ${CASE}_prod time/tlim=$TLIM" | tr -dc '0-9')
[ -n "$j" ] || { echo "RESTART TEST: submit failed"; exit 1; }
echo "RESTART TEST: job $j"
absent=0
while true; do
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=30 "$H" "squeue -h -j $j -o %T" 2>/dev/null)
  if [ $? -ne 0 ]; then sleep 60; continue; fi
  if [ -z "$out" ]; then absent=$((absent+1)); [ "$absent" -ge 3 ] && break; else absent=0; fi
  sleep 30
done
Sx "grep -hoE 'CASE (DONE|INCOMPLETE|FAILED)' $R/logs/$LABEL.$j.log | tail -1"

echo "RESTART TEST: comparing the two states at t = $TLIM"
Sx "cd $R && python3 - <<'PY'
import glob, sys
import numpy as np
def load(d):
    g=[p for p in glob.glob(d+'/out/*.user.hst') if '.z4c.' not in p]
    if not g: return None, None
    names=None
    for line in open(g[0]):
        if line.startswith('#') and '=' in line:
            names=[p.split('=',1)[1] for p in line.replace('#','').split() if '=' in p]
    a=np.loadtxt(g[0], comments='#')
    a=np.atleast_2d(a)
    return names, a
t=$TLIM
n1,a1=load('runs/prod_$CASE')
n2,a2=load('runs/$LABEL')
if a1 is None or a2 is None: sys.exit('missing history file')
i1=int(np.argmin(np.abs(a1[:,0]-t))); i2=int(np.argmin(np.abs(a2[:,0]-t)))
print('production  row t = %.12g' % a1[i1,0])
print('restarted   row t = %.12g' % a2[i2,0])
worst=0.0; wk=None
print()
print('%-14s %22s %22s %12s' % ('column','production','restarted','rel diff'))
for k,name in enumerate(n1):
    if name in ('time','dt'): continue
    v1,v2=a1[i1,k],a2[i2,k]
    d=abs(v1-v2)/max(abs(v1),abs(v2),1e-300)
    if d>worst: worst,wk=d,name
    print('%-14s %22.14e %22.14e %12.3e' % (name,v1,v2,d))
print()
print('WORST relative difference %.3e in %s  ->  %s'
      % (worst, wk, 'PASS' if worst<1e-10 else ('MARGINAL' if worst<1e-6 else 'FAIL')))
PY"
echo "RESTART TEST COMPLETE ($CASE, job $j)"
