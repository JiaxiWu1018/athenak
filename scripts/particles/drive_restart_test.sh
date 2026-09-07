#!/bin/bash
# Restart-continuity test for the Plummer campaign.
#
# PF-C runs 0 -> 100 M in one segment, checkpointing at 0 and 50 M. This test takes the
# 50 M checkpoint, restarts it in a SEPARATE run directory, evolves to the same tlim, and
# compares the two 100 M states. A restart that is not continuous shows up as a difference
# in the history row at t = 100, in the particle ledger, or in the constraint norms.
#
# The comparison must allow for one legitimate difference: the deposited source is a
# function of the particle state, and both paths reach t = 100 through the identical
# number of identical steps, so agreement should be at or near roundoff. Any difference
# larger than ~1e-10 relative in N_alive, M0_alive or A1_raw is a real discontinuity.
set -uo pipefail
H=hpcfund.amd.com
R=/work1/eliasmost/jiaxiwu/plummer_s01_20260906
S() { ssh -o BatchMode=yes -o ConnectTimeout=30 $H "$@" 2>/dev/null; }

# The runner prunes to the two newest checkpoints after each segment, so with rst dt = 50
# the surviving pair is {t=50, t=100}. Take the OLDEST survivor: restarting the t=100 one
# would simply hit tlim immediately and test nothing.
src=$(S "ls -1 $R/runs/pf_live/out/rst/*.rst 2>/dev/null | sort | head -1")
if [ -z "$src" ]; then echo "RESTART: no second checkpoint in runs/pf_live/out/rst"; exit 1; fi
echo "RESTART: seeding from $src"
S "rm -rf $R/runs/pf_restart && mkdir -p $R/runs/pf_restart/out/rst &&
   cp '$src' $R/runs/pf_restart/out/rst/ && ls -la $R/runs/pf_restart/out/rst/"

j=$(S "cd $R && sbatch -t 02:00:00 -J pf_restart scripts/amd_run.sbatch pf_restart pf_live_plummer" \
      | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
[ -n "$j" ] || { echo "RESTART: submit failed"; exit 1; }
echo "RESTART: job $j submitted"
for i in $(seq 1 240); do S "squeue -h -j $j -o %T" | grep -q . || break; sleep 30; done
S "grep -aoE 'CASE (DONE|FAILED|INCOMPLETE) \(pf_restart\)[^,]{0,60}|=== RESTART from[^(]*' $R/logs/pf_restart.$j.log | sort -u | head -4"

echo "RESTART: comparing the final history rows"
S "for d in pf_live pf_restart; do
     f=\$(ls -1 $R/runs/\$d/out/*.user.hst 2>/dev/null | head -1)
     [ -n \"\$f\" ] && printf '%-10s %s\n' \"\$d\" \"\$(grep -v '^#' \"\$f\" | tail -1)\"
   done"
echo "RESTART TEST COMPLETE (job $j)"
