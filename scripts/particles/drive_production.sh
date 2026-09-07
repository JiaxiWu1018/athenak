#!/bin/bash
# Perseus-side driver for the Plummer production run: 0 -> 3 P_1/2 = 3577.490343 M,
# chained restart segments on one AMD MI210 node.
#
#   usage: drive_production.sh [SEGMENT_HOURS] [MAX_SEGMENTS]
#
# The AMD runner (amd_run.sbatch) restarts from the newest checkpoint whenever the same
# command is resubmitted, and hands Athena a wall budget 15 min short of Slurm's so it
# always writes a final checkpoint. This driver therefore just resubmits until the run
# reports reaching tlim, and prunes checkpoints between segments -- keeping the two
# newest plus every checkpoint that lands on an integer P_1/2 (the rst cadence is
# P_1/2/4, so those are dump indices 0, 4, 8, 12).
#
# It emits one progress line per segment; each line becomes a notification.
set -uo pipefail
SEGH=${1:-12}
MAXSEG=${2:-12}
H=hpcfund.amd.com
R=/work1/eliasmost/jiaxiwu/plummer_s01_20260906
LABEL=prod_plummer
DECK=nr_pic_plummer
S() { ssh -o BatchMode=yes -o ConnectTimeout=30 $H "$@" 2>/dev/null; }

S "mkdir -p $R/runs/$LABEL && touch $R/runs/$LABEL/.protect_all_rst"
# A cancelled segment leaves no verdict banner. Distinguish that from a genuine failure by
# reporting it and stopping, which is what the operator wants either way.

for seg in $(seq 1 "$MAXSEG"); do
  # Adopt an already-queued or running segment rather than submitting a duplicate: the
  # first segment is often launched by hand, and two concurrent segments writing the same
  # run directory would corrupt it.
  j=$(S "squeue -h -u jiaxiwu -o '%i %j' | awk -v L=$LABEL '\$2 == L {print \$1; exit}'")
  if [ -n "$j" ]; then
    echo "PROD: segment $seg adopting the existing job $j"
  else
    j=$(S "cd $R && sbatch -t ${SEGH}:00:00 -J $LABEL scripts/amd_run.sbatch $LABEL $DECK" \
          | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
    if [ -z "$j" ]; then echo "PROD: segment $seg SUBMIT FAILED"; exit 1; fi
    echo "PROD: segment $seg submitted as job $j (${SEGH} h)"
  fi
  for i in $(seq 1 2000); do
    S "squeue -h -j $j -o %T" | grep -q . || break
    sleep 60
  done
  L=$R/logs/$LABEL.$j.log
  v=$(S "grep -aoE 'CASE (DONE|FAILED|INCOMPLETE) \($LABEL\)[^,]{0,60}' $L | tail -1")
  t=$(S "grep -aoE 'time=[0-9.e+-]+ cycle=[0-9]+' $L | tail -1")
  sz=$(S "du -sh $R/runs/$LABEL 2>/dev/null | cut -f1")
  u=$(S "du -sh /work1/eliasmost/jiaxiwu 2>/dev/null | cut -f1")
  echo "PROD: segment $seg job $j -> ${v:-no verdict}; last $t; run $sz; user root $u"
  # prune: keep the two newest and every integer-P_1/2 checkpoint (index % 4 == 0)
  S "cd $R/runs/$LABEL/out/rst 2>/dev/null && ls -1t *.rst 2>/dev/null | tail -n +3 |
     while read -r f; do
       idx=\$(echo \"\$f\" | grep -oE '[0-9]{5}' | tail -1)
       if [ -n \"\$idx\" ] && [ \$((10#\$idx % 4)) -eq 0 ]; then
         echo \"KEEP integer-P checkpoint \$f\"
       else
         rm -f \"\$f\"
       fi
     done"
  wd=$(S "grep -ac 'WATCHDOG CANCEL' $L | head -1"); wd=${wd//[^0-9]/}
  case "$v" in
    *"CASE DONE"*)     echo "PROD: reached tlim after $seg segment(s)"; break ;;
    *"CASE FAILED"*)   echo "PROD: segment $seg FAILED, stopping the chain"
                       S "grep -aiE '### FATAL|Memory access fault|nan|Terminating' $L | head -10"
                       exit 1 ;;
    *"CASE INCOMPLETE"*) : ;;                       # hit the wall limit; resubmit
    *)
      # No verdict banner. The common benign cause is the storage watchdog cancelling the
      # job because the SHARED user root crossed 1.7 TiB -- which other campaigns can
      # cause. Restart continuity is verified bit-continuous, so the right response is to
      # wait for headroom and resume from the last checkpoint, not to abandon the run.
      if [ -n "$wd" ] && [ "$wd" -gt 0 ] 2>/dev/null; then
        echo "PROD: segment $seg was cancelled by the storage watchdog (shared user root)."
        S "grep -a 'WATCHDOG CANCEL' $L | tail -2"
        for w in $(seq 1 96); do
          uk=$(S "du -sk /work1/eliasmost/jiaxiwu 2>/dev/null | awk '{print \$1}'")
          uk=${uk//[^0-9]/}
          [ -z "$uk" ] && { sleep 600; continue; }
          if [ "$uk" -lt 1400000000 ] 2>/dev/null; then
            echo "PROD: user root down to $((uk/1048576)) GiB, resuming the chain"
            break
          fi
          [ "$w" = "1" ] && echo "PROD: waiting for the shared user root to fall below 1.3 TiB (now $((uk/1048576)) GiB)"
          sleep 600
        done
        continue
      fi
      echo "PROD: segment $seg produced no verdict banner and no watchdog cancel, stopping"
      exit 1 ;;
  esac
done
echo "PROD: PRODUCTION CHAIN COMPLETE ($LABEL)"
