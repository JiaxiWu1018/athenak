#!/bin/bash
# Perseus-side driver for the Plummer production run: 0 -> 3 P_1/2 = 3577.490343 M,
# chained restart segments on one AMD MI210 node.
#
#   usage: drive_production.sh [SEGMENT_HOURS] [MAX_SEGMENTS] [LABEL] [DECK] [overrides...]
#
# LABEL/DECK default to the production run; passing them lets the same chaining,
# pruning and failure-classification logic drive any other long run on this node
# (it is used to extend the frozen-metric null control to the production baseline).
# Any further arguments are passed to Athena verbatim as parameter overrides.
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
# Shorter segments deliberately bound the per-process VRAM growth: a fresh process resets
# the allocation, and restart continuity is verified bit-continuous, so segmenting is
# almost free. 12 h segments reached 70 % VRAM; 4 h keeps it near 57 %.
SEGH=${1:-4}
MAXSEG=${2:-16}
LABEL=${3:-prod_plummer}
DECK=${4:-nr_pic_plummer}
shift $(( $# < 4 ? $# : 4 ))
OVR="$*"
rtry=0
H=hpcfund.amd.com
R=/work1/eliasmost/jiaxiwu/plummer_s01_20260906
S() { ssh -o BatchMode=yes -o ConnectTimeout=30 $H "$@" 2>/dev/null; }

S "mkdir -p $R/runs/$LABEL && touch $R/runs/$LABEL/.protect_all_rst"
# A cancelled segment leaves no verdict banner. A storage-watchdog cancellation is
# distinguished below and resumed from the last checkpoint; anything else stops the chain.

for seg in $(seq 1 "$MAXSEG"); do
  # Adopt an already-queued or running segment rather than submitting a duplicate: the
  # first segment is often launched by hand, and two concurrent segments writing the same
  # run directory would corrupt it.
  j=$(S "squeue -h -u jiaxiwu -o '%i %j' | awk -v L=$LABEL '\$2 == L {print \$1; exit}'")
  if [ -n "$j" ]; then
    echo "PROD: segment $seg adopting the existing job $j"
  else
    # --parsable prints the bare job id. Scraping the id out of the human banner with a
    # loose digit regex once matched an unrelated 8-digit number in the allocation header
    # and gave a later segment a dependency on a job that never existed.
    j=$(S "cd $R && sbatch --parsable -t ${SEGH}:00:00 -J $LABEL scripts/amd_run.sbatch $LABEL $DECK $OVR")
    j=${j%%;*}; j=${j//[^0-9]/}
    if [ -z "$j" ]; then echo "PROD: segment $seg SUBMIT FAILED"; exit 1; fi
    echo "PROD: segment $seg submitted as job $j (${SEGH} h)"
  fi
  # Wait for the segment. A transient ssh failure returns nothing, which says NOTHING
  # about the job; treating empty output as "finished" ended this chain spuriously while
  # job 407606 was still running at cycle 3000. So check ssh's own exit status, ignore
  # failed probes entirely, and require three CONSECUTIVE successful probes reporting the
  # job absent (squeue forgets a finished job quickly, so one absence is not evidence).
  miss=0; sfail=0
  for i in $(seq 1 4000); do
    out=$(ssh -o BatchMode=yes -o ConnectTimeout=30 "$H" "squeue -h -j $j -o %T" 2>/dev/null)
    if [ $? -ne 0 ]; then
      sfail=$((sfail + 1))
      [ "$sfail" = "20" ] && echo "PROD: 20 consecutive ssh probe failures for job $j; not assuming it ended"
      sleep 60
      continue
    fi
    sfail=0
    if [ -n "$out" ]; then miss=0; else miss=$((miss + 1)); fi
    [ "$miss" -ge 3 ] && break
    sleep 30
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
    *"CASE FAILED"*)
      # Distinguish a TRANSIENT RESOURCE failure from a physics or code failure. Job
      # 407606 died at cycle 7200 with
      #   HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Available Free mem : 19868 MB
      # after 10.5 h, with per-card VRAM having climbed 49 % -> 70 % over the segment and
      # the node still healthy afterwards. That is the VRAM-growth failure mode the
      # archive already records for this stack, and the correct response is to restart
      # from the last checkpoint in a FRESH process, which resets the allocation. A NaN,
      # a memory-access fault or a code assertion is different and must stop the chain.
      res=$(S "grep -acE 'HSA_STATUS_ERROR_OUT_OF_RESOURCES|hipErrorOutOfMemory|out of memory' $L | head -1")
      res=${res//[^0-9]/}
      hard=$(S "grep -acE '### FATAL ERROR|Memory access fault|nan detected|Assertion' $L | head -1")
      hard=${hard//[^0-9]/}
      if [ -n "$res" ] && [ "$res" -gt 0 ] 2>/dev/null && \
         { [ -z "$hard" ] || [ "$hard" -eq 0 ] 2>/dev/null; }; then
        rtry=$((rtry + 1))
        if [ "$rtry" -le 6 ]; then
          echo "PROD: segment $seg hit a transient GPU resource exhaustion (retry $rtry/6); resuming from the last checkpoint"
          continue
        fi
        echo "PROD: segment $seg hit GPU resource exhaustion and the retry budget is spent"
        exit 1
      fi
      echo "PROD: segment $seg FAILED for a non-resource reason, stopping the chain"
      S "grep -aiE '### FATAL ERROR|Memory access fault|nan|Assertion' $L | head -10"
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
