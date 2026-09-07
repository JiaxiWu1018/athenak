#!/bin/bash
# Progress watcher for the Plummer production run.
#
#   usage: watch_production.sh [LABEL] [REPORT_STEP_IN_P] [POLL_SECONDS]
#
# Emits one line per REPORT_STEP of t/P_1/2, one line if a genuine failure signature
# appears, and one line when t first crosses P_1/2 (at which point it runs the interim
# reduction). Coverage note: the failure grep must match every terminal state, not just
# the happy path -- a watcher that only greps for progress is silent through a crash.
#
# NOTE ON grep -c: it exits 1 when the count is zero, so `grep -c ... || echo 0` appends a
# second "0" and any test against the result misfires. Use `|| true` and read the count.
set -uo pipefail
LABEL=${1:-prod_plummer}
STEP=${2:-0.25}
POLL=${3:-300}
H=hpcfund.amd.com
R=/work1/eliasmost/jiaxiwu/plummer_s01_20260906
P=1192.496781
S() { ssh -o BatchMode=yes -o ConnectTimeout=30 $H "$@" 2>/dev/null || true; }

last=-1
fired=0
absent=0
for i in $(seq 1 900); do
  # Resolve the CURRENT job id, then use ITS log. Taking the newest log file instead means
  # that after a failure the watcher reads the FAILED job's log and exits immediately on a
  # perfectly healthy resumed run -- which is exactly what happened on the first resume.
  jid=$(ssh -o BatchMode=yes -o ConnectTimeout=30 "$H" \
        "squeue -h -u jiaxiwu -o '%i %j' | awk -v L=$LABEL '\$2 == L {print \$1; exit}'" 2>/dev/null)
  if [ -n "$jid" ]; then
    L="$R/logs/$LABEL.$jid.log"
  else
    L=$(S "ls -1t $R/logs/$LABEL.*.log 2>/dev/null | head -1")
  fi
  if [ -n "$L" ]; then
    bad=$(S "grep -acE '### FATAL ERROR|Memory access fault|CASE FAILED|Segmentation|nan detected' '$L' | head -1")
    bad=${bad//[^0-9]/}
    if [ -n "$bad" ] && [ "$bad" -gt 0 ] 2>/dev/null; then
      echo "PRODUCTION FAILURE SIGNATURE in $L:"
      S "grep -aE '### FATAL ERROR|Memory access fault|CASE FAILED|Segmentation' '$L' | head -6"
      break
    fi
    line=$(S "grep -aE '^elapsed=' '$L' | tail -1")
    t=$(echo "$line" | grep -oE 'time=[0-9.eE+-]+' | cut -d= -f2 | head -1)
    if [ -n "$t" ]; then
      step=$(python3 -c "print(int($t/$P/$STEP))" 2>/dev/null || echo -1)
      if [ "$step" -gt "$last" ] 2>/dev/null; then
        last=$step
        nf=$(S "ls -1 $R/runs/$LABEL/out/pvtk/*.part.vtk 2>/dev/null | wc -l")
        sz=$(S "du -sh $R/runs/$LABEL 2>/dev/null | cut -f1")
        ur=$(S "du -sh /work1/eliasmost/jiaxiwu 2>/dev/null | cut -f1")
        echo "PROD $LABEL: t=$t M  t/P=$(python3 -c "print(f'{$t/$P:.4f}')")  $(echo "$line" | grep -oE 'cycle=[0-9]+')  pvtk=$nf  run=$sz  userroot=$ur"
        # Prune restarts DURING the segment, not only between segments. At the P_1/2/4
        # cadence a 12 h segment accumulates ~9 checkpoints of 5.3 GB, and the 1.9 TiB
        # budget is shared with another campaign whose growth we do not control. Keep the
        # two NEWEST (never touch a file that may still be being written) plus every
        # integer-P_1/2 checkpoint (rst dt = P/4, so index % 4 == 0).
        pruned=$(S "cd $R/runs/$LABEL/out/rst 2>/dev/null || exit 0
          n=\$(ls -1 *.rst 2>/dev/null | wc -l); [ \"\$n\" -le 4 ] && exit 0
          ls -1t *.rst 2>/dev/null | tail -n +3 | while read -r f; do
            idx=\$(echo \"\$f\" | grep -oE '[0-9]{5}' | tail -1)
            if [ -n \"\$idx\" ] && [ \$((10#\$idx % 4)) -eq 0 ]; then continue; fi
            rm -f \"\$f\" && echo \"\$f\"
          done")
        [ -n "$pruned" ] && echo "PROD $LABEL: pruned superseded checkpoints: $(echo $pruned | tr '\n' ' ')"
      fi
      if [ "$fired" = "0" ] && python3 -c "import sys; sys.exit(0 if $t >= $P else 1)" 2>/dev/null; then
        fired=1
        echo "PROD $LABEL: t has passed P_1/2 -- running the interim reduction"
        /data/jiaxiwu/NRPIC/Plummer-cluster/scripts/analyze_production.sh "$LABEL" Phalf 0.25 1.0 2>&1 | tail -30
        echo "PROD $LABEL: INTERIM P_1/2 ANALYSIS COMPLETE"
      fi
    fi
  fi
  # Same care as the chain driver: an ssh failure is not evidence the job ended. Check
  # ssh's exit status, ignore failed probes, require three consecutive absences.
  qout=$(ssh -o BatchMode=yes -o ConnectTimeout=30 "$H" "squeue -h -u jiaxiwu -o %j" 2>/dev/null)
  if [ $? -eq 0 ]; then
    if printf '%s\n' "$qout" | grep -qx "$LABEL"; then absent=0; else absent=$((absent + 1)); fi
    if [ "$absent" -ge 3 ]; then
      echo "PROD $LABEL: no job of that name in the queue (segment ended or chain finished)"
      S "grep -aoE 'CASE (DONE|FAILED|INCOMPLETE) \($LABEL\)[^,]{0,60}' '$L' | tail -1"
      break
    fi
  fi
  sleep "$POLL"
done
