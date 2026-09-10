#!/usr/bin/env bash
# Session-2 production driver: run one case to 5 P_1/2 in exact one-period segments,
# analysing after every completed P_1/2.
#
#   drive_milestones.sh CASE [FIRST_K] [LAST_K] [SEGHOURS]
#     CASE      R10 | R6p5
#     FIRST_K   first milestone to work towards (default: resume from state)
#     LAST_K    last milestone (default 5)
#     SEGHOURS  Slurm wall time per job (default 4, the MI210 VRAM-safe length)
#
# WHY MILESTONES ARE EXACT RATHER THAN DETECTED.  main.cpp:288 applies
# ModifyFromCmdline AFTER loading the restart's parameter dump, so `time/tlim=X` on the
# command line overrides the deck on a restart; and mesh.cpp:668 clamps the final step
# with `if ((time < tlim) && (time + dt > tlim)) dt = tlim - time`, so the run lands on
# tlim EXACTLY.  Driver::Finalize then writes every output type including the restart
# (driver.cpp:683-689).  So submitting a segment with tlim = k*P_1/2 makes "milestone k
# reached" a discrete, deterministic event with a complete output set and a checkpoint at
# exactly that time -- no polling for "t has crossed k*P", which is what Session 1 did and
# which double-fired once and truncated its CSVs.
#
# A milestone may need more than one Slurm job: each job runs until the earlier of tlim
# and its wall-clock budget.  "CASE INCOMPLETE" means resubmit the identical command;
# "CASE DONE" means the milestone is reached.  The loop is idempotent, so re-running the
# driver after any interruption resumes correctly from the on-disk state.
set -uo pipefail

CASE=${1:?usage: CASE [FIRST_K] [LAST_K] [SEGHOURS]}
LAST_K=${3:-5}
SEGH=${4:-4}

S=/data/jiaxiwu/NRPIC/Plummer-cluster/session_02_compactness_scan_20260910
H=hpcfund.amd.com
R=/work1/eliasmost/jiaxiwu/plummer_s02_20260910
LABEL=prod_$CASE
DECK=${CASE}_prod
STATE=$S/state/$CASE
mkdir -p "$STATE" "$S/logs"

# An ssh failure returns nothing, which says NOTHING about the remote state.  Every
# caller must distinguish "command ran and said X" from "the probe failed".
Sx() { ssh -o BatchMode=yes -o ConnectTimeout=30 "$H" "$@" 2>/dev/null; }

P=$(python3 - "$CASE" <<'PY'
import json, sys
c = sys.argv[1]
print(repr(json.load(open(
  '/data/jiaxiwu/NRPIC/Plummer-cluster/session_02_compactness_scan_20260910'
  '/initial_data/reference_values_%s.json' % c))['P_half']))
PY
)
[ -n "$P" ] || { echo "FATAL: could not read P_half for $CASE"; exit 2; }
echo "DRIVE $CASE: P_1/2 = $P M, milestones 1..$LAST_K, ${SEGH} h Slurm segments"

FIRST_K=${2:-}
if [ -z "$FIRST_K" ]; then
  FIRST_K=1
  for k in $(seq 1 "$LAST_K"); do
    [ -f "$STATE/milestone_$k.done" ] && FIRST_K=$((k + 1))
  done
fi
echo "DRIVE $CASE: starting at milestone $FIRST_K"

# The driver owns checkpoint pruning, so disable the in-job prune.
Sx "mkdir -p $R/runs/$LABEL && touch $R/runs/$LABEL/.protect_all_rst" || true

for k in $(seq "$FIRST_K" "$LAST_K"); do
  if [ -f "$STATE/milestone_$k.done" ]; then
    echo "DRIVE $CASE: milestone $k already done, skipping"
    continue
  fi
  TLIM=$(python3 -c "print(repr($k*$P))")
  echo "=== $CASE milestone $k/$LAST_K : tlim = $TLIM M  ($(date -u +%FT%TZ)) ==="

  # up to 12 Slurm jobs per milestone; each is wall-clock bounded, not milestone bounded
  for attempt in $(seq 1 12); do
    # adopt an already-queued job for this label rather than submitting a duplicate
    j=$(Sx "squeue -h -u jiaxiwu -o '%i %j' | awk -v L=$LABEL '\$2 == L {print \$1; exit}'")
    j=${j//[^0-9]/}
    if [ -n "$j" ]; then
      echo "DRIVE $CASE: adopting queued job $j"
    else
      j=$(Sx "cd $R && sbatch --parsable -t ${SEGH}:00:00 -J $LABEL scripts/amd_run.sbatch $LABEL $DECK time/tlim=$TLIM" | tr -dc '0-9')
      [ -n "$j" ] || { echo "DRIVE $CASE: submit failed, retrying in 300 s"; sleep 300; continue; }
      echo "DRIVE $CASE: submitted job $j (milestone $k, attempt $attempt)"
      echo "$j" >> "$STATE/jobs.txt"
    fi

    # Wait for the job to leave the queue.  A failed probe is NOT evidence the job ended,
    # so require three consecutive successful absences before believing it.
    absent=0; sfail=0
    while true; do
      out=$(ssh -o BatchMode=yes -o ConnectTimeout=30 "$H" "squeue -h -j $j -o %T" 2>/dev/null)
      if [ $? -ne 0 ]; then
        sfail=$((sfail + 1))
        [ "$sfail" = 20 ] && echo "DRIVE $CASE: 20 consecutive ssh probe failures for $j; NOT assuming it ended"
        sleep 60; continue
      fi
      sfail=0
      if [ -z "$out" ]; then
        absent=$((absent + 1)); [ "$absent" -ge 3 ] && break
      else
        absent=0
      fi
      sleep 30
    done

    verdict=$(Sx "grep -hoE 'CASE (DONE|INCOMPLETE|FAILED)' $R/logs/$LABEL.$j.log | tail -1")
    t_now=$(Sx "tail -400 $R/logs/$LABEL.$j.log | grep -oE 'time=[0-9.e+-]+' | tail -1")
    echo "DRIVE $CASE: job $j verdict='${verdict:-unknown}' ${t_now:-}"

    case "$verdict" in
      "CASE DONE") break ;;
      "CASE INCOMPLETE") echo "DRIVE $CASE: wall clock reached, continuing milestone $k"; continue ;;
      *)
        # transient GPU OOM is retryable; a genuine numerical failure is not
        if Sx "grep -qE 'HSA_STATUS_ERROR_OUT_OF_RESOURCES|hipErrorOutOfMemory|out of memory' $R/logs/$LABEL.$j.log" \
           && ! Sx "grep -qE '### FATAL ERROR|Memory access fault|nan detected|Assertion' $R/logs/$LABEL.$j.log"; then
          echo "DRIVE $CASE: transient GPU OOM on $j, retrying"; sleep 120; continue
        fi
        echo "DRIVE $CASE: job $j FAILED for a non-transient reason; stopping. Inspect $R/logs/$LABEL.$j.log"
        exit 1 ;;
    esac
  done

  # prune: keep every integer-P_1/2 checkpoint (rst cadence is P/4, so index % 4 == 0)
  # plus the two newest, and drop the rest.
  Sx "cd $R/runs/$LABEL/out/rst 2>/dev/null && ls -1t *.rst 2>/dev/null | tail -n +3 | while read -r f; do
        idx=\$(echo \"\$f\" | grep -oE '[0-9]{5}' | tail -1)
        if [ -n \"\$idx\" ] && [ \$((10#\$idx % 4)) -eq 0 ]; then echo \"KEEP \$f\"; else echo \"PRUNE \$f\"; rm -f \"\$f\"; fi
      done" || true

  touch "$STATE/milestone_$k.done"
  echo "DRIVE $CASE: milestone $k REACHED at t = $TLIM M; running the milestone analysis"
  "$S/scripts/analyze_milestone.sh" "$CASE" "$k" 2>&1 | tee -a "$S/logs/analyze_${CASE}_P${k}.log"
  echo "DRIVE $CASE: milestone $k analysis complete ($(date -u +%FT%TZ))"
done

echo "DRIVE $CASE: ALL MILESTONES 1..$LAST_K COMPLETE"
