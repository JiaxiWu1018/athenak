#!/bin/bash
# Restart-aware AMD MI210 runner for the NRPIC Plummer campaign (session 1).
# Wrapper logic inherited from ST_migration/session_06 scripts/amd_prod.sbatch.
# usage: sbatch -t HH:MM:SS -J LABEL amd_run.sbatch LABEL DECK [athena overrides...]
#   LABEL  run subdirectory under $root/runs
#   DECK   basename (no .athinput) of a deck in $root/inputs
# Resubmitting the SAME command continues the run from the newest checkpoint.
#SBATCH -o /work1/eliasmost/jiaxiwu/plummer_s02_20260910/logs/%x.%j.log
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p mi2104x
#SBATCH --exclude=k003-002,k003-009,k003-010
#SBATCH --open-mode=append

set -uo pipefail
label=${1:?usage: LABEL DECK [overrides...]}
deckname=${2:?usage: LABEL DECK [overrides...]}
shift 2

root=/work1/eliasmost/jiaxiwu/plummer_s02_20260910
exe=$root/build_rocm641/src/athena
deck=$root/inputs/${deckname}.athinput
run=$root/runs/$label

module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3
module load rocm/6.4.1
# Preserved gate: OpenMPI vader single-copy is disabled on this stack.
export OMPI_MCA_btl_vader_single_copy_mechanism=none

[ -x "$exe" ] || { echo "FATAL: no executable $exe"; exit 3; }
[ -r "$deck" ] || { echo "FATAL: no deck $deck"; exit 3; }
mkdir -p "$run/out"
cd "$run"
[ -f deck.athinput ] || cp "$deck" deck.athinput
printf '%s\n' "$@" > overrides.txt
seg=$(cat .segment 2>/dev/null || echo 0)
seg=$((seg + 1))
echo "$seg" > .segment

# Hand Athena a wall-clock budget 15 min short of Slurm's so it can write a final
# checkpoint and exit cleanly instead of being killed.
lim=$(squeue -h -j "$SLURM_JOB_ID" -o %l)
lim_s=$(awk -F'[-:]' '{if(NF==4) print $1*86400+$2*3600+$3*60+$4; else if(NF==3) print $1*3600+$2*60+$3; else print $1*60+$2}' <<<"$lim")
budget=$((lim_s - 900)); [ "$budget" -gt 600 ] || budget=$((lim_s*8/10))
athtime=$(printf '%02d:%02d:%02d' $((budget/3600)) $(((budget%3600)/60)) $((budget%60)))
newest=$(ls -1t "$run/out/rst/"*.rst 2>/dev/null | head -1)

echo "=== $label segment $seg job $SLURM_JOB_ID host=$(hostname) date=$(date -Is) ==="
echo "=== Slurm $lim -> Athena -t $athtime; overrides: $* ==="
sha256sum "$exe"
cat "$root/build_rocm641/git-head.txt"
if [ -n "$newest" ]; then
  echo "=== RESTART from $newest ($(stat -c %s "$newest") B) ==="
  args=(-r "$newest")
else
  echo "=== FRESH START ==="
  args=(-i ./deck.athinput)
fi
cat ./deck.athinput

source "$root/scripts/amd_watchdog.inc"
start_watchdog "$root"

# Per-GPU memory poller: two earlier campaigns lost jobs to silent VRAM exhaustion
# (utilisation collapsing with no error message), so this is on the record every segment.
( set +e +o pipefail
  while true; do
    echo "VRAM $(date '+%F %T') $(rocm-smi --showmemuse --csv 2>/dev/null | tr '\n' ' ')"
    sleep 60
  done ) &
VRAM_PID=$!

rc=0
prun "$exe" --kokkos-map-device-id-by=mpi_rank -t "$athtime" \
  "${args[@]}" -d "$run/out" "$@" || rc=$?
kill ${WATCHDOG_PID} 2>/dev/null || true
kill ${VRAM_PID} 2>/dev/null || true

# keep only the two newest checkpoints unless the run declares them protected
if [ ! -f "$run/.protect_all_rst" ]; then
  nrst=$(ls -1t "$run/out/rst/"*.rst 2>/dev/null | wc -l)
  if [ "$nrst" -gt 2 ]; then
    ls -1t "$run/out/rst/"*.rst | tail -n +3 | while read -r f; do
      case "$(basename "$f")" in
        *.protected.rst) echo "KEEP (protected) $f" ;;
        *) echo "PRUNE $f ($(stat -c %s "$f") B)"; rm -f "$f" ;;
      esac
    done
  fi
fi

echo "=== segment $seg rc=$rc; size $(du -sh "$run" | cut -f1); rst kept $(ls -1 "$run/out/rst/"*.rst 2>/dev/null | wc -l) ==="
mylog=$root/logs/$SLURM_JOB_NAME.$SLURM_JOB_ID.log
if grep -q "Terminating on time limit" "$mylog"; then
  echo "CASE DONE ($label): reached tlim"
elif grep -q "Terminating on cycle limit" "$mylog"; then
  echo "CASE DONE ($label): reached nlim"
elif grep -q "Terminating on wall clock limit" "$mylog"; then
  echo "CASE INCOMPLETE ($label): resubmit the same command to continue"
else
  echo "CASE FAILED ($label): no termination banner, rc=$rc"
  [ "$rc" -eq 0 ] && rc=1
fi
exit "$rc"
