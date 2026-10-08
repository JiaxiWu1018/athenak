#!/bin/bash
#SBATCH -J jeans9-gate
#SBATCH -A eliasmost
#SBATCH -p mi2104x
#SBATCH -N 12
#SBATCH -n 48
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --exclude=k003-[009-010]
#SBATCH --nodelist=k002-005,k003-[003-007],k005-[002-006,009]
#SBATCH -t 04:00:00
#SBATCH --no-requeue
set -euo pipefail
source /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/amd_env.inc
python3 -m unittest discover -s "$ROOT/scripts" -p test_session9.py
python3 -m py_compile "$ROOT/scripts/"*.py
python3 "$ROOT/scripts/input_contract.py" "$DECK"
python3 "$ROOT/scripts/workflow.py" bindings
python3 "$ROOT/scripts/mpi_nodes.py"
test ! -e "$ROOT/control/REQUEST_STOP"
test -x "$EXE"
resume=false
if [[ "${1:-}" == --resume-reference && $# -eq 1 ]]; then
 python3 "$ROOT/scripts/validate_reference.py" --receipt-only
 resume=true
elif [[ $# -ne 0 ]]; then
 exit 2
fi
{
 date -u; hostname; module list; scontrol show job "$SLURM_JOB_ID"
 sha256sum "$EXE" "$DECK"; git -C "$ROOT/athenak" rev-parse HEAD
 git -C "$ROOT/athenak" submodule status
} > "$ROOT/evidence/gate_provenance.txt" 2>&1
prun "$ROOT/build/mpi_probe" > "$ROOT/evidence/mpi_48rank_gate.txt" 2>&1
if ! "$resume"; then
 export RUN=$ROOT/runs/gate_mesh
 mkdir -p "$RUN"
 (cd "$RUN"; prun "$ROOT/scripts/amd_rank_wrapper.sh" "$EXE" --kokkos-map-device-id-by=mpi_rank -i "$DECK" -m) > "$RUN/run.log" 2>&1
 printf '0\n' > "$RUN/EXIT_CODE"
fi
common=(output7/last_time=0 time/ndiag=1 z4c/co_0_out_every=1 z4c/co_1_out_every=1 fastflow/start_time_0=0 fastflow/start_time_1=0 fastflow/start_time_2=0)
run_case() {
 export RUN=$ROOT/runs/$1; shift
 stop_path=$ROOT/control/REQUEST_STOP
 if [[ "$RUN" == "$ROOT/runs/gate_clean_stop" ]]; then stop_path=$RUN/REQUEST_STOP; fi
 mkdir -p "$RUN"
 slim=()
 if [[ "$RUN" != "$ROOT/runs/gate_reference" && "$RUN" != "$ROOT/runs/gate_output" ]]; then
  slim=(output6/dt=0 output18/dt=0 output19/dt=0 output20/dt=0 output21/dt=0)
 fi
 python3 "$ROOT/scripts/runtime.py" monitor --run "$RUN" &
 monitor=$!
 rc=0
 (cd "$RUN"; prun "$ROOT/scripts/amd_rank_wrapper.sh" "$EXE" --kokkos-map-device-id-by=mpi_rank "$@" -i "$DECK" -d "$RUN" -t 00:20:00 "${common[@]}" "${slim[@]}" time/stop_file="$stop_path") > "$RUN/run.log" 2>&1 || rc=$?
 kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true
 printf '%s\n' "$rc" > "$RUN/EXIT_CODE"
 test "$rc" -eq 0
 test ! -e "$ROOT/control/RESOURCE_STOP.json"
 test ! -e "$ROOT/control/USER_STOP"
 python3 "$ROOT/scripts/runtime.py" retain --run "$RUN"
 used=$(du -sx --block-size=1 "$ROOT" | awk '{print $1}')
 test "$used" -lt 1099511627776
}
if ! "$resume"; then
 run_case gate_reference time/nlim=2
 python3 "$ROOT/scripts/validate_s9.py" --root "$ROOT"
fi
run_case gate_split time/nlim=1
RST=$(python3 -c 'import json,os; from pathlib import Path; r=Path(os.environ["ROOT"]); d=json.loads((r/"control/checkpoints.json").read_text()); print(max((v for v in d.values() if "/gate_split/" in v["path"] and Path(v["path"]).exists()),key=lambda v:v["cycle"])["path"])')
run_case gate_restart -r "$RST" time/nlim=2
python3 "$ROOT/scripts/compare_restart_s7.py" --reference "$ROOT/runs/gate_reference" --restarted "$ROOT/runs/gate_restart" --athenak "$ROOT/athenak" --output "$ROOT/evidence/restart_validation.json"
RST=$(python3 -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["ROOT"])/"control/latest_checkpoint.json").read_text())["path"])')
mkdir -p "$ROOT/runs/gate_clean_stop"
touch "$ROOT/runs/gate_clean_stop/REQUEST_STOP"
for closed in gate_reference gate_split gate_restart; do python3 "$ROOT/scripts/runtime.py" seal --run "$ROOT/runs/$closed"; done
run_case gate_clean_stop -r "$RST" time/nlim=100000000 time/stop_check_every=1 time/stop_file="$ROOT/runs/gate_clean_stop/REQUEST_STOP"
grep -q 'Terminating on clean stop-file request' "$ROOT/runs/gate_clean_stop/run.log"
RST=$(python3 -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["ROOT"])/"control/latest_checkpoint.json").read_text())["path"])')
run_case gate_output -r "$RST" time/nlim=5 time/stop_file="$ROOT/control/REQUEST_STOP"
python3 "$ROOT/scripts/workflow.py" gate_receipt
for RUN in "$ROOT"/runs/gate_*; do
 python3 "$ROOT/scripts/runtime.py" seal --run "$RUN"
done
