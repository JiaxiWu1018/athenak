#!/bin/bash
set -uo pipefail
CONFIG=${1:?config required}
SEGMENT=${2:?segment number required}
HANDOFF=$(python3 -c 'import json, os, sys; print(os.path.expanduser(os.path.expandvars(json.load(open(sys.argv[1]))["handoff_dir"])))' "$CONFIG")
SCRIPT_DIR=$HANDOFF/scripts
WF=$SCRIPT_DIR/workflow.py
test -f "$WF"
get() { python3 "$WF" --config "$CONFIG" config-get "$1"; }
test "$(get handoff_dir)" = "$HANDOFF"
WORK_ROOT=$(get work_root)
SCRATCH_ROOT=$(get scratch_root)
CID=$(get campaign_id)
MODULES=$(get modules)
UCX_GDR=$(get ucx_ib_gpu_direct_rdma)
EXPECT_HEAD=$(get source_commit)
EXPECT_INPUT=$(get input_sha256)
INPUT_REL=$(get input_relpath)
ATHENA_WALL=$(get athena_wall)
NODES=$(get production_nodes)
MAX_NMB=$(get production_max_nmb_per_rank)
MIN_CHECKPOINT_BYTES=$(get minimum_checkpoint_bytes)
CAMPAIGN=$SCRATCH_ROOT/$CID
readarray -t OPTIONAL < <(python3 - "$CONFIG" "$CAMPAIGN" <<'PY'
import json, os, sys
cfg=json.load(open(sys.argv[1]))
def value(key, default=""):
    item=cfg.get(key, default)
    return os.path.expanduser(os.path.expandvars(item)) if isinstance(item, str) else item
print(value("initial_checkpoint"))
print(value("initial_checkpoint_sha256"))
print(value("initial_checkpoint_bytes"))
print(value("storage_accounting_root", sys.argv[2]))
PY
)
INITIAL_CHECKPOINT=${OPTIONAL[0]}
INITIAL_CHECKPOINT_SHA256=${OPTIONAL[1]}
INITIAL_CHECKPOINT_BYTES=${OPTIONAL[2]}
ACCOUNTING_ROOT=${OPTIONAL[3]}
RUN=$CAMPAIGN/segments/segment_$(printf '%03d' "$SEGMENT")
CONTROL=$WORK_ROOT/control
EXE=$WORK_ROOT/build_cuda_gi/src/athena
DECK=$HANDOFF/$INPUT_REL
LOG=$RUN/athena.log
python3 "$WF" --config "$CONFIG" gate-check verify build quicktest
test -x "$EXE"
test "$(git -C "$WORK_ROOT/source/athenak" rev-parse HEAD)" = "$EXPECT_HEAD"
test "$(sha256sum "$DECK" | awk '{print $1}')" = "$EXPECT_INPUT"
test ! -e "$RUN" || { echo "FATAL: segment directory already exists: $RUN"; exit 3; }
mkdir -p "$RUN/out"
cp "$DECK" "$RUN/canonical.athinput"
STARTED=$(date +%s)
module reset >/dev/null 2>&1
module load $MODULES
export UCX_IB_GPU_DIRECT_RDMA=$UCX_GDR
export OMP_NUM_THREADS=1
python3 "$SCRIPT_DIR/monitor.py" --config "$CONFIG" --run "$RUN/out" \
  --campaign-root "$ACCOUNTING_ROOT" --log "$LOG" --started-epoch "$STARTED" &
MONITOR_PID=$!
if [ "$SEGMENT" -eq 1 ]; then
  if [ -n "$INITIAL_CHECKPOINT" ]; then
    test -f "$INITIAL_CHECKPOINT"
    test "$(stat -c %s "$INITIAL_CHECKPOINT")" = "$INITIAL_CHECKPOINT_BYTES"
    test "$(sha256sum "$INITIAL_CHECKPOINT" | awk '{print $1}')" = \
      "$INITIAL_CHECKPOINT_SHA256"
    LAUNCH=("$EXE" --kokkos-map-device-id-by=mpi_rank -r "$INITIAL_CHECKPOINT" \
      -i "$RUN/canonical.athinput" -d "$RUN/out" \
      "time/stop_file=$RUN/out/REQUEST_STOP" \
      "mesh_refinement/max_nmb_per_rank=$MAX_NMB")
  else
    LAUNCH=("$EXE" --kokkos-map-device-id-by=mpi_rank -i "$RUN/canonical.athinput" \
      -d "$RUN/out" "time/stop_file=$RUN/out/REQUEST_STOP" \
      "mesh_refinement/max_nmb_per_rank=$MAX_NMB")
  fi
else
  RST=$(python3 "$WF" --config "$CONFIG" config-get work_root)/control/state.json
  CHECKPOINT=$(python3 - "$RST" <<'PY'
import json,sys
state=json.load(open(sys.argv[1]))
value=state.get("latest_checkpoint")
if value:
    print(value)
elif int(state.get("segments_completed", 0)) == 0:
    print("FRESH_INFRA_RETRY")
else:
    raise SystemExit("missing latest_checkpoint after completed progress")
PY
)
  if [ "$CHECKPOINT" = FRESH_INFRA_RETRY ]; then
    LAUNCH=("$EXE" --kokkos-map-device-id-by=mpi_rank -i "$RUN/canonical.athinput" \
    -d "$RUN/out" "time/stop_file=$RUN/out/REQUEST_STOP" \
    "mesh_refinement/max_nmb_per_rank=$MAX_NMB")
  else
    test -s "$CHECKPOINT"
    LAUNCH=("$EXE" --kokkos-map-device-id-by=mpi_rank -r "$CHECKPOINT" \
      -i "$RUN/canonical.athinput" -d "$RUN/out" \
      "time/stop_file=$RUN/out/REQUEST_STOP" \
      "mesh_refinement/max_nmb_per_rank=$MAX_NMB")
  fi
fi
{
  echo "host=$(hostname)"
  echo "job_id=${SLURM_JOB_ID:-none}"
  echo "segment=$SEGMENT nodes=${SLURM_NNODES:-none} tasks=${SLURM_NTASKS:-none} max_nmb_per_rank=$MAX_NMB"
  echo "UCX_IB_GPU_DIRECT_RDMA=$UCX_IB_GPU_DIRECT_RDMA"
  echo "started=$(date -Is)"
  echo "command=ibrun ${LAUNCH[*]} -t $ATHENA_WALL"
} > "$RUN/provenance.txt"
set +e
(
  cd "$RUN/out"
  ibrun "${LAUNCH[@]}" -t "$ATHENA_WALL"
) 2>&1 | tee "$LOG"
APP_RC=${PIPESTATUS[0]}
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
python3 "$SCRIPT_DIR/segment_finalize.py" --run "$RUN" \
  --campaign-root "$ACCOUNTING_ROOT" --segment "$SEGMENT" --exit-code "$APP_RC" \
  --nodes "$NODES" --started-epoch "$STARTED" --log "$LOG" \
  --telemetry "$CONTROL/telemetry.json" \
  --minimum-checkpoint-bytes "$MIN_CHECKPOINT_BYTES"
FINAL_RC=$?
set -e
echo "completed=$(date -Is) app_exit=$APP_RC finalize_exit=$FINAL_RC" >> "$RUN/provenance.txt"
if [ "$APP_RC" -ne 0 ]; then exit "$APP_RC"; fi
exit "$FINAL_RC"
