#!/bin/bash
set -euo pipefail
CONFIG=${1:?usage: vista_quicktest.sbatch SITE_CONFIG.json}
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
NODES=$(get startup_nodes)
ALT_RANKS=$(get restart_test_ranks)
NLIM=$(get quicktest_nlim)
PROD_CAP=$(get production_max_nmb_per_rank)
readarray -t PRODUCTION_RESTART < <(python3 - "$CONFIG" <<'PY'
import json, os, sys
cfg=json.load(open(sys.argv[1]))
for key in ("initial_checkpoint", "initial_checkpoint_sha256",
            "initial_checkpoint_bytes", "initial_cycle",
            "production_restart_smoke"):
    value=cfg.get(key, "")
    print(os.path.expanduser(os.path.expandvars(value)) if isinstance(value, str) else value)
PY
)
INITIAL_CHECKPOINT=${PRODUCTION_RESTART[0]}
INITIAL_CHECKPOINT_SHA256=${PRODUCTION_RESTART[1]}
INITIAL_CHECKPOINT_BYTES=${PRODUCTION_RESTART[2]}
INITIAL_CYCLE=${PRODUCTION_RESTART[3]}
PRODUCTION_RESTART_SMOKE=${PRODUCTION_RESTART[4]}
EXE=$WORK_ROOT/build_cuda_gi/src/athena
DECK=$HANDOFF/$(get input_relpath)
CONTROL_DECK=$HANDOFF/$(get control_input_relpath)
VALIDATION_DECK=$HANDOFF/inputs/VALIDATION_gi_cluster_s7_two_clump_ccw_con3d.athinput
VALIDATION_CONTROL=$HANDOFF/inputs/VALIDATION_CONTROL_gi_cluster_s7_two_clump_unboosted_con3d.athinput
RUN=$SCRATCH_ROOT/$CID/quicktest_${SLURM_JOB_ID:?}
DETAILS=$WORK_ROOT/control/quicktest_details.json
python3 "$WF" --config "$CONFIG" gate-check verify build
[ "$ALT_RANKS" -ge 1 ] && [ "$ALT_RANKS" -le "$NODES" ]
test "$(sha256sum "$CONTROL_DECK" | awk '{print $1}')" = "$(get control_input_sha256)"
test "$(sha256sum "$VALIDATION_DECK" | awk '{print $1}')" = e7b3930a25ef556951bea31aa82fd941b2fc163f97b168a5060362e142d729e3
test "$(sha256sum "$VALIDATION_CONTROL" | awk '{print $1}')" = bdd2344414533865a50d35e5990ccda894b9cdfeadac5fa67121fb634bd4cf3c
test ! -e "$RUN" || { echo "FATAL: quick-test directory exists: $RUN"; exit 3; }
mkdir -p "$RUN"
module reset >/dev/null 2>&1
module load $MODULES
python3 -c 'import numpy'
export UCX_IB_GPU_DIRECT_RDMA=$UCX_GDR
export OMP_NUM_THREADS=1
{
  echo "UCX_IB_GPU_DIRECT_RDMA=$UCX_IB_GPU_DIRECT_RDMA"
  ucx_info -v
} > "$RUN/transport.txt"
run_case() {
  local label=$1 ranks=$2; shift 2
  mkdir -p "$RUN/$label"
  local stop=$RUN/.stop_gpu_sampler_${label}
  srun --overlap --nodes="$ranks" --ntasks="$ranks" --ntasks-per-node=1 \
    bash -c 'out="$1/gpu_memory_'"$label"'_$(hostname).csv"; while [ ! -e "$2" ]; do nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader,nounits >> "$out"; sleep 5; done' \
    _ "$RUN" "$stop" &
  local sampler=$!
  local ready=0
  for ((i=0; i<60; ++i)); do
    ready=$(find "$RUN" -maxdepth 1 -name "gpu_memory_${label}_*.csv" | wc -l)
    [ "$ready" -ge "$ranks" ] && break
    sleep 1
  done
  [ "$ready" -ge "$ranks" ]
  set +e
  (
    cd "$RUN/$label"
    ibrun -n "$ranks" "$@"
  ) 2>&1 | tee "$RUN/${label}.log"
  local rc=${PIPESTATUS[0]}
  set -e
  touch "$stop"
  wait "$sampler"
  [ "$rc" -eq 0 ]
}
run_case parse 1 "$EXE" --kokkos-map-device-id-by=mpi_rank -i "$DECK" -n \
  mesh_refinement/max_nmb_per_rank=1600
(
  cd "$RUN"
  run_case mesh 1 "$EXE" --kokkos-map-device-id-by=mpi_rank -i "$DECK" -m \
    mesh_refinement/max_nmb_per_rank=1600
)
COMMON=(z4c/co_0_out_every=1 z4c/co_1_out_every=1 \
  fastflow/start_time_0=0 fastflow/start_time_1=0 fastflow/start_time_2=0)
run_case uninterrupted "$NODES" "$EXE" --kokkos-map-device-id-by=mpi_rank \
  -i "$VALIDATION_DECK" -d "$RUN/uninterrupted" "time/nlim=$NLIM" \
  "mesh_refinement/max_nmb_per_rank=$PROD_CAP" "${COMMON[@]}"
run_case production_output "$NODES" "$EXE" --kokkos-map-device-id-by=mpi_rank \
  -i "$DECK" -d "$RUN/production_output" time/nlim=1 \
  "mesh_refinement/max_nmb_per_rank=$PROD_CAP" "${COMMON[@]}"
run_case split_a "$NODES" "$EXE" --kokkos-map-device-id-by=mpi_rank \
  -i "$VALIDATION_DECK" -d "$RUN/split_a" time/nlim=1 \
  "mesh_refinement/max_nmb_per_rank=$PROD_CAP" "${COMMON[@]}"
RST=$(find "$RUN/split_a" -type f -name '*.rst' -size +0c -printf '%T@ %p\n' \
  | sort -n | tail -1 | cut -d' ' -f2-)
test -n "$RST" && test -s "$RST"
run_case restart_b "$ALT_RANKS" "$EXE" --kokkos-map-device-id-by=mpi_rank \
  -r "$RST" -i "$VALIDATION_DECK" -d "$RUN/restart_b" "time/nlim=$NLIM" \
  "mesh_refinement/max_nmb_per_rank=$PROD_CAP" "${COMMON[@]}"
mkdir -p "$RUN/clean_stop"
touch "$RUN/clean_stop/REQUEST_STOP"
run_case clean_stop "$ALT_RANKS" "$EXE" --kokkos-map-device-id-by=mpi_rank \
  -r "$RST" -i "$VALIDATION_DECK" -d "$RUN/clean_stop" "time/nlim=$NLIM" \
  time/stop_check_every=1 "time/stop_file=$RUN/clean_stop/REQUEST_STOP" \
  "mesh_refinement/max_nmb_per_rank=$PROD_CAP" \
  "${COMMON[@]}"
grep -q "Terminating on clean stop-file request" "$RUN/clean_stop.log"
find "$RUN/clean_stop" -type f -name '*.rst' -size +0c -print -quit | grep -q .
if [ -n "$INITIAL_CHECKPOINT" ]; then
  test "$PRODUCTION_RESTART_SMOKE" = True
  test -f "$INITIAL_CHECKPOINT"
  test "$(stat -c %s "$INITIAL_CHECKPOINT")" = "$INITIAL_CHECKPOINT_BYTES"
  test "$(sha256sum "$INITIAL_CHECKPOINT" | awk '{print $1}')" = \
    "$INITIAL_CHECKPOINT_SHA256"
  PRODUCTION_RESTART_NLIM=$((INITIAL_CYCLE + 1))
  run_case production_restart "$NODES" "$EXE" --kokkos-map-device-id-by=mpi_rank \
    -r "$INITIAL_CHECKPOINT" -i "$DECK" -d "$RUN/production_restart" \
    "time/nlim=$PRODUCTION_RESTART_NLIM" \
    "mesh_refinement/max_nmb_per_rank=$PROD_CAP" "${COMMON[@]}"
  PRODUCTION_RESTART_OUTPUT=$(find "$RUN/production_restart" -type f -name '*.rst' \
    -size +"$(get minimum_checkpoint_bytes)"c -printf '%T@ %p\n' \
    | sort -n | tail -1 | cut -d' ' -f2-)
  test -n "$PRODUCTION_RESTART_OUTPUT" && test -s "$PRODUCTION_RESTART_OUTPUT"
  run_case production_restart_parse 1 "$EXE" --kokkos-map-device-id-by=mpi_rank \
    -r "$PRODUCTION_RESTART_OUTPUT" -n
fi
run_case unboosted "$NODES" "$EXE" --kokkos-map-device-id-by=mpi_rank \
  -i "$VALIDATION_CONTROL" -d "$RUN/unboosted" "time/nlim=$NLIM" \
  "mesh_refinement/max_nmb_per_rank=$PROD_CAP" "${COMMON[@]}"
python3 "$SCRIPT_DIR/validate_initial_s7.py" \
  --boosted-run "$RUN/uninterrupted" --unboosted-run "$RUN/unboosted" \
  --boosted-log "$RUN/uninterrupted.log" --unboosted-log "$RUN/unboosted.log" \
  --athenak "$WORK_ROOT/source/athenak" --output "$RUN/initial_validation.json"
python3 "$SCRIPT_DIR/compare_restart_s7.py" \
  --reference "$RUN/uninterrupted" --restarted "$RUN/restart_b" \
  --athenak "$WORK_ROOT/source/athenak" --output "$RUN/restart_validation.json"
python3 "$SCRIPT_DIR/summarize_quicktest.py" --root "$RUN" --config "$CONFIG" \
  --output "$DETAILS"
python3 "$WF" --config "$CONFIG" mark quicktest pass --details "$DETAILS"
