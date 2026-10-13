#!/bin/bash
set -euo pipefail
CONFIG=${1:?usage: vista_resource_probe.sbatch workflow_probe_config.json}
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
EXE=$WORK_ROOT/build_cuda_gi/src/athena
DECK=$(get handoff_dir)/$(get input_relpath)
RUN=$SCRATCH_ROOT/$CID/resource_probe_${SLURM_JOB_ID:?}
RANKS=4
CAPS=${RESOURCE_CAPS:-"360 400 440 480 520"}
python3 "$WF" --config "$CONFIG" gate-check verify build
test -x "$EXE"
test ! -e "$RUN" || { echo "FATAL: resource-probe directory exists: $RUN"; exit 3; }
mkdir -p "$RUN"
module reset >/dev/null 2>&1
module load $MODULES
export UCX_IB_GPU_DIRECT_RDMA=$UCX_GDR
export OMP_NUM_THREADS=1
{
  echo "UCX_IB_GPU_DIRECT_RDMA=$UCX_IB_GPU_DIRECT_RDMA"
  ucx_info -v
} > "$RUN/transport.txt"
printf 'capacity\texit_code\n' > "$RUN/cases.tsv"
DISABLE_OUTPUTS=()
for i in $(seq 1 21); do DISABLE_OUTPUTS+=("output${i}/dt=-1"); done
for cap in $CAPS; do
  case_dir=$RUN/cap_$cap
  mkdir -p "$case_dir/out"
  stop=$RUN/.stop_gpu_sampler_cap${cap}
  srun --overlap --nodes="$RANKS" --ntasks="$RANKS" --ntasks-per-node=1 \
    bash -c 'out="$1/gpu_memory_cap'"$cap"'_$(hostname).csv"; while [ ! -e "$2" ]; do nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader,nounits >> "$out"; sleep 2; done' \
    _ "$RUN" "$stop" &
  sampler=$!
  ready=0
  for ((attempt=0; attempt<60; ++attempt)); do
    ready=$(find "$RUN" -maxdepth 1 -name "gpu_memory_cap${cap}_*.csv" | wc -l)
    [ "$ready" -ge "$RANKS" ] && break
    sleep 1
  done
  if [ "$ready" -lt "$RANKS" ]; then
    touch "$stop"
    wait "$sampler" || true
    printf '%s\t%s\n' "$cap" 98 >> "$RUN/cases.tsv"
    continue
  fi
  set +e
  (
    cd "$case_dir"
    ibrun -n "$RANKS" "$EXE" --kokkos-map-device-id-by=mpi_rank -i "$DECK" \
      -d "$case_dir/out" time/nlim=1 z4c/nrad_wave_extraction=0 \
      "mesh_refinement/max_nmb_per_rank=$cap" "${DISABLE_OUTPUTS[@]}"
  ) > "$RUN/cap_${cap}.log" 2>&1
  rc=$?
  set -e
  touch "$stop"
  wait "$sampler" || true
  printf '%s\t%s\n' "$cap" "$rc" >> "$RUN/cases.tsv"
done
python3 "$SCRIPT_DIR/summarize_resource_probe.py" --root "$RUN" --ranks "$RANKS" \
  --output "$RUN/resource_selection.json"
echo "RESOURCE_SELECTION=$RUN/resource_selection.json"
