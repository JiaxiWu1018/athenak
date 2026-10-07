#!/bin/bash
#SBATCH -J jeans9-input-preflight
#SBATCH -A eliasmost
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH --no-requeue
set -euo pipefail
source /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/amd_env.inc
python3 "$ROOT/scripts/workflow.py" bindings
python3 "$ROOT/scripts/mpi_nodes.py" --witness-only
python3 "$ROOT/scripts/input_contract.py" "$DECK"
python3 -m unittest discover -s "$ROOT/scripts" -p test_session9.py
python3 -m unittest discover -s "$ROOT/scripts" -p test_r40_operations.py
mkdir -p "$ROOT/evidence/r40_preflight_$SLURM_JOB_ID"
PREFLIGHT=$ROOT/evidence/r40_preflight_$SLURM_JOB_ID
{ date -u; hostname; module list; sha256sum "$EXE" "$DECK"; } > "$PREFLIGHT/provenance.txt" 2>&1
common=(output7/last_time=0 time/ndiag=1 z4c/co_0_out_every=1 z4c/co_1_out_every=1 fastflow/start_time_0=0 fastflow/start_time_1=0 fastflow/start_time_2=0)
prun "$EXE" --kokkos-map-device-id-by=mpi_rank -n -i "$DECK" "${common[@]}" time/nlim=2 time/stop_file="$ROOT/control/REQUEST_STOP" > "$PREFLIGHT/full_input.log" 2>&1
prun "$EXE" --kokkos-map-device-id-by=mpi_rank -n -i "$DECK" "${common[@]}" time/nlim=1 output6/dt=0 output18/dt=0 output19/dt=0 output20/dt=0 output21/dt=0 time/stop_check_every=1 time/stop_file="$ROOT/control/REQUEST_STOP" > "$PREFLIGHT/slim_input.log" 2>&1
prun "$EXE" --kokkos-map-device-id-by=mpi_rank -n -i "$DECK" time/tlim=12 time/stop_file="$ROOT/control/REQUEST_STOP" > "$PREFLIGHT/segment_input.log" 2>&1
python3 - <<'ENDPY'
import os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(os.environ['ROOT'])/'scripts'))
from runtime import atomic,digest
root=Path(os.environ['ROOT']);atomic(root/'evidence/r40_input_preflight.json',dict(passed=True,job=os.environ['SLURM_JOB_ID'],input_sha256=digest(root/'inputs/gi_cluster_s9.athinput'),utc=time.time(),method='Actual compiled AMD executable -n with full, slim and continuation overrides; no particles or evolution.'))
ENDPY
