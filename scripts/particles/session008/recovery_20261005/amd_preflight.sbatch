#!/bin/bash
#SBATCH -A eliasmost
#SBATCH -p mi2104x
#SBATCH -N 3
#SBATCH -n 12
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --exclude=k002-[005-006],k003-[001-006,008-010],k005-[006,008-010]
#SBATCH -t 00:10:00
#SBATCH --no-requeue
set -euo pipefail
source /work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/scripts/amd_env.inc
RECOVERY=$ROOT/scripts/recovery_20261005/recovery.py
python3 "$RECOVERY" verify
export RUN=$ROOT/runs/recovery_mesh_20261005
test ! -e "$RUN"
mkdir -p "$RUN"
{
 date -u; hostname; module list; scontrol show job "$SLURM_JOB_ID"
 git -C "$ROOT/athenak" rev-parse HEAD; git -C "$ROOT/athenak" submodule status
} > "$RUN/provenance.txt" 2>&1
rc=0
(cd "$RUN"; prun "$ROOT/scripts/amd_rank_wrapper.sh" "$EXE" --kokkos-map-device-id-by=mpi_rank -i "$DECK" -m) > "$RUN/run.log" 2>&1 || rc=$?
printf '%s\n' "$rc" > "$RUN/EXIT_CODE"
python3 "$RECOVERY" preflight --rc "$rc"
