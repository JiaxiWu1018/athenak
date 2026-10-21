#!/bin/bash
#SBATCH -J st_s08_cuda_checks
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 1
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH -t 00:30:00
#SBATCH --no-requeue
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/perseus/logs/%x.%j.log
set -euo pipefail
source /data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/src/athenak/scripts/reconstruction_s08/perseus_env.sh
record=$ROOT/evidence/jobs/$SLURM_JOB_ID
mkdir -p "$record"
{ date -u; hostname; scontrol show job "$SLURM_JOB_ID"; mpicxx --showme; nvidia-smi --query-gpu=uuid,name,memory.total --format=csv; sha256sum "$ROOT/protected/athena" "$ROOT/samples/frozen_ah.bin"; } > "$record/environment.txt" 2>&1
export FROZEN_WORK=$ROOT/tests/frozen_ah_$SLURM_JOB_ID
mkdir "$FROZEN_WORK"
mpirun --bind-to none -np 2 "$ROOT/protected/athena" --kokkos-map-device-id-by=mpi_rank -i "$ROOT/inputs/frozen_ah.athinput" -d "$FROZEN_WORK" fastflow/verbose=true "problem/cluster_t0_dump=$FROZEN_WORK/initial_particles.bin" > "$record/frozen_ah.log" 2>&1
test -f "$FROZEN_WORK/FROZEN_AH_OK"
python3 - <<'PYCODE'
from common import *
atomic(ROOT/'evidence/BUILD_AND_FROZEN_AH_OK.json',dict(utc=now(),job=job(),executable=receipt(ROOT/'protected/athena'),frozen_check=receipt(Path(os.environ['FROZEN_WORK'])/'FROZEN_AH_OK')))
PYCODE
python3 "$ROOT/scripts/native_checks.py" > "$record/native_checks.log" 2>&1
