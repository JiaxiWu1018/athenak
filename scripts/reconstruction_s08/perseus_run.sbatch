#!/bin/bash
#SBATCH -J st_s08_perseus_run
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 1
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH -t 04:00:00
#SBATCH --no-requeue
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/perseus/logs/%x.%j.log
set -euo pipefail
source /data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/src/athenak/scripts/reconstruction_s08/perseus_env.sh
record=$ROOT/evidence/jobs/$SLURM_JOB_ID
mkdir -p "$record"
start_epoch=$(date +%s)
finish() {
 status=$?
 if [ -n "${guard_pid:-}" ]; then kill "$guard_pid" 2>/dev/null || true; fi
 python3 - "$record" "$start_epoch" "$status" <<'PYCODE'
from common import *
import sys,time
r=Path(sys.argv[1]);elapsed=time.time()-int(sys.argv[2]);atomic(r/'ALLOCATION_END.json',dict(utc=now(),job=job(),elapsed_seconds=elapsed,allocated_nodes=int(os.environ['SLURM_NNODES']),gpus=2,cpus=2,returncode=int(sys.argv[3])))
PYCODE
}
trap finish EXIT
{ date -u; hostname; scontrol show job "$SLURM_JOB_ID"; mpicxx --showme; nvidia-smi --query-gpu=uuid,name,memory.total --format=csv; sha256sum "$ROOT/protected/athena"; git -C "$SESSION/src/athenak" rev-parse HEAD; } > "$record/environment.txt" 2>&1
python3 "$ROOT/scripts/perseus_resources.py" > "$record/resource_guard.log" 2>&1 &
guard_pid=$!
python3 "$ROOT/scripts/run_segment.py" "$1" --target 300 --wall-seconds 12600 > "$record/segment.log" 2>&1
# The same allocation performs the closed-output scientific audit.
python3 "$ROOT/scripts/reduce.py" "$1" --partial > "$record/audit.log" 2>&1
