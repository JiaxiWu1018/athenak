#!/bin/bash
#SBATCH --job-name=pl_s3_ledger_t0
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005/logs/ledger_t0_%j.log
set -euo pipefail
session=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:${LD_LIBRARY_PATH:-}
run=$session/runs/cpu_ledger_t0_$SLURM_JOB_ID
mkdir -p "$run/out"
cd "$run"
sha256sum "$session/initial_data/build_cpu/src/athena"
"$session/initial_data/build_cpu/src/athena" -i "$session/inputs/main_live.athinput" -d "$run/out" time/nlim=0 output1/dt=0 output2/dt=0 output3/dt=0 output4/dt=0 output5/dt=0 output7/dt=0 > "segment_$SLURM_JOB_ID.log" 2>&1
python3 "$session/analysis/health.py" "$run"
python3 "$session/analysis/check_ledger.py" "$run" "$session/evidence/ledger_t0.json"
