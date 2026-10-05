#!/bin/bash
#SBATCH --job-name=pl_s3_t0_finish
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005/logs/cpu_t0_finish_%j.log
set -euo pipefail
ulimit -c 0
session=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:${LD_LIBRARY_PATH:-}
python3 "$session/analysis/check_sampling.py" "$session/runs/cpu_t0_frozen_11058" "$session/initial_data/cpp_profile.csv" "$session/evidence/sampling_cpu_frozen.json"
run=$session/runs/cpu_t0_live_$SLURM_JOB_ID
mkdir -p "$run/out"
cd "$run"
sha256sum "$session/initial_data/build_cpu/src/athena"
"$session/initial_data/build_cpu/src/athena" -i "$session/inputs/main_live.athinput" -d "$run/out" time/nlim=0 > "$run/segment_$SLURM_JOB_ID.log" 2>&1
python3 "$session/analysis/check_sampling.py" "$run" "$session/initial_data/cpp_profile.csv" "$session/evidence/sampling_cpu_live.json"
