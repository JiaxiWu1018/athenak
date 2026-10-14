#!/bin/bash
#SBATCH -J st_s05_cppqa
#SBATCH -p preempt
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014/logs/cppqa.%j.log
set -euo pipefail
root=/data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014
cmake -S "$root/src/athenak" -B "$root/build_cpu_qa" \
  -DAthena_ENABLE_MPI=OFF -DPROBLEM=particles/relativistic_cluster
cmake --build "$root/build_cpu_qa" -j2
{ git -C "$root/src/athenak" rev-parse HEAD; sha256sum "$root/build_cpu_qa/src/athena"; } > "$root/evidence/CPU_COMPILE_${SLURM_JOB_ID}.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$root/evidence/CPU_COMPILE_OK"
