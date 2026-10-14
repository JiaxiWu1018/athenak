#!/bin/bash
#SBATCH --job-name=eff-finish-link
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/data/jiaxiwu/NRPIC/paper/session_02_code_efficiency/logs/build.%j.log
set -euo pipefail
SESSION=${EFFICIENCY_SESSION:-/data/jiaxiwu/NRPIC/paper/session_02_code_efficiency}
export PATH=/home/emost/codes/openmpi/bin:/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
REC=$SESSION/provenance/build_$SLURM_JOB_ID
mkdir -p "$REC"
cd "$SESSION/build_release/src"
cmake -E cmake_link_script CMakeFiles/athena.dir/link.txt --verbose=1 > "$REC/release_link.log" 2>&1
mkdir -p "$SESSION/provenance/executables"
old_hash=$(sha256sum "$SESSION/binaries/release/athena" | cut -d' ' -f1)
if [ ! -f "$SESSION/provenance/executables/$old_hash" ]; then
  cp --reflink=auto "$SESSION/binaries/release/athena" "$SESSION/provenance/executables/$old_hash"
fi
cp athena "$SESSION/binaries/release/athena.$SLURM_JOB_ID"
mv "$SESSION/binaries/release/athena.$SLURM_JOB_ID" "$SESSION/binaries/release/athena"
sha256sum "$SESSION/binaries/release/athena" "$SESSION/binaries/profile/athena" > "$REC/executables.sha256"
printf '{"job_id":"%s","status":"built","release":"%s","profile":"%s"}\n' "$SLURM_JOB_ID" "$SESSION/binaries/release/athena" "$SESSION/binaries/profile/athena" > "$SESSION/provenance/build_receipt.json"
