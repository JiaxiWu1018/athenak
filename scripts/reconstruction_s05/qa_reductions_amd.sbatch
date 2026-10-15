#!/bin/bash
#SBATCH -J st_s05_reduceqa
#SBATCH -N 1
#SBATCH -p mi2104x
#SBATCH -A eliasmost
#SBATCH -n 4
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --exclude=k003-002,k003-009,k003-010
#SBATCH -t 00:10:00
#SBATCH -o /work1/eliasmost/jiaxiwu/st_reconstruction_s05_20261014/logs/%x.%j.log
source /work1/eliasmost/jiaxiwu/st_reconstruction_s05_20261014/scripts/amd_common.sh
qa_repo=$1
shift
git -C "$qa_repo" diff --exit-code HEAD
git -C "$qa_repo" rev-parse HEAD > "$ROOT/evidence/jobs/$SLURM_JOB_ID/qa_source_commit.txt"
sha256sum "$qa_repo/scripts/reconstruction_s05/qa_reductions.py" \
  "$qa_repo/scripts/reconstruction_s05/qa_reductions_amd.sbatch" \
  > "$ROOT/evidence/jobs/$SLURM_JOB_ID/qa_source_sha256.txt"
python3 "$qa_repo/scripts/reconstruction_s05/qa_reductions.py" "$@"
