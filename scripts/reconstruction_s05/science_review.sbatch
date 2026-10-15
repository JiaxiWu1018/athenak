#!/bin/bash
#SBATCH -J st_s05_science
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
review_repo=$1
shift
git -C "$review_repo" diff --exit-code HEAD
git -C "$review_repo" rev-parse HEAD > "$ROOT/evidence/jobs/$SLURM_JOB_ID/review_source_commit.txt"
sha256sum "$review_repo/scripts/reconstruction_s05/science_review.py" \
  "$review_repo/scripts/reconstruction_s05/science_review.sbatch" \
  > "$ROOT/evidence/jobs/$SLURM_JOB_ID/review_source_sha256.txt"
python3 "$review_repo/scripts/reconstruction_s05/science_review.py" "$@"
