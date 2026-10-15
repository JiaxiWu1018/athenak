#!/bin/bash
#SBATCH -J st_s05_finalqa
#SBATCH -N 1
#SBATCH -p mi2104x
#SBATCH -A eliasmost
#SBATCH -n 4
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --exclude=k003-002,k003-009,k003-010
#SBATCH -t 00:20:00
#SBATCH -o /work1/eliasmost/jiaxiwu/st_reconstruction_s05_20261014/logs/%x.%j.log
source /work1/eliasmost/jiaxiwu/st_reconstruction_s05_20261014/scripts/amd_common.sh
qa_repo=$1
git -C "$qa_repo" diff --exit-code HEAD
git -C "$qa_repo" rev-parse HEAD > "$ROOT/evidence/jobs/$SLURM_JOB_ID/qa_source_commit.txt"
sha256sum "$qa_repo/scripts/reconstruction_s05/qa_final.py" \
  "$qa_repo/scripts/reconstruction_s05/qa_final.sbatch" \
  "$qa_repo/scripts/reconstruction_s05/qa_reductions.py" \
  > "$ROOT/evidence/jobs/$SLURM_JOB_ID/qa_source_sha256.txt"
# The first3 actual cases already passed463822; validate only the remaining21.
python3 "$qa_repo/scripts/reconstruction_s05/qa_reductions.py" --cases \
  legacy_ablation_s1 legacy_mc_s2 legacy_stratified_s2 legacy_cell_s2 legacy_ablation_s2 \
  legacy_mc_s3 legacy_stratified_s3 legacy_cell_s3 legacy_ablation_s3 \
  proper_mc_s1 proper_stratified_s1 proper_cell_s1 proper_ablation_s1 \
  proper_mc_s2 proper_stratified_s2 proper_cell_s2 proper_ablation_s2 \
  proper_mc_s3 proper_stratified_s3 proper_cell_s3 proper_ablation_s3
python3 "$qa_repo/scripts/reconstruction_s05/qa_final.py"
