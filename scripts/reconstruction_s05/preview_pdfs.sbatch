#!/bin/bash
#SBATCH -J st_s05_pdfreview
#SBATCH -p preempt
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gpus=1
#SBATCH --mem=4G
#SBATCH -t 00:15:00
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014/logs/pdfreview.%j.log
set -euo pipefail
export ROOT=/data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014
export LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
review_repo="$ROOT/src/athenak"
mkdir -p "$ROOT/evidence/jobs/$SLURM_JOB_ID"
git -C "$review_repo" diff --exit-code HEAD
git -C "$review_repo" rev-parse HEAD > "$ROOT/evidence/jobs/$SLURM_JOB_ID/pdf_source_commit.txt"
sha256sum "$review_repo/scripts/reconstruction_s05/preview_pdfs.py" \
  "$review_repo/scripts/reconstruction_s05/preview_pdfs.sbatch" \
  > "$ROOT/evidence/jobs/$SLURM_JOB_ID/pdf_source_sha256.txt"
{ hostname; date -Is; python3 --version; pdftoppm -v 2>&1; pdfinfo -v 2>&1; } \
  > "$ROOT/evidence/jobs/$SLURM_JOB_ID/environment.txt"
python3 "$review_repo/scripts/reconstruction_s05/preview_pdfs.py" "$@"
