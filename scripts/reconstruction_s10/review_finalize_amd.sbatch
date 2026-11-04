#!/bin/bash
#SBATCH --job-name=st10_review
#SBATCH --partition=mi2101x
#SBATCH --account=eliasmost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102/logs/final_review.%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg ST10_SCHEME=fixed_fast_v1
root=/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102
scripts=$root/src/athenak/scripts/reconstruction_s10
out=$root/analysis/review_$SLURM_JOB_ID
mkdir "$out"
cp -a "$root/analysis/job_498278/reduced" "$out/reduced"
python3 "$scripts/plot.py" --data "$out/reduced" --out "$out/figures"
python3 "$scripts/final_report.py" --data "$out/reduced" --out "$out"
python3 "$scripts/review_final.py" "$out"
