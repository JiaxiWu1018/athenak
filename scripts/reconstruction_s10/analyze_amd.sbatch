#!/bin/bash
#SBATCH --job-name=st10_analysis
#SBATCH --partition=mi2101x
#SBATCH --account=eliasmost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102/logs/analysis.%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
root=/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102
scripts=$root/src/athenak/scripts/reconstruction_s10
out=$root/analysis/job_$SLURM_JOB_ID
if [[ ${1:-interim} == pilot ]]; then
  python3 "$scripts/pilot_review.py"
  exit 0
fi
python3 "$scripts/analyze.py" --temporal --dest "$out/reduced"
if [[ ${1:-interim} == final ]]; then
  python3 "$scripts/plot.py" --data "$out/reduced" --out "$out/figures"
fi
