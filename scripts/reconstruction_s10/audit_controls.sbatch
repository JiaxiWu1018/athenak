#!/bin/bash
#SBATCH --job-name=st10_control_audit
#SBATCH --partition=mi2101x
#SBATCH --account=eliasmost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102/logs/control_audit.%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
python3 /work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102/src/athenak/scripts/reconstruction_s10/audit_controls.py
