#!/bin/bash
#SBATCH --job-name=st10_inventory
#SBATCH --partition=mi2101x
#SBATCH --account=eliasmost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102/logs/archive_inventory.%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 /work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102/src/athenak/scripts/reconstruction_s10/archive_inventory.py
