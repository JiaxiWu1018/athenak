#!/bin/bash
#SBATCH -J s11-final-150M
#SBATCH -A AST21006
#SBATCH -p gh
#SBATCH -N 16
#SBATCH -n 16
#SBATCH --ntasks-per-node=1
#SBATCH -t 36:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaxiwu@caltech.edu
set -euo pipefail
module reset
module load nvidia/24.7 openmpi/5.0.5
module list
hostname
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UCX_IB_GPU_DIRECT_RDMA=no
: "${S11_ROOT:?Use submit_s11.sh production}"
: "${S11_PRODUCTION_MANIFEST_SHA256:?Missing submitted production hash binding}"
: "${S11_PRODUCTION_COMMIT:?Missing submitted production commit}"
cd "$S11_ROOT"
python3 -B code/scripts/particles/session011_early/early_run.py production
