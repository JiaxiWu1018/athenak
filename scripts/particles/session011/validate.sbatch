#!/bin/bash
#SBATCH -J s11-ah-validation
#SBATCH -A AST21006
#SBATCH -p gh-dev
#SBATCH -N 4
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaxiwu@caltech.edu
set -euo pipefail
module reset
module load nvidia/24.7 openmpi/5.0.5
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UCX_IB_GPU_DIRECT_RDMA=no
: "${S11_ROOT:?Export absolute Session11 root when submitting}"
cd "$S11_ROOT"
python3 -B code/scripts/particles/session011/launch.py validate
