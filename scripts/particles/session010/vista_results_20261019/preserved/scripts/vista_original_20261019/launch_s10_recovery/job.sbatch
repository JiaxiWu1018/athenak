#!/usr/bin/env bash
# All resources/dependencies are supplied by the login submission script.
#SBATCH --mail-user=jiaxiwu@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
STAGE=${1:?stage required}
RUN=${2:?run required}
SEGMENT=${3:?segment required}
HERE=${4:?recovery launcher directory required}
PYTHON=${5:?Python executable required}
module reset
module load nvidia/24.7 openmpi/5.0.5 cmake/4.1.1
export UCX_IB_GPU_DIRECT_RDMA=no
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$RUN/logs"
{
  date -u
  hostname
  scontrol show job "$SLURM_JOB_ID"
  module list
  mpicxx --showme
  ucx_info -v
  nvidia-smi -L
} > "$RUN/logs/job_${SLURM_JOB_ID}.provenance.txt" 2>&1
exec "$PYTHON" "$HERE/recover.py" job "$RUN" --segment "$SEGMENT"
