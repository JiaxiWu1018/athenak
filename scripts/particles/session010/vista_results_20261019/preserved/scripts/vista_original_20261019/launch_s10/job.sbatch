#!/bin/bash
# Resources and dependencies are supplied by the login launcher.
#SBATCH --mail-user=jiaxiwu@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
STAGE=${1:?stage required}
RUN=${2:?absolute run root required}
SEGMENT=${3:-0}
# Slurm copies this script to spool; use the explicit original launcher directory.
HERE=${4:?original launcher directory required}
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
  if [ "$STAGE" != build ]; then nvidia-smi -L; fi
  printf 'UCX_IB_GPU_DIRECT_RDMA=%s\n' "$UCX_IB_GPU_DIRECT_RDMA"
} > "$RUN/logs/job_${SLURM_JOB_ID}.provenance.txt" 2>&1
if [ "$STAGE" = build ]; then
  bash "$HERE/build.sh"
else
  "$PYTHON" "$HERE/s10.py" job "$STAGE" "$RUN" --segment "$SEGMENT"
fi
