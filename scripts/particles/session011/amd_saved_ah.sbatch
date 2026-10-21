#!/bin/bash
#SBATCH -J jeans11-saved-AH
#SBATCH -A eliasmost
#SBATCH -p mi2104x
#SBATCH --exclusive
#SBATCH -t 01:00:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=${SESSION_ROOT:?}
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset OMPI_MCA_pml OMPI_MCA_btl UCX_TLS UCX_MEMTYPE_CACHE
python3 "$ROOT/scripts/endpoint.py" probe --root "$ROOT"
