#!/bin/bash
#SBATCH -A eliasmost
#SBATCH -p mi2104x
#SBATCH --exclusive
#SBATCH -N 6
#SBATCH -n 24
#SBATCH --ntasks-per-node=4
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH -J s11-analysis-saved
#SBATCH --no-requeue
set -euo pipefail
ROOT=${SESSION_ROOT:?}
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
export OMPI_MCA_btl_vader_single_copy_mechanism=none OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset OMPI_MCA_pml OMPI_MCA_btl UCX_TLS UCX_MEMTYPE_CACHE
TOOLING=$ROOT/evidence/render_qa_${LAUNCH_QA_JOB:?}/source/scripts/particles/session011
python3 "$TOOLING/diagnostics/check_analysis_saved_state.py"
