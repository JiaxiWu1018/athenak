#!/bin/bash
#SBATCH -J jeans10-compact
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -c 1
#SBATCH --gres=gpu:3
#SBATCH --mem=256G
#SBATCH -t 06:00:00
#SBATCH --mail-user=jiaxiwu@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
SESSION=/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013
export PATH=/home/emost/codes/openmpi/bin:/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export OMPI_MCA_hwloc_base_binding_policy=none OMPI_MCA_rmaps_base_mapping_policy=slot
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export OMPI_MCA_opal_cuda_support=true OMPI_MCA_pml=ucx
export UCX_TLS=self,sm,cuda_copy,cuda_ipc UCX_MEMTYPE_CACHE=n
{
 date -u;hostname;scontrol show job "$SLURM_JOB_ID";nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
 mpicxx --showme;df -h /data
} > "$SESSION/evidence/compact_provenance_${SLURM_JOB_ID}.txt" 2>&1
python3 "$SESSION/code/scripts/particles/session010/compact_checks.py" --session "$SESSION"
