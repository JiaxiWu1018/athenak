#!/bin/bash
export SESSION=/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021
export ROOT=$SESSION/perseus S8_MACHINE=perseus
export PATH=/home/emost/codes/openmpi/bin:/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT/scripts:$SESSION/src/athenak/scripts/reconstruction_s05:$SESSION/src/athenak/vis/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg TZ=UTC
export OMPI_MCA_hwloc_base_binding_policy=none OMPI_MCA_rmaps_base_mapping_policy=slot
export OMPI_MCA_opal_cuda_support=true OMPI_MCA_pml=ucx
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export UCX_TLS=self,sm,cuda_copy,cuda_ipc UCX_MEMTYPE_CACHE=n
