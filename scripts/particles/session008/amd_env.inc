export ROOT=/work1/eliasmost/jiaxiwu/gi_s008_amd_20261002
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset OMPI_MCA_pml OMPI_MCA_btl UCX_TLS UCX_MEMTYPE_CACHE
EXE=$ROOT/build/src/athena
DECK=$ROOT/inputs/gi_cluster_s8.athinput
