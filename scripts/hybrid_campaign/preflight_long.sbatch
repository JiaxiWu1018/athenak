#!/usr/bin/env bash
#SBATCH --job-name=hyb_preflight
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
# Long-domain preflight: run ~300 cycles of the production q6p1 input on the
# 2816M causally clean mesh; measure MeshBlock count/distribution, wall-clock
# per M, and per-GPU VRAM, to validate the 512-block cost model before
# committing 6 x 8P production jobs.
set -euo pipefail
module purge
module load autotools cmake/3.25.2 prun/2.3 rocm/6.4.1 gnu12/12.2.0 openmpi4/4.1.8
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1 UCX_LOG_LEVEL=error OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx
R=/work1/eliasmost/jiaxiwu/nrpic_hybrid_stability_20260802
OUT=$R/runs/_preflight_long; rm -rf $OUT; mkdir -p $OUT
REC=$R/provenance/preflight_long_job${SLURM_JOB_ID}; mkdir -p $REC
( while true; do rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -4; sleep 20; done ) > $REC/vram_samples.csv 2>&1 &
MON=$!
trap "kill $MON 2>/dev/null || true" EXIT
prun /work1/eliasmost/jiaxiwu/athenak_hybrid_20260802/build_hybrid_gfx90a_rocm6_4_1/src/athena   --kokkos-map-device-id-by=mpi_rank   -i $R/inputs_long/long_q6p1_stratified_antithetic_s1985.athinput -d $OUT   time/nlim=300 output2/dt=1000000 output3/dt=1000000 output4/dt=1000000   output5/dt=1000000 output6/dt=1000000 output7/dt=1000000   > $REC/solver.out 2> $REC/solver.err
grep -E 'Total number of MeshBlocks|Rank = ' $REC/solver.out | head -8
grep -E '^elapsed=' $REC/solver.out | tail -3
echo PREFLIGHT_OK
