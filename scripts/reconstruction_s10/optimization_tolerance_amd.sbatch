#!/bin/bash
#SBATCH --job-name=st10_tolerance_check
#SBATCH --partition=mi2104x
#SBATCH --account=eliasmost
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102/logs/tolerance.%j.log
set -euo pipefail
root=/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
record=$root/evidence/tolerance_$SLURM_JOB_ID
mkdir -p "$record"
{ hostname; date -u; module list 2>&1; } > "$record/environment.txt"
python3 "$root/src/athenak/scripts/reconstruction_s10/optimization_tolerance.py"
