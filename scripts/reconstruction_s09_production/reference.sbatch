#!/bin/bash
#SBATCH -J st9_reference
#SBATCH -A eliasmost
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 00:30:00
#SBATCH --no-requeue
#SBATCH -o /work1/eliasmost/jiaxiwu/st_s09_perturbed_production_20261024/logs/%x.%j.log
set -euo pipefail
batch=/work1/eliasmost/jiaxiwu/st_s09_perturbed_production_20261024
export S8_MACHINE=amd ST8X_ROOT="$batch" ST8_CASE=yc0283_u0005 ROOT="$batch/reference"
export ST8X_BASE=/work1/eliasmost/jiaxiwu/st_reconstruction_s08_production_20261021
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=UTC
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
record="$ROOT/evidence/jobs/$SLURM_JOB_ID"
mkdir -p "$record/workflow"
cp -p "$batch/scripts/"*.py "$record/workflow/"
cp -p "$0" "$record/executed.sbatch"
export PYTHONPATH="$record/workflow:$ST8X_BASE/dependencies/site:$ST8X_BASE/src/athenak/scripts/reconstruction_s05:$ST8X_BASE/src/athenak/vis/python"
{ hostname; date -u; module list; g++ --version; scontrol show job "$SLURM_JOB_ID"; } > "$record/environment.txt" 2>&1
python3 "$record/workflow/prepare.py" > "$record/preparation.log" 2>&1
python3 "$record/workflow/constraint_reference.py" > "$record/reference.log" 2>&1
scontrol show job "$SLURM_JOB_ID" > "$record/scontrol_before_exit.txt"
