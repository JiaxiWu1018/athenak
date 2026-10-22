#!/bin/bash
#SBATCH -A eliasmost
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:05:00
#SBATCH -J s11-real-report
#SBATCH --no-requeue
set -euo pipefail
ROOT=${SESSION_ROOT:?}
SOURCE_COMMIT=${SOURCE_COMMIT:?}
SOURCE_DIR=$ROOT/evidence/real_report_source_$SLURM_JOB_ID
mkdir "$SOURCE_DIR"
git -C "$ROOT/code" archive "$SOURCE_COMMIT" scripts/particles/session011 | tar -x -C "$SOURCE_DIR"
export TOOLING_DIR=$SOURCE_DIR/scripts/particles/session011
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=$ROOT/.mpl_cache
{
 date -u
 hostname
 scontrol show job "$SLURM_JOB_ID"
 git -C "$ROOT/code" rev-parse "$SOURCE_COMMIT"
} > "$SOURCE_DIR/provenance.txt"
python3 "$TOOLING_DIR/diagnostics/check_real_report.py"
