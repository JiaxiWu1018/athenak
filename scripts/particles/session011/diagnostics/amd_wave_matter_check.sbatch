#!/bin/bash
#SBATCH -A eliasmost
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:10:00
#SBATCH -J s11-wave-matter-check
#SBATCH --no-requeue
set -euo pipefail
ROOT=${SESSION_ROOT:?}
SOURCE_COMMIT=${SOURCE_COMMIT:?}
REVIEW=${REVIEW:?}
SOURCE_DIR=$ROOT/evidence/wave_matter_source_$SLURM_JOB_ID
OUT=$ROOT/evidence/wave_matter_$SLURM_JOB_ID
mkdir "$SOURCE_DIR"
git -C "$ROOT/code" archive "$SOURCE_COMMIT" scripts/particles/session011 | tar -x -C "$SOURCE_DIR"
export TOOLING_DIR=$SOURCE_DIR/scripts/particles/session011
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=$ROOT/.mpl_cache
{
 date -u
 hostname
 scontrol show job "$SLURM_JOB_ID"
 git -C "$ROOT/code" rev-parse "$SOURCE_COMMIT"
 git -C "$ROOT/code" submodule status --recursive
 python3 --version
} > "$SOURCE_DIR/provenance.txt"
scontrol write batch_script "$SLURM_JOB_ID" "$SOURCE_DIR/submitted.sbatch"
python3 "$TOOLING_DIR/diagnostics/check_wave_and_matter.py" --review "$REVIEW" --out "$OUT" > "$SOURCE_DIR/check.log" 2>&1
