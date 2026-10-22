#!/bin/bash
#SBATCH -A eliasmost
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:10:00
#SBATCH -J s11-ffi-check
#SBATCH --no-requeue
set -euo pipefail
ROOT=${SESSION_ROOT:?}
SOURCE_COMMIT=${SOURCE_COMMIT:?}
REVIEW=${REVIEW:?}
SOURCE_DIR=$ROOT/evidence/ffi_source_$SLURM_JOB_ID
OUT=$ROOT/evidence/ffi_$SLURM_JOB_ID
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
python3 "$TOOLING_DIR/diagnostics/check_ffi_zero_mode.py" --review "$REVIEW" \
 --legacy-products "$ROOT/hourly_tools/ddb12983/scripts/particles/session011/products.py" \
 --out "$OUT" > "$SOURCE_DIR/check.log" 2>&1
