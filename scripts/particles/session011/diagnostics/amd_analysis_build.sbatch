#!/bin/bash
#SBATCH -A eliasmost
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH -J s11-analysis-build
#SBATCH --no-requeue
set -euo pipefail
ROOT=${SESSION_ROOT:?}
SOURCE_COMMIT=${SOURCE_COMMIT:?}
OUT=$ROOT/evidence/analysis_build_$SLURM_JOB_ID
mkdir "$OUT"
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=$ROOT/.mpl_cache PRODUCT_QA=$OUT/qa
SCRIPTS=$ROOT/code/scripts/particles/session011
test "$(git -C "$ROOT/code" rev-parse HEAD)" = "$SOURCE_COMMIT"
{
 date -u
 hostname
 module list
 scontrol show job "$SLURM_JOB_ID"
 git -C "$ROOT/code" rev-parse HEAD
 git -C "$ROOT/code" submodule status
 hipcc --version
 mpicxx --showme
} > "$OUT/provenance.txt" 2>&1
git -C "$ROOT/code" diff df6ab41d -- src > "$OUT/analysis_only_source.diff"
test "$(git -C "$ROOT/code" diff --name-only df6ab41d -- src)" = src/z4c/frozen_ah_checks.hpp
python3 "$SCRIPTS/test_runtime.py" > "$OUT/runtime_tests.log" 2>&1
python3 "$SCRIPTS/test_products.py" > "$OUT/product_tests.log" 2>&1
cmake -S "$ROOT/code" -B "$ROOT/build" -DPROBLEM=particles/gi_cluster > "$OUT/configure.log" 2>&1
cp "$ROOT/build/CMakeCache.txt" "$OUT/CMakeCache.txt"
cmake --build "$ROOT/build" -j 16 > "$OUT/build.log" 2>&1
cp "$ROOT/build/src/athena" "$OUT/athena_saved_ah"
python3 - "$ROOT" "$OUT" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
gate=json.loads((root/'control/VALIDATION_PASS.json').read_text())
assert sha(Path(gate['executable']))==gate['executable_sha256']
names=('endpoint.py','products.py','test_runtime.py','test_products.py')
record=dict(status='build_and_python_tests_pass',job_id=os.environ['SLURM_JOB_ID'],source=os.environ['SOURCE_COMMIT'],
 executable=str(out/'athena_saved_ah'),executable_sha256=sha(out/'athena_saved_ah'),
 production_executable_sha256=gate['executable_sha256'],
 tooling_sha256={n:sha(root/'code/scripts/particles/session011'/n) for n in names})
(out/'RESULT.json').write_text(json.dumps(record,indent=2)+'\n')
PY
