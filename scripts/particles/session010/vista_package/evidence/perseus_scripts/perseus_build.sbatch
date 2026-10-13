#!/bin/bash
#SBATCH -J jeans10-build
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH --mail-user=jiaxiwu@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
SESSION=/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013
export PATH=/home/emost/codes/openmpi/bin:/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ROOT=$SESSION/evidence/build_${SLURM_JOB_ID}
mkdir -p "$ROOT"
{
 date -u; hostname; scontrol show job "$SLURM_JOB_ID"
 nvcc --version; mpicxx --showme; nvidia-smi --query-gpu=name,memory.total --format=csv
 git -C "$SESSION/code" rev-parse HEAD
 git -C "$SESSION/code" submodule status
} > "$ROOT/provenance.txt" 2>&1
for kind in baseline patched; do
 repo=$SESSION/code
 [ "$kind" != baseline ] || repo=$SESSION/baseline_code
 build=$ROOT/$kind
 cmake -S "$repo" -B "$build" -DKokkos_ENABLE_CUDA=On -DKokkos_ARCH_AMPERE80=On \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER="$repo/kokkos/bin/nvcc_wrapper" \
  -DAthena_ENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release -DPROBLEM=particles/gi_cluster \
  > "$ROOT/$kind.configure.log" 2>&1
 cmake --build "$build" -j 2 > "$ROOT/$kind.build.log" 2>&1
 sha256sum "$build/src/athena" > "$ROOT/$kind.sha256"
 git -C "$repo" rev-parse HEAD > "$ROOT/$kind.source"
done
python3 - "$SESSION" "$ROOT" <<'PY'
import json,sys
from pathlib import Path
s,r=map(Path,sys.argv[1:]);(s/'evidence/build_receipt.json').write_text(json.dumps(dict(passed=True,root=str(r),baseline=str(r/'baseline/src/athena'),patched=str(r/'patched/src/athena')),indent=2)+'\n')
PY
