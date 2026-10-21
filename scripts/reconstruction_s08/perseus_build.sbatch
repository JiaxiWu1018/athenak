#!/bin/bash
#SBATCH -J st_s08_cuda_build
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH --no-requeue
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/perseus/logs/%x.%j.log
set -euo pipefail
SESSION=/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021
repo=$SESSION/src/athenak
export ROOT=$SESSION/perseus
record=$ROOT/evidence/build_$SLURM_JOB_ID
mkdir -p "$record" "$ROOT/protected"
export PATH=/home/emost/codes/openmpi/bin:/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
{
 date -u; hostname; scontrol show job "$SLURM_JOB_ID"
 nvcc --version; mpicxx --showme; mpirun --version; gcc --version; python3 --version; cmake --version
 nvidia-smi --query-gpu=name,memory.total,uuid --format=csv
 git -C "$repo" rev-parse HEAD
 git -C "$repo" submodule status
 git -C "$repo" diff 12633257d601d53c8af7a35d77519d681c900f75 -- src CMakeLists.txt kokkos
 df -B1 /data
} > "$record/provenance.txt" 2>&1
# Native source is frozen; workflow-only commits may be added separately.
test -z "$(git -C "$repo" diff 12633257d601d53c8af7a35d77519d681c900f75 -- src CMakeLists.txt kokkos)"
cmake -S "$repo" -B "$ROOT/build" -DKokkos_ENABLE_CUDA=On -DKokkos_ARCH_AMPERE80=On  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER="$repo/kokkos/bin/nvcc_wrapper"  -DAthena_ENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-O3 -DCMAKE_C_FLAGS=-O3  -DPROBLEM=particles/relativistic_cluster > "$record/configure.log" 2>&1
cmake --build "$ROOT/build" -j"${SLURM_CPUS_PER_TASK:-2}" > "$record/build.log" 2>&1
cp -p "$ROOT/build/src/athena" "$ROOT/protected/athena"
cp "$ROOT/build/CMakeCache.txt" "$record/CMakeCache.txt"
sha256sum "$ROOT/protected/athena" > "$record/executable.sha256"
python3 - "$record" <<'PYCODE'
import json,sys,datetime
from pathlib import Path
r=Path(sys.argv[1]); (r/'BUILD_OK.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),build_passed=True,native_gates_passed=False,executable_sha256=(r/'executable.sha256').read_text().split()[0]),indent=2)+'\n')
PYCODE
