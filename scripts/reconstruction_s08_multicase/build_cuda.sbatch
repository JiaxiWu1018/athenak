#!/bin/bash
#SBATCH -J st8b_cuda_build
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH --no-requeue
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023/logs/%x.%j.log
set -euo pipefail
batch=/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023
repo="$batch/src/athenak"
record="$batch/perseus/evidence/jobs/$SLURM_JOB_ID"
mkdir -p "$record" "$batch/perseus/protected"
export PATH=/home/emost/codes/openmpi/bin:/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TZ=UTC
start=$(date +%s)
finish() { status=$?; python3 - "$record" "$start" "$status" <<'END'
import datetime,json,os,sys,time
from pathlib import Path
p=Path(sys.argv[1]);(p/'ALLOCATION_END.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_before_exit_seconds=time.time()-int(sys.argv[2]),exit_status=int(sys.argv[3]),nodes=int(os.environ['SLURM_NNODES'])),indent=2)+'\n')
END
}
trap finish EXIT
cp -p "$0" "$record/executed_build.sbatch"
{ hostname; date -u; scontrol show job "$SLURM_JOB_ID"; nvcc --version; mpicxx --showme; gcc --version; cmake --version; git -C "$repo" rev-parse HEAD; git -C "$repo" submodule status; nvidia-smi; } > "$record/environment.txt" 2>&1
test -z "$(git -C "$repo" diff db942607bd251dab36bbd4a022ec58ee62317d01 -- src CMakeLists.txt kokkos)"
cmake -S "$repo" -B "$batch/perseus/build_inward" -DKokkos_ENABLE_CUDA=On -DKokkos_ARCH_AMPERE80=On -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER="$repo/kokkos/bin/nvcc_wrapper" -DAthena_ENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-O3 -DCMAKE_C_FLAGS=-O3 -DPROBLEM=particles/relativistic_cluster > "$record/configure.log" 2>&1
cmake --build "$batch/perseus/build_inward" -j"$SLURM_CPUS_PER_TASK" > "$record/build.log" 2>&1
cp -p "$batch/perseus/build_inward/src/athena" "$batch/perseus/protected/athena_inward_db942"
cp -p "$batch/perseus/build_inward/CMakeCache.txt" "$record/CMakeCache.txt"
python3 - "$batch" "$record" <<'DONE'
import datetime,hashlib,json,os,sys
from pathlib import Path
b=Path(sys.argv[1]);p=b/'perseus/protected/athena_inward_db942';r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job=os.environ['SLURM_JOB_ID'],path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),native_revision='db942607bd251dab36bbd4a022ec58ee62317d01',native_gates_passed=False,toolchain='CUDA12.9/OpenMPI; matched native controls and restart gates still required')
(b/'perseus/evidence/CUDA_BUILD_OK.json').write_text(json.dumps(r,indent=2)+'\n');(Path(sys.argv[2])/'BUILD_OK.json').write_text(json.dumps(r,indent=2)+'\n')
DONE
