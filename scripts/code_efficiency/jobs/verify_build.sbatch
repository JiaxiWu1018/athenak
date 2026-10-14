#!/bin/bash
#SBATCH --job-name=eff-verify-build
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/jiaxiwu/NRPIC/paper/session_02_code_efficiency/logs/verify_build.%j.log
set -euo pipefail
SESSION=${EFFICIENCY_SESSION:-/data/jiaxiwu/NRPIC/paper/session_02_code_efficiency}
REPO=$SESSION/src/athenak
BUILD=$SESSION/build_release
REC=$SESSION/provenance/build_$SLURM_JOB_ID
mkdir -p "$REC"
export PATH=/home/emost/codes/openmpi/bin:/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
{ date -u; hostname; scontrol show job "$SLURM_JOB_ID"; git -C "$REPO" rev-parse HEAD; git -C "$REPO" status --short; git -C "$REPO" submodule status; nvcc --version; mpicxx --showme; gcc --version; cmake --version; } > "$REC/environment.txt" 2>&1
for mode in OFF ON; do
  kind=release
  [ "$mode" = OFF ] || kind=profile
  cmake -S "$REPO" -B "$BUILD" -DAthena_EFFICIENCY_DEVICE_TIMING="$mode" > "$REC/$kind.configure.log" 2>&1
  # Force both modes so no stale pusher object can enter either executable.
  gmake -C "$BUILD" -f src/CMakeFiles/athena.dir/build.make src/CMakeFiles/athena.dir/particles/gr_boris.cpp.o -B > "$REC/$kind.pusher.log" 2>&1
  gmake -C "$BUILD" -f src/CMakeFiles/athena.dir/build.make src/CMakeFiles/athena.dir/build -j 2 -o src/CMakeFiles/athena.dir/flags.make > "$REC/$kind.link.log" 2>&1
  cp "$BUILD/src/athena" "$SESSION/binaries/$kind/athena.$SLURM_JOB_ID"
  mv "$SESSION/binaries/$kind/athena.$SLURM_JOB_ID" "$SESSION/binaries/$kind/athena"
  cp "$BUILD/CMakeCache.txt" "$REC/$kind.CMakeCache.txt"
  cp "$BUILD/src/CMakeFiles/athena.dir/flags.make" "$REC/$kind.flags.make"
done
sha256sum "$SESSION/binaries/release/athena" "$SESSION/binaries/profile/athena" > "$REC/executables.sha256"
python3 - "$SESSION" "$SLURM_JOB_ID" <<'PY'
import sys,json,hashlib,subprocess
from pathlib import Path
root=Path(sys.argv[1]);job=sys.argv[2];manifest={}
for kind in ['release','profile']:
 exe=root/'binaries'/kind/'athena';digest=hashlib.file_digest(exe.open('rb'),'sha256').hexdigest()
 dest=root/'provenance/executables'/digest
 if not dest.exists():dest.write_bytes(exe.read_bytes())
 manifest[digest]=dict(kind=kind,source_commit=subprocess.check_output(['git','-C',str(root/'src/athenak'),'rev-parse','HEAD'],text=True).strip(),build_job=job,build_record=f'provenance/build_{job}',device_timing=kind=='profile')
(root/'provenance/executables/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(root/'provenance/build_receipt.json').write_text(json.dumps(dict(job_id=job,status='built'))+'\n')
PY
python3 "$SESSION/scripts/validate_numerics.py"
