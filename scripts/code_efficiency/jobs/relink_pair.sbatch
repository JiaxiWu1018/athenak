#!/bin/bash
#SBATCH --job-name=eff-pair-link
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/data/jiaxiwu/NRPIC/paper/session_02_code_efficiency/logs/build.%j.log
set -euo pipefail
SESSION=${EFFICIENCY_SESSION:-/data/jiaxiwu/NRPIC/paper/session_02_code_efficiency}
BUILD=$SESSION/build_release
REC=$SESSION/provenance/build_$SLURM_JOB_ID
mkdir -p "$REC"
export PATH=/home/emost/codes/openmpi/bin:/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
{ hostname; date -u; scontrol show job "$SLURM_JOB_ID"; } > "$REC/environment.txt"
# Finish the instrumented binary with the new input-block whitelist.
gmake -C "$BUILD" -f src/CMakeFiles/athena.dir/build.make src/CMakeFiles/athena.dir/build -j 2 -o src/CMakeFiles/athena.dir/flags.make > "$REC/profile_link.log" 2>&1
cp "$BUILD/src/athena" "$SESSION/binaries/profile/athena"
cp "$BUILD/src/CMakeFiles/athena.dir/flags.make" "$REC/flags.make"
# Compile only gr_boris with its optional definition explicitly undefined. Other objects
# are identical between the executables. Record the full command and linker recipe.
python3 - "$BUILD" "$REC" <<'PY'
import subprocess,sys
from pathlib import Path
b,r=map(Path,sys.argv[1:]);f=(b/'src/CMakeFiles/athena.dir/flags.make').read_text();flags=next(x.split('=',1)[1].strip() for x in f.splitlines() if x.startswith('CXX_FLAGS ='))
args=['gmake','-C',str(b),'-f','src/CMakeFiles/athena.dir/build.make','src/CMakeFiles/athena.dir/particles/gr_boris.cpp.o','-B','CXX_FLAGS='+flags+' -UATHENA_EFFICIENCY_DEVICE_TIMING']
(r/'release_object.command.txt').write_text(repr(args)+'\n')
with (r/'release_object.log').open('w') as out:subprocess.run(args,stdout=out,stderr=subprocess.STDOUT,check=True)
PY
cd "$BUILD"
cmake -E cmake_link_script "$BUILD/src/CMakeFiles/athena.dir/link.txt" --verbose=1 > "$REC/release_link.log" 2>&1
cp "$BUILD/src/athena" "$SESSION/binaries/release/athena"
sha256sum "$SESSION/binaries/release/athena" "$SESSION/binaries/profile/athena" > "$REC/executables.sha256"
printf '{"job_id":"%s","status":"built","release":"%s","profile":"%s"}\n' "$SLURM_JOB_ID" "$SESSION/binaries/release/athena" "$SESSION/binaries/profile/athena" > "$SESSION/provenance/build_receipt.json"
