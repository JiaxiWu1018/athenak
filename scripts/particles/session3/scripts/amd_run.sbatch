#!/bin/bash
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/work1/eliasmost/jiaxiwu/plummer_s03_20261005/logs/%x_%j.log
set -euo pipefail
root=${PLUMMER_S3_ROOT:-/work1/eliasmost/jiaxiwu/plummer_s03_20261005}
label=${1:?run label}
shift
case "$label" in *[!a-zA-Z0-9_]*|'') exit 3;; esac
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
build=$(cat "$root/state/build_path.txt")
exe=$build/src/athena
run=$root/runs/$label
mkdir -p "$run/out"
exec 8>"$run/runner.lock"
flock -n 8 || { printf 'Run already active\n'; exit 3; }
cd "$run"
cp "$root/inputs/$label.athinput" "input_$SLURM_JOB_ID.athinput"
log=$run/segment_$SLURM_JOB_ID.log
exec > >(tee -a "$log") 2>&1
hostname
date -u
module list 2>&1
cat "$build/git-head.txt"
git -C "$root/src/athenak" submodule status
sha256sum "$exe" "input_$SLURM_JOB_ID.athinput"
test -x "$exe"
args=(-i "input_$SLURM_JOB_ID.athinput")
if [ -f segment_state.json ]; then
  python3 "$root/scripts/resume_args.py" "$run" > resume.json
  restart=$(python3 -c 'import json;print(json.load(open("resume.json"))["checkpoint"])')
  args=(-r "$restart")
fi
# Reserve fifteen minutes in four-hour production segments, and twenty percent
# in short preflights so Slurm can backfill them into smaller queue openings.
limit=$(squeue -h -j "$SLURM_JOB_ID" -o %l)
limit_seconds=$(python3 - "$limit" <<'PY'
import sys
s=sys.argv[1];days=0
if '-' in s:days,s=s.split('-');days=int(days)
parts=list(map(int,s.split(':')))
seconds=days*86400+(parts[0]*3600+parts[1]*60+parts[2] if len(parts)==3 else parts[0]*60+parts[1])
print(seconds)
PY
)
allowance=900
if [ "$limit_seconds" -le 3600 ]; then allowance=$((limit_seconds/5)); fi
simulation_seconds=$((limit_seconds-allowance))
athena_time=$(printf '%02d:%02d:%02d' $((simulation_seconds/3600)) $(((simulation_seconds%3600)/60)) $((simulation_seconds%60)))
source "$root/scripts/amd_watchdog.sh"
start_storage_watchdog "$root"
(
  set +e
  while true; do date -u; rocm-smi --showmemuse --csv; sleep 60; done
) > "vram_$SLURM_JOB_ID.log" 2>&1 &
vram_pid=$!
trap 'kill "$STORAGE_PID" "$vram_pid" 2>/dev/null || true' EXIT
printf 'LAUNCH prun %s --kokkos-map-device-id-by=mpi_rank -t %s -d %s' "$exe" "$athena_time" "$run/out"
printf ' %q' "${args[@]}" "$@"
printf '\n'
started=$(date +%s)
rc=0
prun "$exe" --kokkos-map-device-id-by=mpi_rank -t "$athena_time" "${args[@]}" -d "$run/out" "$@" || rc=$?
elapsed=$(($(date +%s)-started))
python3 "$root/scripts/finish_segment.py" "$root" "$label" "$SLURM_JOB_ID" "$rc" "$elapsed"
exit "$rc"
