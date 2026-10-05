#!/bin/bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/work1/eliasmost/jiaxiwu/plummer_s03_20261005/logs/analysis_%j.log
set -euo pipefail
root=${PLUMMER_S3_ROOT:-/work1/eliasmost/jiaxiwu/plummer_s03_20261005}
source "$root/scripts/amd_watchdog.sh"
start_storage_watchdog "$root"
trap 'kill "$STORAGE_PID" 2>/dev/null || true' EXIT
hostname
date -u
python3 --version
case ${1:?analysis operation} in
 budget) python3 "$root/analysis/budget_preflight.py" "$root";;
 pilot) python3 "$root/analysis/validate_pilot.py" "$root";;
 reduce) python3 "$root/analysis/reduce_session3.py" "$root/runs/${2:?label}" "$root/reduced/$2";;
 audit) python3 "$root/scripts/finish_segment.py" "$root" "${2:?label}" "${3:?job}" "${4:?rc}" "${5:?elapsed}" "${6:?Slurm state}";;
 *) exit 3;;
esac
