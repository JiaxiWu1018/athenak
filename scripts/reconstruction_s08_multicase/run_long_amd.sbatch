#!/bin/bash
#SBATCH -J st8b_long700_amd
#SBATCH -A eliasmost
#SBATCH -p mi2508x
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -c 2
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 04:00:00
#SBATCH -o /work1/eliasmost/jiaxiwu/st_s08_scan_20261023/logs/%x.%j.log
set -euo pipefail
batch=/work1/eliasmost/jiaxiwu/st_s08_scan_20261023
export ST8_IMPORT_SNAPSHOT="$batch/staging/snapshot_yc0500_long700_$1"
export ST8X_WALL_SECONDS=14400
# The source export publishes this marker only after its copy is complete.
# Destination qualification checks all received and copied payloads and then
# performs the actual eight-rank HIP zero-step gate before any evolution.
for attempt in $(seq 1 60); do
  if [ -f "$ST8_IMPORT_SNAPSHOT/SNAPSHOT_MANIFEST.json" ]; then
    export ST8X_WALL_SECONDS=$((14400-SECONDS))
    bash "$batch/scripts/launch_amd.sh" yc0500_long700 0
    exit 0
  fi
  sleep 10
done
echo "Snapshot is not ready; source and staging remain preserved."
exit 1
