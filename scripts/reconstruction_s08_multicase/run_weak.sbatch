#!/bin/bash
#SBATCH -J st8b_weak_scan
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 1
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH -t 00:20:00
#SBATCH --no-requeue
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023/logs/%x.%j.log
set -euo pipefail
batch=/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023
record="$batch/perseus/evidence/jobs/$SLURM_JOB_ID"
mkdir -p "$record"
cp -p "$0" "$record/executed.sbatch"
start=$(date +%s)
finish() {
 status=$?
 python3 - "$record" "$start" "$status" <<'END'
import datetime,json,os,sys,time
from pathlib import Path
(Path(sys.argv[1])/'ALLOCATION_END.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job=os.environ['SLURM_JOB_ID'],elapsed_before_exit_seconds=time.time()-int(sys.argv[2]),nodes=int(os.environ['SLURM_NNODES']),exit_status=int(sys.argv[3])),indent=2)+'\n')
END
 scontrol show job "$SLURM_JOB_ID" > "$record/scontrol_before_exit.txt" 2>&1 || true
}
trap finish EXIT
test -f "$batch/perseus/evidence/CUDA_BUILD_OK.json"
for specification in 'yc0500_u0010 -0.01' 'yc0500_u0005 -0.005'; do
 read -r case_name mode_u <<< "$specification"
 if [ "$case_name" = yc0500_u0005 ] && [ -f "$batch/evidence/U0005_DEDICATED.json" ]; then continue; fi
 remaining=$((1200-$(date +%s)+start))
 if [ "$remaining" -lt 600 ]; then break; fi
 export ST8X_WALL_SECONDS="$remaining"
 bash "$batch/src/athenak/scripts/reconstruction_s08_multicase/launch_perseus.sh" "$case_name" "$mode_u"
done
