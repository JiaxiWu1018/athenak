#!/bin/bash
#SBATCH -J st8b_anchor500
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH -t 00:30:00
#SBATCH --no-requeue
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023/logs/%x.%j.log
set -euo pipefail
batch=/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023
record="$batch/perseus/evidence/jobs/$SLURM_JOB_ID"
mkdir -p "$record" "$batch/perseus/anchors" "$batch/perseus/protected"
export TZ=UTC PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_long_and_inward_20261023/src/athenak/scripts/reconstruction_s08_extensions
cp -p "$0" "$record/executed_transfer.sbatch"
{ hostname; date -u; scontrol show job "$SLURM_JOB_ID"; } > "$record/environment.txt"
rsync -a hpcfund.amd.com:/work1/eliasmost/jiaxiwu/st_s08_yc0500_extensions_20261023/long/runs/yc0500/rst/yc0500.00013.rst "$batch/perseus/anchors/yc0500_500.rst"
python3 - "$batch" "$record" <<'VERIFY'
import datetime,json,os,sys
from pathlib import Path
from checkpoint import validate
from common import receipt
b=Path(sys.argv[1]);p=b/'perseus/anchors/yc0500_500.rst';v=validate(p);assert v['sha256']=='7309c1b16b29d5c3de300dadce3ab37be2ba73796ae92b78ed45681134ff2e64' and v['time']==500 and v['particle_count']==2000000 and v['ah_onset']<0
v.pop('parameters');exe=Path('/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/perseus/protected/athena');sample=Path('/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/followups/independent_df_20261022/original_samples/yc0500.bin');s=receipt(sample);assert s['sha256']=='7c031edd772dd42b3823a0fb26f9566d35732c939c169e7b20271b91f96c4339'
link=b/'perseus/protected/athena_long_original'
if not link.exists():link.symlink_to(exe)
r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job=os.environ['SLURM_JOB_ID'],checkpoint=v,source='/work1/eliasmost/jiaxiwu/st_s08_yc0500_extensions_20261023/long/runs/yc0500/rst/yc0500.00013.rst',source_byte_identity=True,executable=receipt(exe),sample=s,zero_step_machine_gate_passed=False)
(b/'perseus/evidence/ANCHOR_500_OK.json').write_text(json.dumps(r,indent=2)+'\n');(Path(sys.argv[2])/'TRANSFER_OK.json').write_text(json.dumps(r,indent=2)+'\n')
VERIFY
