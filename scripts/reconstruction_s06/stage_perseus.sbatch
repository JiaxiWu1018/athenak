#!/bin/bash
#SBATCH -J st_s06_stage
#SBATCH -p preempt
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o /data/jiaxiwu/NRPIC/reconstruction_ST/06_yc0819_long_stability_20261016/logs/%x.%j.log
set -euo pipefail
session_root=/data/jiaxiwu/NRPIC/reconstruction_ST/06_yc0819_long_stability_20261016
previous_root=/data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014
amd_root=/work1/eliasmost/jiaxiwu/st_reconstruction_s06_20261016
ssh hpcfund.amd.com "mkdir -p '$amd_root'/src '$amd_root'/restart '$amd_root'/protected '$amd_root'/samples '$amd_root'/reference '$amd_root'/baseline '$amd_root'/evidence '$amd_root'/logs; test ! -e '$amd_root'/src/athenak; git clone --branch experiment/ST-reconstruction-s06-long-stability --single-branch --recurse-submodules git@github.com:JiaxiWu1018/athenak.git '$amd_root'/src/athenak"
rsync -a "$previous_root/restarts/proper_stratified_s1/proper_stratified_s1.00003.rst" hpcfund.amd.com:"$amd_root/restart/"
rsync -a "$previous_root/archive/amd_closeout_20261016/protected/athena" hpcfund.amd.com:"$amd_root/protected/"
rsync -a "$previous_root/archive/amd_closeout_20261016/samples/proper_stratified_s1.bin" "$previous_root/samples/model.npz" hpcfund.amd.com:"$amd_root/samples/"
rsync -a "$previous_root/reference/proper.npz" hpcfund.amd.com:"$amd_root/reference/"
rsync -a "$session_root/baseline/" hpcfund.amd.com:"$amd_root/baseline/"
rsync -a "$session_root/evidence/model.json" "$session_root/evidence/SETUP.json" hpcfund.amd.com:"$amd_root/evidence/"
mkdir -p "$session_root/evidence/jobs/$SLURM_JOB_ID"
{ hostname; date -Is; git -C "$session_root/src/athenak" rev-parse HEAD; } > "$session_root/evidence/jobs/$SLURM_JOB_ID/stage.txt"
touch "$session_root/evidence/STAGE_COMPLETE"
