#!/bin/bash
#SBATCH -J jeans11-hourly-archive
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH -t 00:30:00
set -euo pipefail
TASK_DIR=$(dirname "$0")
# Slurm executes a spool copy; use the explicitly pinned archive helper path.
python3 /data/jiaxiwu/NRPIC/GI_in_cluster/session_011_amd_ah_orbit_20261021/hourly_archive_tools/hourly_archive.py
