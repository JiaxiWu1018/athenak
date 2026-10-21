#!/bin/bash
#SBATCH -J jeans11-archive
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH -t 00:30:00
set -euo pipefail
python3 /data/jiaxiwu/NRPIC/GI_in_cluster/session_011_amd_ah_orbit_20261021/scripts/archive_watch.py
