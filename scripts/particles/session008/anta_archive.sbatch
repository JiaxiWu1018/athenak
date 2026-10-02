#!/bin/bash
#SBATCH -J jeans8-archive
#SBATCH -p long
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002
test ! -e "$ROOT/evidence/ARCHIVE_ANALYSIS_COMPLETE.json" || exit 0
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ ! -d "$ROOT/python_deps/imageio_ffmpeg" ]]; then
 /home/jiaxiwu/miniconda3/bin/python -m pip install --only-binary=:all: --target "$ROOT/python_deps" imageio-ffmpeg==0.6.0
fi
python3 "$ROOT/scripts/anta_archive.py" --once
