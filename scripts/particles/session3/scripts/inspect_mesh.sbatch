#!/bin/bash
#SBATCH --job-name=pl_s3_mesh
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005/logs/mesh_%j.log
set -euo pipefail
session=/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:${LD_LIBRARY_PATH:-}
cd "$session/evidence"
"$session/initial_data/build_cpu/src/athena" -i "$session/inputs/main_live.athinput" -m > mesh_main.txt
mv mesh_structure.dat mesh_main_structure.dat
"$session/initial_data/build_cpu/src/athena" -i "$session/inputs/coarse_live.athinput" -m > mesh_coarse.txt
mv mesh_structure.dat mesh_coarse_structure.dat
cat mesh_main.txt mesh_coarse.txt
