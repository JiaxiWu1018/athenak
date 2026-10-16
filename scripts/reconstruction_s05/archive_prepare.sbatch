#!/bin/bash
#SBATCH -J st_s05_archive_prepare
#SBATCH -A eliasmost
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 00:15:00
#SBATCH -o /work1/eliasmost/jiaxiwu/st_reconstruction_s05_20261014/closeout_transfer/prepare.%j.log
set -euo pipefail
root=/work1/eliasmost/jiaxiwu/st_reconstruction_s05_20261014
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
{ hostname; date -Is; module list 2>&1; hipcc --version; python3 --version; } > "$root/closeout_transfer/toolchain.txt"
python3 "$root/closeout_transfer/archive_closeout.py" prepare
