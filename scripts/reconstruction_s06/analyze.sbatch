#!/bin/bash
#SBATCH -J st_s06_analysis
#SBATCH -N 1
#SBATCH -p mi2104x
#SBATCH -A eliasmost
#SBATCH -n 4
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --exclude=k003-002,k003-009,k003-010
#SBATCH -t 01:00:00
#SBATCH -o /work1/eliasmost/jiaxiwu/st_reconstruction_s06_20261016/logs/%x.%j.log
source /work1/eliasmost/jiaxiwu/st_reconstruction_s06_20261016/src/athenak/scripts/reconstruction_s06/amd_common.sh
python3 "$ROOT/src/athenak/scripts/reconstruction_s06/analyze_extension.py"
python3 "$ROOT/src/athenak/scripts/reconstruction_s06/verify.py" --final
python3 "$ROOT/src/athenak/scripts/reconstruction_s06/plot_results.py"
python3 "$ROOT/src/athenak/scripts/reconstruction_s06/finalize_evidence.py"
