#!/bin/bash
#SBATCH -J jeans9-reference-validation
#SBATCH -A eliasmost
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --exclusive
#SBATCH -t 00:30:00
#SBATCH --no-requeue
set -euo pipefail
source /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/amd_env.inc
python3 "$ROOT/scripts/workflow.py" bindings
python3 -m unittest discover -s "$ROOT/scripts" -p test_reference_resume.py
python3 "$ROOT/scripts/validate_reference.py"
