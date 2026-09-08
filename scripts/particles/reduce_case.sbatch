#!/bin/bash
# Slurm-side reduction of one run's pvtk dumps.  Bulk data pass: never on a login node.
#   sbatch -J reduce_<tag> reduce_case.sbatch RUNDIR OUTTAG
#SBATCH -o /work1/eliasmost/jiaxiwu/plummer_s01_20260906/logs/%x.%j.log
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p mi2104x
#SBATCH -t 2:00:00
set -uo pipefail
rundir=${1:?usage: RUNDIR OUTTAG}
outtag=${2:?usage: RUNDIR OUTTAG}
R=/work1/eliasmost/jiaxiwu/plummer_s01_20260906
cd "$R"
mkdir -p "reduced/$outtag"
python3 analysis/reduce_run.py --rundir "$rundir" --out "reduced/$outtag" \
  --period 1192.496781 --lmax 4 --macro 4 --ncohort 32
rc=$?
echo "REDUCE DONE ($outtag) rc=$rc"
exit $rc
