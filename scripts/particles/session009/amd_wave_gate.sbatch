#!/bin/bash
#SBATCH -J jeans9-wave-gate
#SBATCH -A eliasmost
#SBATCH -p mi2104x
#SBATCH -N 3
#SBATCH -n 12
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH --no-requeue
set -euo pipefail
source /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/amd_env.inc
python3 "$ROOT/scripts/workflow.py" bindings
python3 "$ROOT/scripts/wave_gate.py" generate
prun "$ROOT/build/mpi_probe" > "$ROOT/evidence/mpi_wave_gate.txt" 2>&1
for wavelength in 2.5 5.0; do
 export RUN=$ROOT/runs/wave_lambda_$wavelength
 mkdir -p "$RUN"
 rc=0
 (cd "$RUN"; prun "$ROOT/build_wave/src/athena" --kokkos-map-device-id-by=mpi_rank -i "$ROOT/inputs/wave_tests/lambda_$wavelength.athinput" -d "$RUN" -t 00:50:00) > "$RUN/run.log" 2>&1 || rc=$?
 printf '%s\n' "$rc" > "$RUN/EXIT_CODE"
 test "$rc" -eq 0
 python3 "$ROOT/scripts/runtime.py" seal --run "$RUN"
done
python3 "$ROOT/scripts/wave_gate.py" assess
