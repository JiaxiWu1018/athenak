#!/bin/bash
#SBATCH --job-name=st10_midpoint_diagnosis
#SBATCH --partition=mi2101x
#SBATCH --account=eliasmost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102/logs/diagnose.%j.log
set -euo pipefail
root=/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102
repo=$root/src/athenak
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export PYTHONPATH=$repo/scripts/reconstruction_s10
work=$root/diagnostics/job_$SLURM_JOB_ID
mkdir -p "$work"
{ hostname; date -u; module list 2>&1; } > "$work/environment.txt"
cp "$root/evidence/DIAGNOSTIC_SOURCE.json" "$work/SOURCE.json"
cmake --build "$root/build_hip" -j 16
cp "$root/build_hip/src/athena" "$work/athena.diagnostic"
sha256sum "$work/athena.diagnostic" > "$work/executable.sha256"
python3 - <<'PY'
import os,json,subprocess,time
from pathlib import Path
from campaign import ROOT,input_text,resource_snapshot
from state_io import write_json
work=ROOT/f'diagnostics/job_{os.environ["SLURM_JOB_ID"]}'
write_json(work/'RESOURCES.json',resource_snapshot())
p=work/'input.athinput';p.write_text(input_text('predictor_1m_h020',2))
cmd=['mpirun','-np','1',str(work/'athena.diagnostic'),'--kokkos-map-device-id-by=mpi_rank','-i',str(p),'-d',str(work)]
with (work/'run.log').open('w') as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600).returncode
log=(work/'run.log').read_text()
write_json(work/'DIAGNOSIS.json',dict(command=cmd,exit_code=code,lines=[l for l in log.splitlines() if 'rk3-failure' in l]))
assert '[rk3-failure]' in log,(code,log[-3000:])
PY
