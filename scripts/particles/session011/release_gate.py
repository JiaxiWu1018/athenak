#!/usr/bin/env python3
"""Seal validated engine, input, and runtime scripts; does not submit production."""
import argparse
import datetime
import json
import subprocess
from pathlib import Path
from runtime import atomic,digest,require_compute

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--final',type=Path,required=True);p.add_argument('--preflight',type=Path,required=True);p.add_argument('--held-job');a=p.parse_args()
require_compute()
root=a.root;control=root/'control';control.mkdir(exist_ok=True)
if (control/'submission.json').exists():
    submitted=json.loads((control/'submission.json').read_text())
    assert a.held_job==submitted['job_id'] and submitted['segment']==0
    assert not (root/'runs/segment_0000').exists() and not (control/'latest_checkpoint.json').exists()
    scheduler=subprocess.check_output(['scontrol','show','job',a.held_job,'-o'],text=True)
    assert 'JobState=PENDING' in scheduler and 'Reason=JobHeldUser' in scheduler
    old=control/'VALIDATION_PASS.json'
    if old.exists():
        archive=root/'evidence'/('gate_before_'+a.preflight.name+'.json')
        assert not archive.exists()
        archive.write_text(old.read_text())
final=json.loads((a.final/'RESULT.json').read_text());preflight=json.loads((a.preflight/'RESULT.json').read_text())
assert final['status']=='pass' and final['mpi_ranks']==4 and final['evolution_advances']==0
assert preflight['status']=='pass'
head=subprocess.check_output(['git','-C',str(root/'code'),'rev-parse','HEAD'],text=True).strip()
assert head==preflight['source']
assert not subprocess.check_output(['git','-C',str(root/'code'),'status','--porcelain'],text=True).strip()
subprocess.run(['git','-C',str(root/'code'),'diff','--exit-code',final['source'],head,'--','src','CMakeLists.txt','cmake','kokkos'],check=True)
exe=a.final/'athena_production';input_path=root/'inputs/production.athinput'
exe_hash=digest(exe);input_hash=digest(input_path)
assert input_hash==preflight['input_sha256']
recorded={line.split()[1]:line.split()[0] for line in (a.final/'SHA256SUMS').read_text().splitlines()}
assert recorded[str(exe)]==exe_hash
assert recorded[str(a.final/'production.athinput')]==input_hash
scripts={str(path.relative_to(root)):digest(path) for path in sorted((root/'scripts').iterdir()) if path.is_file()}
assert scripts==preflight['runtime_scripts']
gate=dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          source=final['source'],tooling_source=head,executable=str(exe),executable_sha256=exe_hash,
          input_sha256=input_hash,runtime_scripts=scripts,final_checks=final,preflight=preflight,
          endpoint=75,one_fresh_trajectory=True,initial_nodes=6,initial_ranks=24,
          evidence=[str(a.final),str(a.preflight)])
atomic(control/'VALIDATION_PASS.json',gate)
print(json.dumps(gate,indent=2))
