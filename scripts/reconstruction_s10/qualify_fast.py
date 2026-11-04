"""Qualify a separately pinned strict predicted-fixed-node executable; no old pins changed."""
import json
import os
from pathlib import Path
import shutil
import sys
from campaign import ROOT, resource_snapshot
from state_io import require_job, sha, write_json

require_job()
repo=ROOT/'src/athenak'
source=json.loads((ROOT/'evidence/FIXED_FAST_V1_SOURCE.json').read_text())
for name,digest in source['files'].items():
    assert sha(repo/name)==digest,name
manifest=json.loads((ROOT/'evidence/TRANSFER_MANIFEST.json').read_text())
for row in manifest['files']:
    p=ROOT/'reference/session7'/row['path']
    assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
assert sha(ROOT/'protected/athena.rk3')=='f0e278f08855fbc521f9ae6c7c9387cc25b5be2a1da9b7709a8174d7e8c5d200'
snapshot=resource_snapshot()
write_json(ROOT/f'evidence/fast_gate_{os.environ["SLURM_JOB_ID"]}/resources_{sys.argv[1]}.json',snapshot)
assert not snapshot['stop']
if sys.argv[1]=='promote':
    gates={}
    for kind in ('cpu','hip','seams'):
        p=ROOT/f'tests/fast_{kind}_{os.environ["SLURM_JOB_ID"]}/GATES_OK.json'
        gates[kind]=dict(path=str(p),sha256=sha(p),results=json.loads(p.read_text()))
    exe=ROOT/'build_hip/src/athena';out=ROOT/'protected/athena.fixed_fast_v1'
    assert gates['hip']['results']['exe_sha256']==sha(exe)
    if out.exists():assert sha(out)==sha(exe)
    else:shutil.copy2(exe,out)
    write_json(ROOT/'evidence/FIXED_FAST_V1_OK.json',dict(passed=True,stencil='predicted_fixed',fallback='none',
        numerical_scheme='fixed_v1 with guarded identical-stage derivative reuse',source_commit=source['commit'],source_manifest_sha256=sha(ROOT/'evidence/FIXED_FAST_V1_SOURCE.json'),
        hip_sha256=sha(out),cpu_sha256=sha(ROOT/'build_cpu/src/athena'),
        job_id=os.environ['SLURM_JOB_ID'],gates=gates,resources=snapshot))
