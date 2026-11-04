"""Allocated full-count equivalence gate for the guarded same-stage optimization."""
import json
import os
from pathlib import Path
from campaign import ROOT, resource_snapshot
from state_io import require_job, sha, read_rst, compare_states, write_json

require_job()
records=[]
for case in ('fixed_v1_1m_h010','fixed_fast_v1_1m_h010'):
    found=[json.loads(p.read_text()) for p in (ROOT/'runs'/case).glob('*/COMPLETE.json')]
    r=next(r for r in found if r['target']==20)
    assert r['initial']['exact_sample_match'] and r['checkpoint']['original_ids_masses']
    assert r['checkpoint']['removals']==0 and r['solver']['failures']==0
    assert r['solver']['max_residual']<=1.01e-11
    path=Path(r['checkpoint']['path']);assert sha(path)==r['checkpoint']['sha256']
    records.append(r)
states=[read_rst(r['checkpoint']['path']) for r in records]
diff=compare_states(*states)
passed=all(diff[k]<2e-12 for k in ('grid_max_abs','particle_max_abs','energy_max_abs'))
result=dict(passed=passed,time=20,particles=1000000,comparison=diff,records=records,
    policy='Same fixed-node equations, same timestep and sample; only guarded duplicate derivative reuse differs.',
    resources=resource_snapshot(),script_sha256=sha(__file__),job_id=os.environ['SLURM_JOB_ID'])
write_json(ROOT/f'evidence/FAST_EQUIVALENCE_{os.environ["SLURM_JOB_ID"]}.json',result)
print(json.dumps(result,indent=2))
assert passed,diff
