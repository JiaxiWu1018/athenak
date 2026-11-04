"""Full-count tolerance sensitivity; retain the original failed equivalence gate.

Predeclared acceptance: tighter (1e-13) binaries agree within the original 2e-12
absolute gate. For nominal trajectories, each difference must be below 10% of
the measured nominal-to-tight maximum, or already below 2e-12. This establishes
numerical equivalence relative to solver error, not bitwise equivalence.
"""
import json
import os
from pathlib import Path
from campaign import ROOT, CASES, run_segment, resource_snapshot
from state_io import require_job, read_rst, compare_states, sha, write_json

require_job()
records={}
for revision in ('fixed_v1','fixed_fast_v1'):
    case=f'{revision}_1m_h010_tol13'
    CASES[case]=dict(n=1000000,dt=.1,integrator='rk3',coupling='rk3_predictor',
        end=20,revision=revision,stencil='predicted_fixed',tolerance=1e-13)
    records[revision+'_tight']=run_segment(case,20)
    prior=list((ROOT/'runs'/f'{revision}_1m_h010').glob('*/COMPLETE.json'))
    records[revision]=next(json.loads(p.read_text()) for p in prior
        if json.loads(p.read_text())['target']==20)
    r=records[revision]
    assert sha(r['checkpoint']['path'])==r['checkpoint']['sha256']
states={k:read_rst(v['checkpoint']['path']) for k,v in records.items()}
nominal=compare_states(states['fixed_v1'],states['fixed_fast_v1'])
tight=compare_states(states['fixed_v1_tight'],states['fixed_fast_v1_tight'])
sensitivity={rev:compare_states(states[rev],states[rev+'_tight'])
    for rev in ('fixed_v1','fixed_fast_v1')}
limits={k:max(2e-12,0.1*max(sensitivity[rev][k] for rev in sensitivity))
    for k in nominal}
passed=all(tight[k]<2e-12 and nominal[k]<=limits[k] for k in nominal)
result=dict(passed=passed,nominal=nominal,tight=tight,sensitivity=sensitivity,
    nominal_limits=limits,records=records,resources=resource_snapshot(),
    policy=__doc__,script_sha256=sha(__file__),job_id=os.environ['SLURM_JOB_ID'])
write_json(ROOT/f'evidence/OPTIMIZATION_TOLERANCE_{os.environ["SLURM_JOB_ID"]}.json',result)
print(json.dumps(result,indent=2),flush=True)
assert passed, result
