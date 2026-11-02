"""Conservative admission forecast from full-count 1M/8M pilots; no job submission."""
import os
import json
import re
import numpy as np
from pathlib import Path
from campaign import ROOT, resource_snapshot
from analyze import completed
from state_io import require_job, write_json, sha
require_job()
records={n:next(r for r in completed(f'predictor_{n}m_h020') if r['target']==20) for n in (1,8)}
assert all(r['initial']['exact_sample_match'] and r['checkpoint']['original_ids_masses'] and r['checkpoint']['removals']==0 for r in records.values())
assert all(r['solver']['failures']==0 and r['solver']['max_residual']<=1.01e-11 for r in records.values())
# Use elapsed-cycle timing after warmup; retain whole-process costs separately.
measured={}
for n,r in records.items():
    text=(Path(r['work'])/'run.log').read_text()
    a=np.array([(float(e),float(t)) for e,cycle,t in re.findall(r'elapsed=(\S+) cycle=(\d+) time=(\S+) dt=',text)])
    a=a[(a[:,1]>=2)&(a[:,1]<=18.000001)]
    assert len(a)>=4
    delta=np.diff(a,axis=0);valid=delta[:,1]>1e-8
    slopes=delta[valid,0]/delta[valid,1]
    measured[n]=dict(median_seconds_per_M=float(np.median(slopes)),
        window_seconds_per_M=float((a[-1,0]-a[0,0])/(a[-1,1]-a[0,1])),
        whole_process_seconds_per_M=r['seconds_per_M'])
    measured[n]['forecast_seconds_per_M']=max(measured[n]['median_seconds_per_M'],measured[n]['window_seconds_per_M'])
s1=measured[1]['forecast_seconds_per_M'];s8=measured[8]['forecast_seconds_per_M']
slope=max(0,(s8-s1)/7)
seconds={n:s1+(n-1)*slope for n in (1,2,4,8)}
production=sum(seconds[n]*(500-(20 if n in records else 0)) for n in seconds)/3600
# New h=.1/.05: 2x/4x steps. Legacy costs initially bounded by measured new cost;
# a separate legacy pilot can replace that upper model if the forecast needs refinement.
short_new=100*s1*(2+4)/3600
short_legacy=100*s1*(1+2+4+1)/3600
snapshot=resource_snapshot()
remaining=production+short_new+short_legacy
forecast=snapshot.get('node_hours',0)+1.2*remaining+1.0
# At most ~55 double-precision 2.4GB checkpoints plus sparse particle outputs,
# retained gates/builds/samples, and a 35GiB analysis/output reserve.
storage=snapshot.get('campaign_bytes',0)+180*2**30
result=dict(pilots=records,timing=measured,seconds_per_M_model=seconds,remaining_nominal_node_hours=remaining,
    forecast_node_hours_with_20pct_and_support=forecast,forecast_bytes=storage,
    current_resources=snapshot,under_review_limits=bool(forecast<40 and storage<300*2**30),
    assumptions='Linear particle-count runtime, proportional step-count controls,20% runtime reserve +1 support nodeh; 180GiB incremental storage reserve.',
    script_sha256=sha(__file__))
write_json(ROOT/f'evidence/PILOT_REVIEW_{os.environ["SLURM_JOB_ID"]}.json',result)
print(json.dumps(result,indent=2))
