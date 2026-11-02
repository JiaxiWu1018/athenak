"""Independent native-volume and known-error-order checks for checkpoint reductions."""
import os
from pathlib import Path
import numpy as np
from campaign import ROOT
from state_io import require_job, read_rst, final_rst, write_json
from analyze import coordinates, differences
require_job()
work=ROOT/'tests/cpu_495053/np1'
state=read_rst(final_rst(work));vol10=vol16=0.
for m in range(len(state['locations'])):
    r2,dv=coordinates(state,m);vol10+=np.count_nonzero(r2<100)*dv;vol16+=np.count_nonzero(r2<256)*dv
hist=np.loadtxt(work/'gate.user.hst')[-1]
assert vol10==hist[2] and vol16==hist[8],(vol10,vol16,hist[[2,8]])
results=[]
for p in (1,2):
    states=[]
    for h in (.2,.1,.05):
        s=dict(state);s['grid']=np.array(state['grid'])+h**p
        s['particles']=np.array(state['particles']);s['particles'][1:8]+=h**p
        states.append(s)
    rows=differences(states)
    orders=[r['observed_p'] for r in rows]
    assert all(abs(q-p)<1e-10 for q in orders),orders
    results.append(dict(injected_order=p,min_order=min(orders),max_order=max(orders)))
    del states
write_json(ROOT/f'evidence/analysis_gate_{os.environ["SLURM_JOB_ID"]}.json',dict(native_volumes=[vol10,vol16],known_orders=results,passed=True))
