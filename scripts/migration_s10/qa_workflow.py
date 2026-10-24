"""Allocated end-to-end reader/render test using accepted historical control only."""
import csv
import numpy as np
from common import *

def main():
    from extension_resources import admission
    admission()
    from products import resolve_prefix, movie
    from analysis import plot_case
    from run_segment import cohort
    from run_segment import protected_checkpoint_paths
    from reduce import csv_rows
    prefix = resolve_prefix()
    fixture=[dict(path=f'fixture_{t}.rst',time=t,dt=.1,ah_onset=95.5 if t>=95.5 else -1,final=t==300)
             for t in [0,10,20,40,95.5,100,200,210,220,300]]
    assert protected_checkpoint_paths(fixture)=={f'fixture_{t}.rst' for t in [0,20,95.5,100,200,210,220,300]}
    with (prefix/'cohort_index_resolved.csv').open() as f:
        rows = list(csv.DictReader(f))
    for t in [0, 350, 700]:
        row = min(rows,key=lambda r:abs(float(r['time'])-t))
        actual,cycle,mask,values=cohort(row['file'])
        assert abs(actual-t)<.65 and mask.sum()==20000
        assert np.array_equal(values[mask==1,0],np.arange(20000)*100)
    out=EXT/'qa'/job()
    out.mkdir(parents=True)
    raw=dict(id='historical_control_renderer_test',key='historical_control_renderer_test',U=0.,start=0.)
    plot_case(raw,prefix,out)
    datum=dict(case=raw,summary=dict(endpoint=700.,ah_first=None),cohorts=rows)
    path=movie([datum],out,'historical_control_3frame_smoke',targets=[0.,350.,700.])
    h=csv_rows(prefix/'history.csv');assert h[0]['time']==0 and abs(h[-1]['time']-700)<1e-7
    atomic(out/'QA_OK.json',dict(utc=now(),job=job(),original_history=receipt(prefix/'SOURCE_MANIFEST.json'),
        resolved_cohorts=receipt(prefix/'RESOLVED.json'),movie=receipt(path),
        fixtures_only=True,no_new_evolution=True,no_session10_scientific_result=True,
        actual_encoded_visual_review_pending=True))
    print(out)

if __name__=='__main__':
    main()
