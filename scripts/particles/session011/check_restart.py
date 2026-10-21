#!/usr/bin/env python3
"""Same frozen geometry, separately initialized finder: compare requalified surfaces."""
import csv
import json
import math
import os
import sys
from pathlib import Path

def accepted(path):
    with path.open() as f:
        return [r for r in csv.DictReader(f) if r['published']=='1']

def main():
    assert os.environ.get('SLURM_JOB_ID')
    root=Path(sys.argv[1]);result=[]
    assert 'FROZEN_AH_CHECKS_PASS' in (root/'restarted.log').read_text()
    for h in (0,1):
        a=accepted(root/'fresh'/f'frozen_ah.ah_events_{h}.csv')[-1]
        b=accepted(root/'restarted'/f'frozen_ah.ah_events_{h}.csv')[-1]
        assert a['state_time']==b['state_time']=='0'
        delta={k:abs(float(a[k])-float(b[k])) for k in ('mass','area','spin','x','y','z','rmin','rmean')}
        for k in ('mass','area','rmin','rmean'):
            assert delta[k] <= 5e-4*max(abs(float(a[k])),1e-10),(h,k,delta)
        assert delta['spin']<5e-4
        assert max(delta[k] for k in ('x','y','z'))<1e-12
        assert float(a['hrel'])<=.2 and float(b['hrel'])<=.2
        ca=[float(x) for x in a['coefficients_a0_ac_as'].split(';') if x]
        cb=[float(x) for x in b['coefficients_a0_ac_as'].split(';') if x]
        assert len(ca)==len(cb)
        shape_delta=math.sqrt(sum((x-y)**2 for x,y in zip(ca,cb))/sum(x*x for x in ca))
        assert shape_delta<2e-3,(h,shape_delta)
        result.append(dict(horizon=h,absolute_differences=delta,relative_coefficient_norm=shape_delta))
    (root/'restart_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FROZEN_RESTART_COMPARISON_PASS')

if __name__=='__main__':main()
