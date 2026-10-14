#!/usr/bin/env python3
"""Compare release and instrumented frozen trajectories, plus liveness/source gates."""
import json,sys
import numpy as np
from pathlib import Path
from run_case import run,ROOT
matrix=json.loads((ROOT/'matrix.json').read_text());checks=[]
for metric in ['st','flat']:
    c=next(x.copy() for x in matrix if x['name']==f'count_{metric}_dep0_N10000')
    c['name']='numeric_'+metric
    release=run(c,'validation_release',trace=(metric=='st'),target_override=1.5)
    diagnostic=run(c,'device',target_override=1.5)
    same=release['valid'] and diagnostic['valid'] and release['initial'][0]['payload_hash']==diagnostic['initial'][0]['payload_hash'] and release['final'][0]['payload_hash']==diagnostic['final'][0]['payload_hash']
    clocks=diagnostic['timings'][0]['device_cycles'] if diagnostic['timings'] else []
    left=np.genfromtxt(ROOT/release['attempt_path']/'final.payload.rank0.csv',delimiter=',',names=True)
    right=np.genfromtxt(ROOT/diagnostic['attempt_path']/'final.payload.rank0.csv',delimiter=',',names=True)
    left.sort(order='tag');right.sort(order='tag')
    matching_ids=np.array_equal(left['tag'],right['tag'])
    matching_mass=np.array_equal(left['mass'],right['mass'])
    difference=max(float(np.max(np.abs(left[k]-right[k])/np.maximum(1,np.abs(left[k])))) for k in left.dtype.names if k!='tag')
    trace_ok=metric!='st' or any((ROOT/release['attempt_path']).glob('timeline*.nsys-rep'))
    release_clocks=release['timings'][0]['device_cycles'] if release['timings'] else []
    checks.append(dict(matching_ids=matching_ids,matching_mass=matching_mass,max_scaled_payload_difference=difference,numerically_equivalent=bool(matching_ids and matching_mass and difference<1e-12),timeline_captured=trace_ok,release_device_cycles_zero=bool(release_clocks and not any(release_clocks)),metric=metric,matching_frozen_payload=same,device_cycles_recorded=bool(clocks and clocks[1]>0)))
result=dict(passed=all(x['numerically_equivalent'] and x['device_cycles_recorded'] and x['release_device_cycles_zero'] and x['timeline_captured'] for x in checks),checks=checks)
(ROOT/'provenance/numerical_gate.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
sys.exit(0 if result['passed'] else 1)
