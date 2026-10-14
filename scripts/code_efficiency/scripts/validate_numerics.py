#!/usr/bin/env python3
"""Compare release and instrumented frozen trajectories, plus liveness/source gates."""
import json,sys
from pathlib import Path
from run_case import run,ROOT
matrix=json.loads((ROOT/'matrix.json').read_text());checks=[]
for metric in ['st','flat']:
    c=next(x.copy() for x in matrix if x['name']==f'count_{metric}_dep0_N10000')
    c['name']='numeric_'+metric
    release=run(c,'validation_release',target_override=1.5)
    diagnostic=run(c,'device',target_override=1.5)
    same=release['valid'] and diagnostic['valid'] and release['initial'][0]['payload_hash']==diagnostic['initial'][0]['payload_hash'] and release['final'][0]['payload_hash']==diagnostic['final'][0]['payload_hash']
    clocks=diagnostic['timings'][0]['device_cycles'] if diagnostic['timings'] else []
    release_clocks=release['timings'][0]['device_cycles'] if release['timings'] else []
    checks.append(dict(release_device_cycles_zero=bool(release_clocks and not any(release_clocks)),metric=metric,matching_frozen_payload=same,device_cycles_recorded=bool(clocks and clocks[1]>0)))
result=dict(passed=all(x['matching_frozen_payload'] and x['device_cycles_recorded'] and x['release_device_cycles_zero'] for x in checks),checks=checks)
(ROOT/'provenance/numerical_gate.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
sys.exit(0 if result['passed'] else 1)
