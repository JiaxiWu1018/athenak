"""Use measured AMD timings; never infer throughput from a different machine."""
import json,re,sys,math
from pathlib import Path
root=Path(sys.argv[1]);cases={}
for arm in ('frozen','live'):
    run=root/'runs'/('preflight_'+arm)
    state=json.loads((run/'segment_state.json').read_text())
    if not state['health']['valid'] or state['returncode']:
        raise RuntimeError('preflight failed: '+arm)
    log=(run/f'segment_{state["job_id"]}.log').read_text(errors='replace')
    cycles=re.findall(r'elapsed=\s*([0-9.eE+-]+)\s+cycle=(\d+)\s+time=\s*([0-9.eE+-]+)\s+dt=\s*([0-9.eE+-]+)',log)
    if len(cycles)<5:raise RuntimeError('too few throughput samples')
    # Use cycles >=20, excluding startup dumps and the final checkpoint.
    points=[(float(e),int(c),float(t),float(dt)) for e,c,t,dt in cycles if 20<=int(c)<100]
    if len(points)<4:raise RuntimeError('insufficient steady preflight timings')
    rates=[(q[0]-p[0])/(q[1]-p[1]) for p,q in zip(points[:-1],points[1:]) if q[1]>p[1]]
    rates.sort();conservative=rates[min(len(rates)-1,int(.9*len(rates)))]
    cases[arm]=dict(seconds_per_step=conservative,dt_min=min(p[3] for p in points),
                    observed_node_hours=state['node_hours'],job=state['job_id'],rates=rates,
                    output_bytes=sum(f.stat().st_size for f in (run/'out').rglob('*') if f.is_file()))
matrix=json.loads((root/'state/matrix.json').read_text());estimates=[]
for case in matrix:
    arm='live' if case['live'] else 'frozen';measured=cases[arm]
    # Controls receive the full baseline cost estimate until measured separately.
    steps=math.ceil(case['tlim']/measured['dt_min'])
    estimates.append(dict(name=case['name'],steps=steps,node_hours=steps*measured['seconds_per_step']/3600))
base=sum(c['node_hours'] for c in estimates)
# Cover output bursts, checkpoints, sampling/restart/half-dt pilots, builds and
# reductions with explicit reserves and a throughput margin.
pilot_cost=sum(.675*461.99570043659014/c['dt_min']*c['seconds_per_step']/3600 for c in cases.values())
# Per arm: continuous .30P, interrupted/restarted .30P, short .025P and
# half-timestep .025P (twice as many steps). Do not retain a six-hour guess
# once measured performance shows that the actual validation matrix costs more.
reserve=dict(builds=2.,pilots=1.25*pilot_cost+1.,reductions=12.,
             preflights=sum(c['observed_node_hours'] for c in cases.values()),
             checkpoint_output_margin=.25*base)
projected=base+sum(reserve.values())
report=dict(passed=projected<=200,budget_node_hours=200,projected_node_hours=projected,
            matrix_estimates=estimates,measured=cases,reserves=reserve,
            policy='90th-percentile measured step cost; baseline cost for every control; additional 25%; measured 0.675P-per-arm pilot cost plus 25% and startup reserve; explicit build/reduction/preflight costs')
(root/'evidence/budget_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
if not report['passed']:
    (root/'state/budget_review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
sys.exit(0 if report['passed'] else 2)
