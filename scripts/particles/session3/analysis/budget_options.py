"""Build reviewable alternatives without changing any approved run deck."""
import json,sys,math
from pathlib import Path
root=Path(sys.argv[1]);budget=json.loads((root/'evidence/budget_preflight.json').read_text())
matrix=json.loads((root/'state/matrix.json').read_text())
costs={r['name']:r['node_hours'] for r in budget['matrix_estimates']}
fixed=sum(value for key,value in budget['reserves'].items() if key!='checkpoint_output_margin')
options=[]
for label,limit in [('original_full_matrix',425),('shortened_all_comparisons',200),('main_pair_only',200)]:
    cases=[];base=0.
    for case in matrix:
        if label=='main_pair_only' and not case['name'].startswith('main_'):continue
        periods=case['periods'] if label!='shortened_all_comparisons' else (2 if case['name'].startswith(('main_','tail_')) else 1)
        changed=dict(case,periods=periods,tlim=periods*case['P_ref'])
        cost=costs[case['name']]*periods/case['periods'];changed['estimated_node_hours']=cost
        cases.append(changed);base+=cost
    options.append(dict(name=label,proposed_ceiling=limit,projected_node_hours=1.25*base+fixed,
       cases=cases,approved=False,interpretation='Original coverage retained' if label=='original_full_matrix' else
       'Main/tail coverage 2P; mesh and N controls 1P; no five-period conclusion' if label=='shortened_all_comparisons' else
       'Five-period main pair; no tail, mesh or particle-number convergence controls'))
(root/'evidence/budget_options.json').write_text(json.dumps(dict(options=options,
    method='Original measured case costs scaled by duration; identical conservative fixed reserves; no deck changed'),indent=2)+'\n')
for option in options:print(option['name'],option['projected_node_hours'])
