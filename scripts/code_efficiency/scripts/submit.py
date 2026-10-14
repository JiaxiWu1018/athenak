#!/usr/bin/env python3
import os
"""Submit only authorized campaign jobs; dependencies serialize the timing cases."""
import argparse,json,subprocess
from pathlib import Path
ROOT=Path(os.environ.get('EFFICIENCY_SESSION',Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['gate','throughput','profile','device'],required=True);p.add_argument('--dependency');p.add_argument('--dependency-condition',choices=['afterok','afterany'],default='afterany');p.add_argument('--ranks',type=int);p.add_argument('--names',nargs='*');a=p.parse_args()
cases=json.loads((ROOT/('gates.json' if a.kind=='gate' else 'matrix.json')).read_text())
if a.names:cases=[x for x in cases if x['name'] in a.names]
if a.ranks:cases=[x for x in cases if x['ranks']==a.ranks]
if a.kind=='throughput':cases.sort(key=lambda x:(x['ranks'],x['n'],x['test'],x['name']))
if a.kind=='profile':
    cases=[x for x in cases if (x['test']>=4 or x['name'] in ['count_st_dep0_N1000000','count_st_dep1_N1000000','count_flat_dep0_N1000000','count_flat_dep1_N1000000','blocks_st_dep1_B16','blocks_st_dep1_B64','ranks_st_dep1_R2','ranks_st_dep1_R4']) and not x['mesh_only']]
    cases.sort(key=lambda x:(x['ranks'],x['n'],x['name']))
if a.kind=='device':
    cases=[x for x in cases if x['name'] in ['dynamic_ppc6','dynamic_ppc24','dynamic_ppc96','dynamic_ppc384','count_st_dep0_N1000000','count_flat_dep0_N1000000']]
previous=a.dependency;ledger=[]
for c in cases:
    receipt=ROOT/'results'/a.kind/c['name']/'receipt.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('valid'):continue
    trace=a.kind=='profile' and c['name'] in ['count_st_dep1_N1000000','count_st_dep0_N1000000','ranks_st_dep1_R4','dynamic_ppc24','sampling_point_R4']
    minutes=360 if c['n']>=100000000 else (120 if c['n']>=10000000 else 30)
    if a.kind in ['profile','device']:minutes=min(minutes,120)
    cmd=['sbatch','--parsable','--job-name',f'eff-{a.kind[:1]}-{c["name"][:18]}','--ntasks',str(c['ranks']),'--gres',f'gpu:{c["ranks"]}','--mem',f'{max(32*c["ranks"],128 if c["n"]>=100000000 else 32)}G','--time',str(minutes),'--output',str(ROOT/'logs'/f'{a.kind}_{c["name"]}.%j.log'),'--export',f'ALL,CASE={c["name"]},KIND={a.kind},TRACE={int(trace)}']
    if previous:cmd+=['--dependency',f'{a.dependency_condition}:{previous}']
    cmd+=[str(ROOT/'jobs/run.sbatch')]
    result=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    previous=result.stdout.strip().split(';')[0];row=dict(job_id=previous,case=c['name'],kind=a.kind,command=cmd);ledger.append(row)
    with (ROOT/'provenance/jobs.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
print(json.dumps(dict(jobs=ledger,last_job=previous),indent=2))
