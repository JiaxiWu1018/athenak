"""Interim matched-step constraint comparison; no temporal-order claim."""
import os
from pathlib import Path
from campaign import ROOT, resource_snapshot
from analyze import completed, reduce_case, read_csv, window_average, write_csv
from state_io import require_job, write_json, sha

require_job()
out=ROOT/'analysis'/('short_constraints_'+os.environ['SLURM_JOB_ID'])
out.mkdir(exist_ok=False)
case='fixed_v1_1m_h010'
records=completed(case)
assert records[-1]['target']==100
assert all(r['checkpoint']['finite'] and r['checkpoint']['original_ids_masses']
           and r['checkpoint']['removals']==0 and r['solver']['failures']==0 for r in records)
summary=reduce_case(case,out/'reduced')
legacy=ROOT/'analysis/job_495334/reduced/legacy_1m_h010'
rows=[];sources=[]
for region,file in [('r10','health.csv'),('r16','health.csv'),('global','global_constraints.csv')]:
    old=legacy/file;new=out/'reduced'/case/file
    sources.extend([old,new]);a,b=read_csv(old),read_csv(new)
    for metric in ('H','M'):
        key=metric+'_rms'+('' if region=='global' else '_'+region)
        for lo,hi in ((0,20),(20,100)):
            av=window_average(a,key,lo,hi);bv=window_average(b,key,lo,hi)
            assert av is not None and bv is not None and av>0 and bv>=0
            rows.append(dict(region=region,constraint=metric,start=lo,end=hi,
                legacy_mean=av,fixed_node_mean=bv,ratio_new_old=bv/av))
write_csv(out/'constraint_comparison.csv',rows)
write_json(out/'RESULT.json',dict(endpoint=100,particles=1000000,nominal_h=.1,
    synchronized_both=True,primary_executable='fixed_v1 (unoptimized)',
    caveat='Interim single-seed result. Same nominal timestep and endpoint diagnostics; integrator, coupling, fixed-node interpolation and fallback acceptance differ. No temporal-order inference.',
    rows=rows,reduction=summary,resources=resource_snapshot(),script_sha256=sha(__file__),
    sources=[dict(path=str(p),sha256=sha(p)) for p in sorted(set(sources))]))
print(out,flush=True)
