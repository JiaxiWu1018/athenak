#!/usr/bin/env python3
"""Submit the complete serialized campaign after a verified build job."""
import argparse,json,subprocess,os
from pathlib import Path
ROOT=Path(os.environ.get('EFFICIENCY_SESSION',Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser();p.add_argument('--build-job',required=True);p.add_argument('--gate2-job',required=True);p.add_argument('--gate4-job',required=True);a=p.parse_args()
matrix=json.loads((ROOT/'matrix.json').read_text());byname={c['name']:c for c in matrix}
previous=a.gate2_job;submitted=[];covered={k:set() for k in ['throughput','profile','device']}
def submit(kind,names,partition='debug'):
 global previous
 if not names:return
 args=['python3',str(ROOT/'scripts/submit.py'),'--kind',kind,'--names']+names+['--dependency',str(previous),'--partition',partition]
 # Every job requires the validated rebuild; four-rank jobs also require their gate.
 args+=['--gate-job',a.gate4_job if partition=='preempt' else a.build_job]
 result=json.loads(subprocess.check_output(args,text=True));submitted.extend(result['jobs']);covered[kind].update(x['case'] for x in result['jobs']);previous=result['last_job'] or previous
submit('throughput',[c['name'] for c in matrix if c['test']==1 and c['n']<=1000000])
submit('profile',['count_st_dep0_N1000000','count_st_dep1_N1000000'])
submit('device',['count_st_dep0_N1000000','count_flat_dep0_N1000000'])
submit('throughput',[c['name'] for c in matrix if c['test']==2])
submit('throughput',[c['name'] for c in matrix if c['ranks']==1 and c['test']>=4 and c['n']<=6291456])
small_profiles=['blocks_st_dep1_B16','blocks_st_dep1_B64','count_flat_dep0_N1000000','count_flat_dep1_N1000000']+[c['name'] for c in matrix if c['ranks']==1 and c['test']>=4 and c['n']<=6291456 and not c['mesh_only']]
submit('profile',small_profiles)
submit('device',['dynamic_ppc6','dynamic_ppc24'])
submit('throughput',[c['name'] for c in matrix if c['ranks']==1 and c['name'] not in covered['throughput']])
submit('throughput',[c['name'] for c in matrix if c['ranks']==2])
submit('throughput',[c['name'] for c in matrix if c['ranks']==4],partition='preempt')
submit('profile',['dynamic_ppc96','dynamic_ppc384'])
submit('profile',[c['name'] for c in matrix if c['ranks']==2])
submit('profile',[c['name'] for c in matrix if c['ranks']==4],partition='preempt')
submit('device',['dynamic_ppc96','dynamic_ppc384'])
closure=subprocess.check_output(['sbatch','--parsable','--dependency',f'afterany:{previous}',str(ROOT/'jobs/close.sbatch')],text=True).strip().split(';')[0]
record=dict(build_job=a.build_job,gate2_job=a.gate2_job,gate4_job=a.gate4_job,jobs=submitted,last_case_job=previous,closure_job=closure)
(ROOT/'provenance/verified_pipeline.json').write_text(json.dumps(record,indent=2)+'\n')
(ROOT/'provenance/final_pipeline_job.txt').write_text(previous+'\n');(ROOT/'provenance/closure_job.txt').write_text(closure+'\n')
print(json.dumps(dict(case_jobs=len(submitted),closure_job=closure,last_case_job=previous)))
