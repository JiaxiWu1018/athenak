"""Exact destruction ledger, component coverage and separate removed covariant J."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from runtime import atomic

def assess_removals(runs,out,cutoff):
 seen=set();records=[];quality=[]
 for run in runs:
  for p in run.glob('*.prtcl_destroy.csv'):
   with p.open() as f:
    header=f.readline().lstrip('# ').strip().split(',')
    for row in csv.DictReader(f,fieldnames=header):
     if not row.get('time') or float(row['time'])>cutoff:continue
     tag=int(row['tag'])
     if tag in seen:raise RuntimeError('duplicate destruction tag in primary history')
     seen.add(tag);m=float(row['mass']);j=m*(float(row['x1'])*float(row['u2_cov'])-float(row['x2'])*float(row['u1_cov']))
     if not np.isfinite([m,j]).all() or m<=0:raise RuntimeError('invalid destruction mass/momentum')
     if row['reason']=='horizon':raise RuntimeError('AH removal present despite OFF input')
     records.append([float(row['time']),int(row['component']),m,j])
  for i in (0,1,2):
   for p in run.glob('*.horizon_consumer_'+str(i)+'.csv'):
    with p.open() as f:
     for r in csv.DictReader(f):
      if float(r['time'])<=cutoff:quality.append([float(r['time']),i,int(r['cycle'])]+[int(r[k]) for k in ('quality_geometry_ok','quality_persist_ok','association_ok','published_this_candidate')])
 arr=np.asarray(records).reshape(-1,4);q=np.asarray(quality).reshape(-1,7)
 np.savez(out/'removal_ledger_reduced.npz',records=arr,columns=['time','component','rest_mass','Jz_cov'])
 np.savetxt(out/'horizon_quality.csv',q,delimiter=',',header='time,object,cycle,geometry_ok,persistence_ok,association_ok,published')
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for i,label in enumerate(('envelope','left','right')):
  part=arr[arr[:,1]==i];part=part[np.argsort(part[:,0])] if len(part) else part
  if len(part):axes[0].plot(part[:,0],np.arange(1,len(part)+1),label=label);axes[1].plot(part[:,0],np.cumsum(part[:,3]),label=label)
 axes[0].set(xlabel='t/M_ref',ylabel='cumulative removed particle count');axes[1].set(xlabel='t/M_ref',ylabel='removed-matter Jz_cov (separate ledger)')
 if len(arr):axes[0].legend();axes[1].legend()
 fig.tight_layout();fig.savefig(out/'removal_history.png',dpi=170);plt.close(fig)
 coverage=[]
 ledger=out/'particle_components.csv'
 if ledger.exists():
  rows=np.loadtxt(ledger,delimiter=',',ndmin=2)
  for r in rows:
   i=int(r[1]);removed=int(((arr[:,1]==i)&(arr[:,0]<=r[0]+1e-8)).sum());initial=(3000000,1000000,1000000)[i]
   coverage.append(dict(time=float(r[0]),component=i,alive=int(r[2]),logged_removed=removed,matched=int(r[2])+removed==initial))
 atomic(out/'removal_coverage.json',dict(events=len(arr),component_census=coverage,all_available_snapshots_matched=all(r['matched'] for r in coverage),limitations='Exact at removal marking step, not an exact horizon/total angular momentum. Missing ledger coverage remains flagged; raw full death CSV retained. Search failures also remain in raw FastFlow logs; quality CSV describes candidates, not every failed search.'))
