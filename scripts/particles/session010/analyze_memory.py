#!/usr/bin/env python3
"""Matched fixed-state robust slopes; do not extrapolate CUDA tests as Vista proof."""
import csv,json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from checkpoint_io import atomic

def fit(x,y):
 x=np.asarray(x)/60.;y=np.asarray(y)
 slopes=[]
 for i in range(len(x)):
  dt=x[i+1:]-x[i];good=dt>0
  slopes.extend(((y[i+1:]-y[i])[good]/dt[good]).tolist())
 return dict(theil_sen_MiB_min=float(np.median(slopes)),endpoint_MiB_min=float((y[-1]-y[0])/(x[-1]-x[0])),fit_minutes=float(x[-1]-x[0]),fit_samples=len(x),begin_MiB=float(y[0]),end_MiB=float(y[-1]),peak_MiB=float(y.max()))

def main():
 if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Allocated reduction required')
 s=Path(__file__).resolve().parents[4]
 receipt=json.loads((s/'evidence/compact_receipt.json').read_text());root=Path(receipt['root']);done=json.loads((root/'COMPLETE.json').read_text())
 warm=done['seed_checkpoint']['cycle']+done['memory_warmup_advances'];end=done['seed_checkpoint']['cycle']+done['memory_advances']-1
 report=dict(method='Same checkpoint/input, fixed mesh and5M particles,128 advances/arm. Fit solver advances32..126 after seed; exclude startup/final output. Theil-Sen pairwise median and endpoint slopes, not OLS sawtooth.',arms={},full_mesh_Vista_validation_pending=True)
 fig,axs=plt.subplots(2,1,figsize=(10,7),sharex=True)
 for name in ('memory_baseline','memory_patched'):
  rows=list(csv.DictReader((root/name/'memory.csv').open()));groups={}
  for r in rows:
   key=r['kind']+'/'+r['gpu'];groups.setdefault(key,[]).append(r)
  summary={}
  for key,items in groups.items():
   selected=[r for r in items if warm<=int(r['cycle'])<end]
   if len(selected)<3:continue
   x=[float(r['elapsed']) for r in selected];y=[float(r['used_MiB']) for r in selected]
   if key.startswith('gpu/') and max(y)<1024:continue
   summary[key]=fit(x,y)
   ix=1 if key.startswith('host/') else 0
   axs[ix].plot(np.asarray(x)/60,np.asarray(y)-y[0],label=name+'/'+key)
  if not summary or 'host/total' not in summary:raise RuntimeError('Insufficient matched memory samples')
  report['arms'][name]=summary
 for ax,label in zip(axs,('GPU memory change / MiB','Total Athena RSS change / MiB')):
  ax.set_ylabel(label);ax.legend(fontsize=7);ax.grid(alpha=.2)
 axs[-1].set_xlabel('Elapsed minutes within each arm');fig.tight_layout();fig.savefig(root/'matched_memory.png',dpi=170);fig.savefig(root/'matched_memory.pdf')
 before=report['arms']['memory_baseline'];after=report['arms']['memory_patched'];comparison={}
 for key in before.keys()&after.keys():
  b=before[key]['theil_sen_MiB_min'];a=after[key]['theil_sen_MiB_min']
  comparison[key]=dict(before=b,after=a,reduction_percent=100*(b-a)/b if b>0 else None)
 report['comparison']=comparison
 report['segmentation']='Vista must measure every GPU restart baseline, AMR/rank high-water occupancy, host RSS and residual growth. Reserve measured final checkpoint/manifest time and memory. Use T<=min(reviewed wall budget minus finalization, half observed memory headroom divided by conservative positive growth); flat short-test slopes do not authorize unlimited segments. Begin with a bounded short segment and re-evaluate at collapse and after restart.'
 atomic(root/'memory_growth.json',report);atomic(s/'evidence/memory_receipt.json',dict(passed=True,root=str(root),report=str(root/'memory_growth.json')))
 print(json.dumps(comparison,indent=2))
if __name__=='__main__':main()
