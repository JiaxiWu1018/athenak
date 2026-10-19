import os,json
from pathlib import Path
import numpy as np
assert os.environ.get('SLURM_JOB_ID')
s=Path('/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013')
d=s/'session010_summary_20261019/workspace/runtime_s10/analysis/through_segment002_20261018'
r=json.loads((d/'REDUCTIONS.json').read_text())
tr=np.load(d/'track_0.npy');mesh=np.load(d/'mesh.npy')
rows=[]
for i,t in enumerate(r['particle_times']):
 if t>98:continue
 p=d/'cache/particles'/f'{i:04d}.npz'
 with np.load(p,allow_pickle=False) as z:
  tm=float(z['time']); ix=np.argmin(abs(tr[:,1]-tm)); mi=np.argmin(abs(mesh[:,0]-tm))
  rows.append(dict(index=i,t=tm,keys=z.files,points=len(z['tag']),tracker_nearest=tr[ix,[0,1,-1]].tolist(),mesh_nearest=mesh[mi].tolist()))
print(json.dumps(rows,indent=2))
(s/'movies/particles3d_to98_20261019_v1/evidence/cache_probe.json').write_text(json.dumps(rows,indent=2)+'\n')
