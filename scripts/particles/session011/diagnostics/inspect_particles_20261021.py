import json, os, sys, socket
from pathlib import Path
import numpy as np
root=Path('/work1/eliasmost/jiaxiwu/gi_s011_amd_20261021')
sys.path.insert(0,str(root/'scripts'))
from runtime import checkpoint, require_compute
require_compute()
result={'job':os.environ['SLURM_JOB_ID'],'host':socket.gethostname(),'evolution_steps':0,'checkpoints':[]}
for number in (4,5):
    p=root/f'runs/segment_0003/rst/s11_amd.{number:05d}.rst'
    row=checkpoint(p,hashing=False)
    data=np.memmap(p,dtype='<f8',mode='r',offset=row['particle_offset'],shape=(10,row['particles']))
    row['fields']={}
    for i,name in enumerate(('mass','IPEN','x','ux','y','uy','z','uz','gid','tag')):
        bad=np.flatnonzero(~np.isfinite(data[i]))
        row['fields'][name]={'nonfinite':int(bad.size),'first_tags':data[9,bad[:10]].tolist()}
    tags=data[9].astype(np.int64)
    row['tags_exact_unique_in_range']=bool(np.array_equal(tags,data[9]) and len(np.unique(tags))==len(tags) and tags.min()>=0 and tags.max()<5000000)
    row['positive_weights']=bool(np.all(data[0]>0))
    bad=np.flatnonzero(~np.isfinite(data[1]))
    row['invalid_energy_components']={name:int(np.count_nonzero((data[9,bad]>=lo)&(data[9,bad]<hi))) for name,lo,hi in [('envelope',0,3000000),('clump1',3000000,4000000),('clump2',4000000,5000000)]}
    result['checkpoints'].append(row)
(root/f'evidence/nonfinite_20261021/result_{os.environ["SLURM_JOB_ID"]}.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
