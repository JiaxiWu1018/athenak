#!/usr/bin/env python3
"""One compact CUDA AMR update with current 5M physics, after matched memory arms."""
import json,os,re
from pathlib import Path
import numpy as np
from compact_checks import run
from checkpoint_io import atomic

def main():
 s=Path(__file__).resolve().parents[4];build=json.loads((s/'evidence/build_receipt.json').read_text());prior=s/'evidence/compact_12805'
 seed=json.loads((prior/'seed_full_precision/result.json').read_text())['checkpoint']
 root=s/'evidence'/('gpu_amr_'+os.environ['SLURM_JOB_ID']);root.mkdir()
 disable=['output'+str(i)+'/dt=-1' for i in range(1,22) if i not in (1,5,7)]
 args=['-r',seed['path'],'time/nlim=33','mesh_refinement/max_nmb_per_rank=400','time/stop_check_every=1','time/stop_file='+str(root/'cuda_amr/REQUEST_STOP'),'z4c/nrad_wave_extraction=0','fastflow/start_time_0=400','fastflow/start_time_1=400','fastflow/start_time_2=400']+disable
 case,cp=run(Path(build['patched']),s/'inputs/VALIDATION_ONLY_compact.athinput',root,'cuda_amr',3,args,limit=1200)
 assert cp['cycle']==33 and cp['particles']==5_000_000
 text=(case/'run.log').read_text();m=re.search(r'(\d+) MeshBlocks created, (\d+) deleted by AMR',text)
 assert m and int(m[1])+int(m[2])>0,'No actual AMR update occurred'
 n=5_000_000;f=Path(cp['path']);data=np.memmap(f,dtype='<f8',mode='r',offset=f.stat().st_size-n*10*8,shape=(10,n));order=np.argsort(data[9]);assert np.array_equal(data[9,order],np.arange(n))
 assert np.isfinite(data).all() and (data[0]>0).all() and (data[8]>=0).all() and (data[8]<cp['blocks']).all()
 before=Path(seed['path']);old=np.memmap(before,dtype='<f8',mode='r',offset=before.stat().st_size-n*10*8,shape=(10,n));oo=np.argsort(old[9]);assert np.array_equal(data[0,order],old[0,oo])
 atomic(root/'RESULT.json',dict(passed=True,job=os.environ['SLURM_JOB_ID'],ranks=3,checkpoint=cp,created=int(m[1]),deleted=int(m[2]),unique_tags=True,finite_state=True,exact_rest_weights=True,GIDs_in_leaf_range=True))
 atomic(s/'evidence/gpu_amr_receipt.json',dict(passed=True,root=str(root)))
 print('CUDA AMR CHECK PASSED',cp['blocks'],m.groups())
if __name__=='__main__':main()
