"""Non-evolving reproduction of saved particle energies; run only under Slurm."""
import json,os,sys,struct,socket,csv
from pathlib import Path
import numpy as np
root=Path('/work1/eliasmost/jiaxiwu/gi_s011_amd_20261021')
sys.path.insert(0,str(root/'scripts'))
from runtime import checkpoint,require_compute
require_compute()
p=root/'runs/segment_0003/rst/s11_amd.00005.rst'
r=checkpoint(p,hashing=False)
with p.open('rb') as f:
    h=f.read(4*1024**2);offset=h.index(b'<par_end>')+10
    region=np.frombuffer(h,dtype='<f8',count=9,offset=offset+8).copy()
    loc=np.frombuffer(h,dtype='<i4',count=r['blocks']*4,offset=offset+252).reshape(-1,4).copy()
start=r['particle_offset']-32-r['blocks']*25*40**3*8
fields=np.memmap(p,dtype='<f8',mode='r',offset=start,shape=(r['blocks'],25,40,40,40))
data=np.memmap(p,dtype='<f8',mode='r',offset=r['particle_offset'],shape=(10,r['particles']))
bad=np.flatnonzero(~np.isfinite(data[1])); good=np.flatnonzero(np.isfinite(data[1]))[::max(1,r['particles']//330)][:330]
def sym(v):return np.array([[v[0],v[1],v[2]],[v[1],v[3],v[4]],[v[2],v[4],v[5]]])
def calc(part,width):
    gid=int(data[8,part]);pos=data[[2,4,6],part];u=data[[3,5,7],part]
    assert 0<=gid<r['blocks']
    ll=loc[gid];span=(region[3:6]-region[:3])/2.**int(ll[3]);lo=region[:3]+span*ll[:3];dx=span/32
    assert np.all(pos>=lo-1e-10) and np.all(pos<=lo+span+1e-10),(gid,pos,lo,span)
    base=np.clip(np.floor((pos-lo)/dx-.5).astype(int),-1,31)
    indexes=[np.arange(width)+base[a]+(1 if width==8 else 4) for a in range(3)]
    weights=[]
    for a in range(3):
        nodes=lo[a]+(indexes[a]-4+.5)*dx[a]
        weights.append(np.array([np.prod([(pos[a]-nodes[j])/(nodes[i]-nodes[j]) for j in range(width) if j!=i]) for i in range(width)]))
    raw=np.array(fields[gid][:,indexes[2]][:,:,indexes[1]][:,:,:,indexes[0]])
    raw[1:7]/=raw[0]
    vals=np.einsum('vkji,i,j,k->v',raw,weights[0],weights[1],weights[2])
    g=sym(vals[1:7]);alp=vals[18];beta=vals[19:22];eig=np.linalg.eigvalsh(g)
    inv=np.linalg.inv(g);u2=float(u@inv@u);bu=float(beta@u)
    a=-1/alp**2;b=bu/alp**2;c=1+u2-bu**2/alp**2;disc=b*b-a*c
    stable=alp*np.sqrt(1+u2)-bu if 1+u2>=0 else None
    return dict(alpha=float(alp),min_eigenvalue=float(eig.min()),u2=u2,discriminant=float(disc),stable_energy=None if stable is None else float(stable),nodal_finite=bool(np.isfinite(raw).all()),nodal_chi_min=float(raw[0].min()))
rows=[]
for part in np.concatenate((bad,good)):
    rows.append(dict(tag=int(data[9,part]),stored_finite=bool(np.isfinite(data[1,part])),stored_energy=float(data[1,part]) if np.isfinite(data[1,part]) else None,high=calc(part,8),linear=calc(part,2)))
result=dict(job=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),checkpoint=r,evolution_steps=0,rows=rows)
result['summary']={}
for label,selected in [('invalid',rows[:len(bad)]),('finite_sample',rows[len(bad):])]:
    result['summary'][label]={m:{'nonpositive_metric':sum(v[m]['min_eigenvalue']<=0 for v in selected),'negative_discriminant':sum(v[m]['discriminant']<0 for v in selected),'nonfinite_nodes':sum(not v[m]['nodal_finite'] for v in selected),'min_alpha':min(v[m]['alpha'] for v in selected),'min_metric_eigenvalue':min(v[m]['min_eigenvalue'] for v in selected)} for m in ('high','linear')}
# Full saved grid finiteness scan, streaming one block at a time, including ghosts.
result['grid_nonfinite_by_variable']=[0]*25
for block in fields:
    counts=np.count_nonzero(~np.isfinite(block),axis=(1,2,3))
    result['grid_nonfinite_by_variable']=[a+int(b) for a,b in zip(result['grid_nonfinite_by_variable'],counts)]
out=root/f'evidence/nonfinite_20261021/metric_{os.environ["SLURM_JOB_ID"]}.json'
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
