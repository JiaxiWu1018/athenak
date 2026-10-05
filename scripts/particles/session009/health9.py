"""Finite-state planes and measured far-field coordinate characteristic estimates."""
import sys
import numpy as np
from runtime import atomic

def assess(root,runs,out):
 sys.path.insert(0,str(root/'athenak/vis/python'));import bin_convert
 rows=[]
 for run in runs:
  for plane in ('xy','xz','yz'):
   files=sorted(run.glob('*.z4c_'+plane+'.*.bin'))
   if not files:continue
   d=bin_convert.read_binary(str(files[-1]));g=np.asarray(d['mb_geometry']);fields=d['mb_data'];peaks=[];alpha=[]
   finite=all(np.isfinite(v).all() for v in fields.values())
   for i,geom in enumerate(g):
    # Blocks touching far-field buffer; interior tensor inversion, no M*v proxy.
    if np.max(np.abs(geom))<800:continue
    chi=np.asarray(fields['z4c_chi'][i]);alp=np.asarray(fields['z4c_alpha'][i])
    tensor=np.empty(chi.shape+(3,3))
    for a,b,name in ((0,0,'xx'),(0,1,'xy'),(0,2,'xz'),(1,1,'yy'),(1,2,'yz'),(2,2,'zz')):
     tensor[...,a,b]=tensor[...,b,a]=fields['z4c_g'+name][i]
    inv=np.linalg.inv(tensor)
    beta=np.stack([fields['z4c_beta'+a][i] for a in 'xyz'],axis=-1)
    # Largest lapse/gauge characteristic estimate for inherited 1+log, alpha>0.
    if np.any(chi<=0) or np.any(alp<=0):raise RuntimeError('invalid far-field geometry')
    speed=np.sqrt(2*alp[...,None]*chi[...,None]*np.diagonal(inv,axis1=-2,axis2=-1))+np.abs(beta)
    peaks.append(float(np.max(speed)));alpha.append([float(alp.min()),float(alp.max())])
   maximum=max(peaks,default=None)
   rows.append(dict(run=run.name,plane=plane,time=float(d['time']),finite=finite,far_field_gauge_speed_max=maximum,far_field_alpha_ranges=alpha,boundary_to_r50_estimate=974/maximum if maximum else None))
   if not finite:raise RuntimeError('nonfinite saved metric plane')
 atomic(out/'numerical_health.json',dict(samples=rows,method='Three saved central diagnostic planes at their latest times; maximum sqrt(2 alpha gamma^ii)+abs(beta^i) in blocks touching |coordinate|>=800. Outflow/Sommerfeld implementation has Khat sqrt2 weak-field mode. Planning boundary-travel estimate, not a proof for every full3D characteristic or boundary reflection.'))
