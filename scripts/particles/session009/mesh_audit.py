"""Physical cell spacings and propagation coverage from every recorded leaf block."""
import numpy as np
from runtime import atomic

def audit(root,data):
 g=np.asarray(data['mb_geometry']);dx=(g[:,1::2]-g[:,::2])/32
 nearest=np.where(g[:,::2]>0,g[:,::2],np.where(g[:,1::2]<0,g[:,1::2],0))
 rmin=np.linalg.norm(nearest,axis=1)
 floors=[(56,.25),(72,.5),(104,1),(168,2),(296,4)]
 rows=[]
 for radius,limit in floors:
  selected=rmin<radius
  maximum=float(dx[selected].max())
  rows.append(dict(radius=radius,required_max_dx=limit,observed_max_dx=maximum,intersecting_blocks=int(selected.sum())))
  if maximum>limit*(1+1e-12):raise RuntimeError('incomplete propagation floor '+str(rows[-1]))
 if len(g)>11520:raise RuntimeError('mesh exceeds approved capacity')
 if not np.allclose(dx[:,0],dx[:,1]) or not np.allclose(dx[:,0],dx[:,2]):raise RuntimeError('unexpected anisotropic mesh')
 atomic(root/'evidence/mesh_audit.json',dict(blocks=len(g),cells=len(g)*32**3,minimum_dx=float(dx.min()),maximum_dx=float(dx.max()),domain=[-1024,1024],root_dx=8,configured_finest_dx=1/256,physical_levels_available=list(range(12)),wave_region=rows,extraction_radius=50,boundary_gauge_estimate=(1024-50)/np.sqrt(2),initial_grid_not_collapse_grid=True))
