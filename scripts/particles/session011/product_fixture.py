"""Synthetic file-schema fixture only; no equations or particle trajectories evolved."""
import json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,sys.argv[1])
import products
root=Path(sys.argv[2]);root.mkdir()
run=root/'runs/segment_0000';(run/'trk').mkdir(parents=True);(root/'control').mkdir()
times=[0.,.25,.5,.75,75.];states=[];events=[];profiles=[];grid=[];transfer=[]
for cycle,time in enumerate(times):
    d=[6.,2.,.8,.4,.4][cycle];common=cycle>=3
    states.append(dict(cycle=cycle,time=time,dt=.001,blocks=10,max_rank_blocks=1,capacity=256,
                       valid_bits=4 if common else (3 if cycle else 0),ever_bits=7 if common else (3 if cycle else 0),
                       strict_common=int(common),tracker0_live=1,tracker1_live=1,separation=d,
                       x0=-d/2,y0=0.,z0=0.,x1=d/2,y1=.01*cycle,z1=0.,proper_exterior_line=d))
    for h in range(3):
        published=(common and h==2) or (not common and cycle>0 and h<2)
        radius=2. if h==2 else .1
        events.append(dict(state_cycle=cycle,state_time=time,horizon=h,published=int(published),
                           x=0. if h==2 else (-1 if h==0 else 1)*d/2,y=0.,z=0.,mass=.2+.001*cycle,
                           spin=.02,Jx=0.,Jy=0.,Jz=.01,Px=0.,Py=.1,Pz=0.,hrel=.01,lmax=0,
                           coefficients_a0_ac_as=f'{radius*np.sqrt(4*np.pi)};0;0;'))
    for component in range(3):
        for mask in (0,1):
            profiles.append(dict(cycle=cycle,time=time,component=component,shell=1,mask=mask,count=1,
                rest_mass=1.,En_trilinear=1.,Ecoord_trilinear=1.,Ecoord_native=1.,Px=0.,Py=0.,Pz=0.,
                Jx=0.,Jy=0.,Jz=.1,mx=0.,my=0.,mz=0.,mxx=1.,mxy=.1,mxz=0.,myy=.5,myz=0.,mzz=.1,m_ur=0.,invalid_metric_mass=0.))
        transfer.append(dict(cycle=cycle,time=time,component=component,shell=1,mass_in=0.,mass_out=0.,Jz_adv_in=0.,Jz_adv_out=0.,Jz_force_delta=0.,Jz_AH_advection=0.,Jz_AH_sweep=0.,unknown_AH_mass_steps=0.))
    for mask in (0,1,2):
        grid.append(dict(cycle=cycle,time=time,shell=1,mask=mask,proper_volume=1.,H2_volume=.001,M2_volume=.001,alpha_volume=1.,g_rr_volume=1.,beta_r_volume=0.,E_volume=1.,invalid_coordinate_volume=0.))
    data=np.zeros(3,dtype=[('tag','<i8'),('q','<f8',(8,))]);data['tag']=[0,3000000,4000000];data['q'][:,0:2]=1.;data['q'][:,2]=[3.,-d/2,d/2]
    header=f'ATHENAK_TRK2 version=2 endian=native record_bytes=72 count=3 cycle={cycle} time={time}\n'.encode().ljust(512,b'\0')
    (run/'trk'/f'x.c{cycle:010d}.trk2').write_bytes(header+data.tobytes())
for name,data in [('gi_status',states),('ah_events_0',events),('gi_particles',profiles),('gi_grid',grid),('gi_shell_transfer',transfer)]:
    pd.DataFrame(data).to_csv(run/f'x.{name}.csv',index=False)
pd.DataFrame([dict(cycle=i,time=times[i],event=e,restart_number=i) for i,e in enumerate(['initial','first_individual_pair','first_inward_d_lt_1','strict_common'])]).to_csv(run/'x.gi_milestones.csv',index=False)
latest=dict(time=75.,cycle=4,particles=3,removed=[0,0,0,0])
for name in ['ENDPOINT_75','latest_checkpoint']:(root/'control'/f'{name}.json').write_text(json.dumps(latest))
(run/'checkpoint_manifest.json').write_text(json.dumps([latest]));(run/'population_audit.json').write_text(json.dumps(dict(exact_partition=True)))
print(root)
