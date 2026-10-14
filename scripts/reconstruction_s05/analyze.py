"""Validated mass-weighted shell, occupancy, CIC noise and native field reductions."""
import argparse
import csv
import json
import re
from pathlib import Path
import numpy as np
import reduce_run as R
import st_migration_lib as L
from common import ROOT, N, atomic, sample, require_job, sha
from reference import cloud, coordinates, HALF, DX, SIDE

QS=np.arange(1,10)/10
EDGES=np.arange(0,32.0001,.5)

def write_csv(path,rows):
    with open(path,'w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def cube(path):
    fd=L.read_binary(str(path));fields={n:np.full((SIDE,SIDE,SIDE),np.nan) for n in fd['var_names']}
    for m,g in enumerate(fd['mb_geometry']):
        names=fd['var_names'];shape=np.asarray(fd['mb_data'][names[0]][m]).shape
        nz,ny,nx=shape; spacing=np.array([(g[1]-g[0])/nx,(g[3]-g[2])/ny,(g[5]-g[4])/nz])
        if not np.allclose(spacing,DX):continue
        x=g[0]+spacing[0]*(np.arange(nx)+.5);y=g[2]+spacing[1]*(np.arange(ny)+.5);z=g[4]+spacing[2]*(np.arange(nz)+.5)
        keep=[np.where((v>=-HALF)&(v<HALF))[0] for v in (x,y,z)]
        if any(len(k)==0 for k in keep):continue
        target=[np.rint((v[k]+HALF)/DX-.5).astype(int) for v,k in zip((x,y,z),keep)]
        for name in names:
            block=np.asarray(fd['mb_data'][name][m]).transpose(2,1,0)
            fields[name][np.ix_(*target)]=block[np.ix_(*keep)]
    if any(not np.isfinite(a).all() for a in fields.values()):raise ValueError('Incomplete/nonfinite native cube '+str(path))
    return float(fd['time']),{n:a.ravel() for n,a in fields.items()}

def sample_state(pos,m,p,radial):
    centre=np.average(pos,weights=m,axis=0);r=np.linalg.norm(pos-centre,axis=1);ro=np.linalg.norm(pos,axis=1)
    order=np.argsort(r);cum=np.cumsum(m[order]);cum/=cum[-1]
    q=r[order[np.searchsorted(cum,QS)]]
    oo=np.argsort(ro);cw=np.cumsum(m[oo]);cw/=cw[-1]
    qo=ro[oo[np.searchsorted(cw,QS)]]
    bins=np.histogram(r,EDGES,weights=m)[0];vol=4*np.pi/3*np.diff(EDGES**3)
    row={'com_x':float(centre[0]),'com_y':float(centre[1]),'com_z':float(centre[2]),'com_mag':float(np.linalg.norm(centre)),
         'rest_mass':float(m.sum()),'rmax_origin':float(ro.max()),
         'outside_finest_mass_fraction':float(m[np.any(np.abs(pos)>32,axis=1)].sum()/m.sum()),
         'outside_profile_mass_fraction':float(m[r>=32].sum()/m.sum())}
    for f,a,b in zip(range(10,100,10),q,qo):row[f'r{f}']=float(a);row[f'r{f}_origin']=float(b)
    return row,bins,bins/vol,r

def sampling_state(pos,m):
    xyz=coordinates();rr=np.linalg.norm(xyz,axis=1)
    count=np.zeros(SIDE**3);ix=np.floor((pos+HALF)/DX).astype(int)
    inside=np.all((ix>=0)&(ix<SIDE),axis=1);key=(ix[:,0]*SIDE+ix[:,1])*SIDE+ix[:,2]
    count+=np.bincount(key[inside],minlength=SIDE**3)
    w=cloud(pos,m);w2=cloud(pos,m,squared=True)
    neff=np.divide(w*w,w2,out=np.zeros_like(w),where=w2>0)
    radius=json.loads((ROOT/'evidence/model.json').read_text())['Riso_over_M']
    near=np.maximum(np.abs(xyz)-DX/2,0);matter=np.sum(near*near,axis=1)<radius**2
    rows=[]
    for low,high in zip(EDGES[:-1],EDGES[1:]):
        use=(rr>=low)&(rr<high);initial=use&matter
        if not use.any():continue
        rows.append({'r':float((low+high)/2),'cells':int(use.sum()),
          'count_mean':float(count[use].mean()),'neff_mean':float(neff[use].mean()),
          'neff_mass_weighted':float(np.sum(neff[use]*w[use])/w[use].sum()) if w[use].sum()>0 else 0.,
          'initial_matter_cells':int(initial.sum()),
          'initial_matter_empty_fraction':float(np.mean(count[initial]==0)) if initial.any() else None})
    return rows,{'grid_cloud_mass':float(w.sum()),'outside_diagnostic_cube_mass_fraction':float(m[~inside].sum()/m.sum()),
                 'initial_matter_empty_fraction':float(np.mean(count[matter]==0))}

def native_errors(path,radial):
    t,f=cube(path);rr=np.linalg.norm(coordinates(),axis=1);ref=dict(np.load(ROOT/'reference'/(radial+'.npz')))
    eq=dict(np.load(ROOT/'reference/proper.npz'));energy=f['tmunu_E']
    rows=[]
    for low,high in zip(EDGES[:-1],EDGES[1:]):
        use=(rr>=low)&(rr<high)
        if not use.any():continue
        mean=float(energy[use].mean());target=float(ref['E'][use].mean())
        momentum=sum(f['tmunu_S'+a][use]**2 for a in 'xyz')
        diag=[f['tmunu_S'+a+a][use] for a in 'xyz'];trace=sum(diag)/3
        dev=sum((d-trace)**2 for d in diag)+2*sum(f['tmunu_S'+a][use]**2 for a in ('xy','xz','yz'))
        rows.append({'time':t,'r':float((low+high)/2),'E_mean':mean,'E_target_mean':target,
          'E_angular_rms':float(np.std(energy[use])),
          'E_target_error_rms':float(np.sqrt(np.mean((energy[use]-ref['E'][use])**2))),
          'E_equilibrium_error_rms':float(np.sqrt(np.mean((energy[use]-eq['E'][use])**2))),
          'J_coordinate_rms':float(np.sqrt(np.mean(momentum))),
          'stress_coordinate_anisotropy_rms':float(np.sqrt(np.mean(dev)))})
    support=rr<10;norm=np.linalg.norm(ref['E'][support])
    return rows,{'time':t,'E_relative_L2_target':float(np.linalg.norm(energy[support]-ref['E'][support])/norm),
      'E_relative_L2_equilibrium':float(np.linalg.norm(energy[support]-eq['E'][support])/np.linalg.norm(eq['E'][support]))}

def history(path):
    a=np.atleast_2d(np.loadtxt(path));rows=[]
    for v in a:
        # Columns: t,dt,V10,H2,M2,E1,E2,J2,V16,H2,M2,E1,E2,J2,Vcore,alpha,Vouter,alpha.
        row={'time':float(v[0]),'dt':float(v[1]),'constraint_usable':bool(v[0]==0 or v[1]>1e-6),
             'core_lapse':float(v[15]/v[14]),'outer_lapse':float(v[17]/v[16])}
        for rad,o in ((10,2),(16,8)):
            row.update({f'H_rms_r{rad}':float(np.sqrt(max(v[o+1],0)/v[o])),
                        f'M_rms_r{rad}':float(np.sqrt(max(v[o+2],0)/v[o])),
                        f'E_mean_r{rad}':float(v[o+3]/v[o])})
        rows.append(row)
    return rows

def analyze(case,until=None):
    out=ROOT/'runs'/case/'out';dest=ROOT/('reduced_interim' if until is not None else 'reduced')/case;dest.mkdir(parents=True,exist_ok=True)
    radial,method,seed=case.split('_');p=dict(np.load(ROOT/'samples/model.npz'))
    _,initial=sample(ROOT/'samples'/(case+'.bin'))
    im=initial[:,7];ix=initial[:,1:4];ic=np.average(ix,weights=im,axis=0)
    io=np.argsort(np.linalg.norm(ix-ic,axis=1));cohort=np.empty(N,dtype=np.int8)
    cohort[io]=np.minimum((np.cumsum(im[io])/im.sum()*10).astype(int),9)
    cohort_mass=np.bincount(cohort,weights=im,minlength=10)
    shells=[];profiles=[];sampling=[];sampling_checks=[];cohorts=[];sources=[]
    for path in sorted((out/'pvtk').glob('*.part.vtk')):
        t,cycle,n,d=R.read_pvtk(str(path));tags=d['ptag'].astype(np.int64)
        if until is not None and t>until+.25:continue
        if n!=N or not np.array_equal(np.sort(tags),np.arange(N)) or any(not np.isfinite(a).all() for a in d.values()):
            raise ValueError('Invalid native particle state '+str(path))
        m=d['prtcl_mass'];expected=initial[tags,7].astype(np.float32).astype(float)
        if not np.array_equal(m,expected):raise ValueError('VTK weights differ from per-ID initialization')
        pos=np.column_stack([d[a] for a in 'xyz']);row,mh,rho,r=sample_state(pos,m,p,radial)
        slot=round(t/2);skew=t-2*slot
        if not (-.03<=skew<=.25):raise ValueError('Particle cadence error')
        row.update(time=t,cycle=cycle,nominal_time=2*slot);shells.append(row);profiles.append(mh)
        cc=cohort[tags];mass_now=np.bincount(cc,weights=m,minlength=10)
        means=np.bincount(cc,weights=m*r,minlength=10)/mass_now
        cohorts.append(dict(time=t,**{f'cohort{i+1}':float(v) for i,v in enumerate(means)}))
        if slot in (0,50,100,150):
            ss,check=sampling_state(pos,m);sampling.extend([dict(time=t,**v) for v in ss]);sampling_checks.append(dict(time=t,**check))
        sources.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'time':t,'cycle':cycle})
    if not shells or shells[0]['time']!=0:raise ValueError('Initial state missing')
    times=np.array([r['time'] for r in shells]);nominal=np.array([r['nominal_time'] for r in shells])
    if not np.array_equal(nominal,np.arange(0,nominal[-1]+1,2)):raise ValueError('Missing/duplicate snapshot cadence')
    write_csv(dest/'shells.csv',shells);write_csv(dest/'cohorts.csv',cohorts);write_csv(dest/'occupancy.csv',sampling)
    hs=history(next(out.glob('*.user.hst')))
    if until is not None:hs=[r for r in hs if r['time']<=until+.25]
    write_csv(dest/'health.csv',hs)
    ga=np.atleast_2d(np.loadtxt(next(out.glob('*.z4c.hst'))))
    if until is not None:ga=ga[ga[:,0]<=until+.25]
    np.savez_compressed(dest/'profiles.npz',time=times,edges=EDGES,mass_bins=profiles,
                        target_mass=p['M0_over_M']*np.diff(np.interp(EDGES,p['riso_over_m'],p[radial])),
                        equilibrium_mass=p['M0_over_M']*np.diff(np.interp(EDGES,p['riso_over_m'],p['proper'])))
    write_csv(dest/'global_constraints.csv',[{'time':float(v[0]),'constraint_usable':bool(v[0]==0 or v[1]>1e-6),'H_rms':float(np.sqrt(max(v[3],0)/v[10])),
               'M_rms':float(np.sqrt(max(v[4],0)/v[10]))} for v in ga])
    errors=[];error_checks=[]
    for path in sorted((out/'bin').glob('*.tme3d.*.bin')):
        if until is not None:
            fd=L.read_binary(str(path))
            if float(fd['time'])>until+.25:continue
        rows,check=native_errors(path,radial);errors.extend(rows);error_checks.append(check)
    write_csv(dest/'field_errors.csv',errors)
    # Independent fixed-core check of initial history norms from saved 3D fields.
    initial_con=sorted((out/'bin').glob('*.con3d.*.bin'))[0]
    t,con=cube(initial_con);rr=np.linalg.norm(coordinates(),axis=1)
    checks={}
    for rad in (10,16):
        use=rr<rad;h=float(np.sqrt(np.mean(con['con_H'][use]**2)))
        # Native con_M stores the contracted squared norm.
        mm=float(np.sqrt(np.mean(np.maximum(con['con_M'][use],0))))
        if not np.isclose(h,hs[0][f'H_rms_r{rad}'],rtol=2e-6) or not np.isclose(mm,hs[0][f'M_rms_r{rad}'],rtol=2e-6):
            raise ValueError('History/3D initial constraint mismatch')
        checks[str(rad)]={'H_rms':h,'M_rms':mm}
    q=np.array([[s[f'r{i}'] for i in range(10,100,10)] for s in shells]);q0=q[0]
    windows=[]
    for low,high in ((0,100),(100,200),(200,300)):
        if high>nominal[-1]:continue
        use=(nominal>=low)&(nominal<=high);tt=times[use];qq=q[use]
        duration=tt[-1]-tt[0]
        mean=np.trapz(qq,tt,axis=0)/duration
        for f,a,b in zip(range(10,100,10),(qq.max(0)-qq.min(0))/(2*q0),(mean-q0)/q0):
            windows.append({'start':low,'end':high,'quantile':f,'half_range_fraction':float(a),'mean_shift_fraction':float(b)})
    write_csv(dest/'windows.csv',windows)
    atomic(dest/'summary.json',{'case':case,'endpoint_M':float(times[-1]),'initial_error':error_checks[0],
           'field_checks':error_checks,'sampling_checks':sampling_checks,'initial_constraint_crosscheck':checks,
           'initial_sample_sha256':sha(ROOT/'samples'/(case+'.bin')),'sources':sources,
           'max_outside_finest_mass_fraction':max(s['outside_finest_mass_fraction'] for s in shells),
           'definitions':{'radii':'COM coordinate radius weighted by rest mass','density':'rest mass per coordinate shell volume',
                          'momentum_stress':'coordinate covariant-component norms','constraints':'coordinate-volume RMS within fixed origin spheres',
                          'angular_scatter':'within finite radial bins; includes the radial variation inside a bin'}})
    print('REDUCED',case,len(shells),flush=True)

def main():
    require_job();ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--until',type=float);a=ap.parse_args();analyze(a.case,a.until)

if __name__=='__main__':main()
