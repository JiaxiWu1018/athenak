"""Allocated full-N reductions and ID-stable display extraction, one case per job."""
import argparse
import csv
import os
import sys
import numpy as np
from cfg import *
sys.path.insert(0, str(HERE.parent/'reconstruction_s06'))
import analyze as A
import reduce_run as R
import st_migration_lib as L
from analyze_extension import window_metrics

def entry(path):
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))

def frame_times(times):
    times=np.asarray(times);slots=np.rint(times/2).astype(int)
    assert len(times)==501 and np.array_equal(slots,np.arange(501))
    assert np.all(np.diff(times)>0) and times[0]==0 and abs(times[-1]-1000)<1e-7
    assert np.all((times-2*slots>=-.03)&(times-2*slots<=.25))

def reduce(case):
    work=ROOT/'runs'/case['id']/'production';out=work/'out'
    assert (work/'RUN_OK').exists()
    manifest=load(work/'OUTPUT_MANIFEST.json')
    expected={f['path']:f for f in manifest['files']}
    staged=load(ROOT/'evidence/transfers'/(case['id']+'.json'))
    assert staged['all_destination_checksums_verified']
    dest=ROOT/'reduced'/case['id'];dest.mkdir(parents=True,exist_ok=False)
    model=dict(np.load(ROOT/'samples/model.npz'))
    initpath=ROOT/'samples'/(case['id']+'.bin');_,initial=sample(initpath)
    assert len(initial)==case['n'] and np.array_equal(initial[:,0],np.arange(case['n']))
    im=initial[:,7];ix=initial[:,1:4];n=case['n']
    ic=np.average(ix,weights=im,axis=0);order=np.argsort(np.linalg.norm(ix-ic,axis=1))
    cohort=np.empty(n,dtype=np.int8)
    cohort[order]=np.minimum((np.cumsum(im[order])/im.sum()*10).astype(int),9)
    if case['paired']:
        parents=np.sort(np.random.default_rng(2026101505).choice(n//2,10000,replace=False))
        display=np.sort(np.r_[parents,parents+n//2])
    else:
        display=np.sort(np.random.default_rng(2026101505).choice(n,20000,replace=False))
    r0=np.linalg.norm(ix[display],axis=1)
    shells=[];cohorts=[];mixing=[];occupancy=[];sampling_checks=[];profiles=[];symmetry=[];sources=[]
    clouds=[];times=[];independent=[]
    paths=sorted((out/'pvtk').glob('*.part.vtk'));assert len(paths)==501
    for k,path in enumerate(paths):
        key=str(path.relative_to(work));bound=expected[key]
        # The staged receipt verified all files; recheck each native source here.
        assert path.stat().st_size==bound['bytes'] and sha(path)==bound['sha256']
        t,cycle,count,d=R.read_pvtk(str(path));rawtags=d['ptag'];tags=rawtags.astype(np.int64)
        assert count==n and np.array_equal(rawtags,tags) and np.all((tags>=0)&(tags<n))
        assert np.all(np.bincount(tags,minlength=n)==1)
        assert all(np.isfinite(v).all() for v in d.values())
        mass=d['prtcl_mass'];assert np.array_equal(mass,im[tags].astype(np.float32).astype(float))
        pos=np.column_stack([d[a] for a in 'xyz']);inverse=np.empty(n,dtype=np.int32)
        inverse[tags]=np.arange(n,dtype=np.int32)
        if k==0:assert t==0 and np.array_equal(pos,ix[tags].astype(np.float32).astype(float))
        row,mh,rho,r=A.sample_state(pos,mass,model,'proper')
        slot=round(t/2);row.update(time=t,cycle=cycle,nominal_time=2*slot);shells.append(row)
        times.append(t);profiles.append(mh);clouds.append(pos[inverse[display]].astype(np.float32))
        cc=cohort[tags];cm=np.bincount(cc,weights=mass,minlength=10)
        means=np.bincount(cc,weights=mass*r,minlength=10)/cm
        cohorts.append(dict(time=t,**{f'cohort{i+1}':float(v) for i,v in enumerate(means)}))
        if slot%50==0:
            ss,check=A.sampling_state(pos,mass)
            occupancy.extend(dict(time=t,**v) for v in ss);sampling_checks.append(dict(time=t,**check))
            ordered=np.argsort(r);current=np.empty(n,dtype=np.int8)
            current[ordered]=np.minimum((np.cumsum(mass[ordered])/mass.sum()*10).astype(int),9)
            transfer=np.bincount(cc*10+current,weights=mass,minlength=100).reshape(10,10)/cm[:,None]
            mixing.extend(dict(time=t,initial_cohort=a+1,current_mass_decile=b+1,mass_fraction=float(transfer[a,b]))
                          for a in range(10) for b in range(10))
            # Independent equal-weight order-statistic check, retaining rounding
            # brackets because native masses and their CDF are finite precision.
            sorted_r=np.sort(r);ind={}
            for q in (10,50,90):
                j=int(n*q/100)
                brackets=sorted_r[max(0,j-2):min(n,j+3)]
                assert brackets.min()-1e-10<=row[f'r{q}']<=brackets.max()+1e-10
                ind[f'r{q}_bracket']=[float(brackets.min()),float(brackets.max())]
            independent.append(dict(time=t,**ind))
        if case['paired']:
            half=n//2;a=inverse[:half];b=inverse[half:]
            defect=pos[a]+pos[b];pairm=mass[a]+mass[b]
            momentum=np.column_stack([d['u'+c] for c in 'xyz'])
            udef=momentum[a]+momentum[b]
            symmetry.append(dict(time=t,position_pair_sum_rms_M=float(np.sqrt(np.average(np.sum(defect**2,axis=1),weights=pairm))),
                position_pair_sum_max_M=float(np.linalg.norm(defect,axis=1).max()),
                covariant_momentum_pair_sum_rms=float(np.sqrt(np.average(np.sum(udef**2,axis=1),weights=pairm))),
                definition='Norm of x_i+x_partner, and u_i+u_partner; neither divided by two'))
        sources.append(dict(bound,time_M=t,cycle=cycle))
        print('PARTICLES',case['id'],k,t,flush=True)
    times=np.array(times);frame_times(times)
    for name,rows in [('shells',shells),('cohorts',cohorts),('cohort_mixing',mixing),('occupancy',occupancy)]:
        A.write_csv(dest/(name+'.csv'),rows)
    if symmetry:A.write_csv(dest/'pair_symmetry.csv',symmetry)
    np.savez_compressed(dest/'profiles.npz',time=times,edges=A.EDGES,mass_bins=np.array(profiles),
        equilibrium_mass=model['M0_over_M']*np.diff(np.interp(A.EDGES,model['riso_over_m'],model['proper'])))
    health=A.history(out/(case['id']+'.user.hst'));frame_times([h['time'] for h in health])
    A.write_csv(dest/'health.csv',health)
    ga=np.atleast_2d(np.loadtxt(out/(case['id']+'.z4c.user.hst')))
    assert len(ga)==len(health) and np.array_equal(ga[:,0],[h['time'] for h in health])
    A.write_csv(dest/'global_constraints.csv',[dict(time=float(v[0]),dt_completed=h['dt_completed'],
        constraint_usable=h['constraint_usable'],H_rms=float(np.sqrt(max(v[3],0)/v[10])),
        M_rms=float(np.sqrt(max(v[4],0)/v[10])))for v,h in zip(ga,health)])
    errors=[];checks=[]
    for path in sorted((out/'bin').glob('*.tme3d.*.bin')):
        rows,check=A.native_errors(path,'proper');errors.extend(rows);checks.append(check)
    assert len(checks)==11
    A.write_csv(dest/'field_errors.csv',errors)
    central=[]
    for path in sorted((out/'bin').glob('*.tme2d.*.bin')):
        fields=L.read_binary(str(path));rr,v,w=L.bin_flatten(fields,'tmunu_E',slice_x3=0.)
        mask=rr<.75;assert mask.any() and np.isfinite(v).all()
        central.append(dict(time=float(fields['time']),E_central_slice_r075=float(np.average(v[mask],weights=w[mask])),
                            cells=int(mask.sum()),radius_M=.75))
    frame_times([h['time'] for h in central]);A.write_csv(dest/'central_energy.csv',central)
    values=np.array([[r[f'r{q}'] for q in range(10,100,10)]for r in shells])
    windows=[]
    for width,family in ((100.,'100M'),(P,'P'),(2*P,'2P')):
        windows.extend(window_metrics(times,values,values[0],[(j*width,(j+1)*width)for j in range(int(1000/width))],family))
    A.write_csv(dest/'windows.csv',windows)
    cache=ROOT/'movie_cache'/case['id'];cache.mkdir(parents=True,exist_ok=False)
    clouds=np.asarray(clouds);assert clouds.shape==(501,20000,3)
    np.savez_compressed(cache/'trajectories.npz',ids=display,positions=clouds,times=times,initial_radius=r0)
    atomic(cache/'DATA_OK.json',dict(utc=now(),job=os.environ['SLURM_JOB_ID'],source=source(),
        cache=entry(cache/'trajectories.npz'),initial_sample=entry(initpath),frames=501,display_count=20000,
        full_count=n,complete_partner_pairs=case['paired'],display_seed=2026101505,
        max_abs_xy_M=float(np.abs(clouds[:,:,:2]).max()),color_range_M=[0,8.936231120010849],
        definition='Same fixed IDs at native saved times; all-z x-y projection; initial 3D coordinate radius colours.'))
    atomic(dest/'summary.json',dict(case=case,utc=now(),job=os.environ['SLURM_JOB_ID'],source=source(),
        states=501,endpoint_M=float(times[-1]),period_M=P,field_checks=checks,sampling_checks=sampling_checks,
        independent_equal_mass_quantile_brackets=independent,source_particle_outputs=sources,
        max_outside_finest_mass_fraction=max(h['outside_finest_mass_fraction']for h in shells),
        central_energy_definition='Coordinate-area-weighted deposited Eulerian energy at z=0,r<0.75M; not rest-mass density',
        initial_COM_M=ic.tolist(),baseline_definitions_unchanged=True))
    print('REDUCE_AND_CACHE_OK',case['id'],flush=True)

if __name__=='__main__':
    require_job();parser=argparse.ArgumentParser();parser.add_argument('index',type=int)
    reduce(CASES[parser.parse_args().index])
