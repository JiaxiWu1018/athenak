#!/usr/bin/env python3
"""Source-backed Session 011 figures and exact-state particle/AH renders.

Run only on allocated compute. Input is the session's runs tree (raw stays AMD).
No inferred horizon surface is rendered during a missing measurement interval.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BG='#071A2B'
COLORS=['#8E5AB5','#FF9F1C','#31B7E5']
AH_RENDERING='Accepted spectral AH surfaces overlaid above particle markers; particle/surface depth is not resolved'

def require_compute():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm allocation required')

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def load_csvs(runs,pattern,keys,valid_column=None,priority=None):
    tables=[]
    for run in runs:
        for path in sorted(run.glob(pattern)):
            data=pd.read_csv(path)
            data['segment']=run.name;data['source']=str(path)
            tables.append(data)
    if not tables:return pd.DataFrame()
    data=pd.concat(tables,ignore_index=True)
    if valid_column:
        data['_valid']=np.isfinite(data[valid_column])
        data=data.sort_values('_valid',kind='stable')
    if priority:data=data.sort_values(priority,kind='stable')
    if keys is None:return data
    return data.drop_duplicates(keys,keep='last').sort_values(keys).reset_index(drop=True)

def harmonic(l,m,theta,phi):
    """Literal native Wigner-d convention from utils/spherical_harm.hpp."""
    value=np.zeros(np.broadcast_shapes(np.shape(theta),np.shape(phi)))
    for k in range(max(0,m),min(l+m,l)+1):
        value+=((-1.)**k*np.cos(theta/2)**(2*l+m-2*k)*np.sin(theta/2)**(2*k-m)
                /(math.factorial(l+m-k)*math.factorial(l-k)*math.factorial(k)*math.factorial(k-m)))
    value*=math.sqrt((2*l+1)/(4*np.pi))*math.factorial(l)*math.sqrt(math.factorial(l+m)*math.factorial(l-m))
    return value*np.exp(1j*m*phi)

def radius(row,theta,phi):
    lm=int(row['lmax']);n=lm+1;lp=n*n
    coeff=np.fromstring(row['coefficients_a0_ac_as'],sep=';')
    if len(coeff)!=n+2*lp:raise ValueError('incomplete accepted surface coefficients')
    result=np.zeros(np.broadcast_shapes(np.shape(theta),np.shape(phi)))
    for l in range(n):
        result+=coeff[l]*harmonic(l,0,theta,phi).real
        for m in range(1,l+1):
            y=harmonic(l,m,theta,phi);q=l*n+m
            result+=np.sqrt(2)*(coeff[n+q]*y.real+coeff[n+lp+q]*y.imag)
    if np.any(~np.isfinite(result)) or np.any(result<=0):raise ValueError('invalid accepted spectral graph')
    return result

def trk(path):
    with path.open('rb') as f:header=f.read(512).rstrip(b'\0').decode()
    n=int(re.search(r'count=(\d+)',header)[1]);cycle=int(re.search(r'cycle=(\d+)',header)[1]);time=float(re.search(r'time=([^\s]+)',header)[1])
    dtype=np.dtype([('tag','<i8'),('q','<f8',(8,))])
    if path.stat().st_size!=512+n*dtype.itemsize:raise ValueError('truncated cohort')
    records=np.fromfile(path,dtype=dtype,offset=512,count=n)
    # q = mass, derived IPEN, x,y,z,u1,u2,u3. Energy is not used for geometry.
    physical=records['q'][:,[0,2,3,4,5,6,7]]
    if (len(np.unique(records['tag']))!=n or not np.isfinite(physical).all()
        or np.any(records['q'][:,0]<=0) or np.any(records['tag']<0)
        or np.any(records['tag']>=5_000_000)):raise ValueError('invalid cohort records')
    return cycle,time,records

def surface(ax,row):
    theta,phi=np.meshgrid(np.linspace(0,np.pi,37),np.linspace(0,2*np.pi,73),indexing='ij')
    r=radius(row,theta,phi)
    x=row['x']+r*np.sin(theta)*np.cos(phi);y=row['y']+r*np.sin(theta)*np.sin(phi);z=row['z']+r*np.cos(theta)
    ax.plot_surface(x,y,z,color='black',linewidth=0,edgecolor='none',shade=False,alpha=1,zorder=10)

def draw_state(ax,records,horizons,limit,time,caption=True):
    q=records['q'];xyz=q[:,2:5];tags=records['tag']
    group=np.where(tags<3_000_000,0,np.where(tags<4_000_000,1,2))
    visible=np.ones(len(tags),dtype=bool)
    for row in horizons:
        delta=xyz-np.array([row['x'],row['y'],row['z']]);r=np.linalg.norm(delta,axis=1)
        # Only evaluate nearby particles; use the same conservative coefficient bound.
        lm=int(row['lmax']);n=lm+1;lp=n*n;c=np.fromstring(row['coefficients_a0_ac_as'],sep=';')
        bound=sum(abs(c[l])*np.sqrt((2*l+1)/(4*np.pi))+
                  sum(np.sqrt(2)*np.sqrt((2*l+1)/(4*np.pi))*(abs(c[n+l*n+m])+abs(c[n+lp+l*n+m])) for m in range(1,l+1)) for l in range(n))
        near=np.flatnonzero(r<bound*1.00000001)
        if len(near):
            rr=np.maximum(r[near],1e-300)
            theta=np.arccos(np.clip(delta[near,2]/rr,-1,1));phi=np.arctan2(delta[near,1],delta[near,0])
            visible[near[r[near]<radius(row,theta,phi)]]=False
    ax.set_facecolor(BG);ax.set_proj_type('persp',focal_length=1.25)
    ax.view_init(elev=22,azim=-58,roll=0)
    # mplot3d sorts whole scatter collections against surfaces, which can paint
    # every marker above an AH regardless of individual particle depth. Display
    # the exact measured surface as an explicit overlay; never enlarge its shape.
    ax.computed_zorder=False
    for c in range(3):
        use=visible&(group==c)
        ax.scatter(*xyz[use].T,s=1.45 if c==0 else 2.1,alpha=.34 if c==0 else .82,
                   color=COLORS[c],edgecolors='none',linewidths=0,depthshade=False,rasterized=True,zorder=c+1)
    for row in horizons:surface(ax,row)
    ax.set(xlim=(-limit,limit),ylim=(-limit,limit),zlim=(-limit,limit));ax.set_box_aspect((1,1,1));ax.set_axis_off()
    if caption:ax.text2D(.025,.95,f'Saved state  t = {time:.4f} Mref',transform=ax.transAxes,color='white',fontsize=12)
    return dict(saved_time=time,ah_rendering=AH_RENDERING,unknown_native_energy_count=int((~np.isfinite(q[:,1])).sum()),alive_cohort=[int((group==c).sum()) for c in range(3)],visible=[int((visible&(group==c)).sum()) for c in range(3)],accepted_horizons=[int(r['horizon']) for r in horizons])

def intervals(cycles,valid):
    selected=np.flatnonzero(valid)
    if not len(selected):return []
    # Compact diagnostic samples normally span >1 integration cycle. Break on
    # invalid observations and segment boundaries, supplied by the caller.
    return np.split(selected,np.flatnonzero(np.diff(selected)>1)+1)

def savefig(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=200,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)

def complete_ledgers(sums,status):
    # Native sparse histograms omit empty bins. A vanished component is zero,
    # while an unavailable horizon mask remains unknown.
    index=pd.MultiIndex.from_product([status.cycle,range(3),range(5)],names=['cycle','component','mask'])
    sums=sums.set_index(['cycle','component','mask']).reindex(index).reset_index()
    missing=sums.time.isna();lookup=status.set_index('cycle')
    sums['time']=sums.cycle.map(lookup.time)
    valid=sums.cycle.map(lookup.valid_bits).to_numpy(dtype=int)
    ever=sums.cycle.map(lookup.ever_bits).to_numpy(dtype=int)
    common=sums.cycle.map(lookup.strict_common).to_numpy(dtype=bool)
    mask=sums['mask'].to_numpy(dtype=int);required=np.where(common,4,7)
    known=(mask==0)|((mask==1)&((ever&required&~valid)==0))
    known|=(mask>=2)&((valid&(1<<np.maximum(mask-2,0)))!=0)
    columns=[c for c in sums if c not in ('cycle','time','component','mask')]
    sums.loc[missing&known,columns]=0.
    if 'invalid_metric_mass' in sums:
        sums.loc[sums.invalid_metric_mass>0,['En_trilinear','Ecoord_trilinear']]=np.nan
    if 'native_energy_unknown_shells' in sums:
        sums.loc[sums.native_energy_unknown_shells>0,'Ecoord_native']=np.nan
    return sums

def aggregate_particles(particles):
    particles=particles.copy()
    particles['native_energy_unknown_shells']=(~np.isfinite(particles.Ecoord_native)).astype(int)
    keys=['cycle','time','component','mask']
    sums=particles.groupby(keys,dropna=False).sum(numeric_only=True,min_count=1).reset_index()
    sums.loc[sums.native_energy_unknown_shells>0,'Ecoord_native']=np.nan
    return sums

def amd_job_ids(root):
    """Discover AMD allocations only; copied foreign scheduler IDs can collide."""
    jobs=set()
    for path in sorted((root/'evidence').glob('*/RESULT.json')):
        # These are one-way copies of foreign receipts, not AMD allocations.
        if path.parent.name.startswith(('perseus_','vista_','anta_')):continue
        row=json.loads(path.read_text())
        if row.get('scheduler_cluster','amd')!='amd':continue
        if row.get('job_id'):jobs.add(str(row['job_id']))
    for path in (root/'evidence').glob('*.slurm.log'):
        if path.name.startswith(('perseus_','vista_','anta_')):continue
        match=re.search(r'_(\d+)\.slurm\.log$',path.name)
        if match:jobs.add(match[1])
    for run in sorted((root/'runs').glob('segment_*')):
        text=(run/'provenance.txt').read_text() if (run/'provenance.txt').exists() else ''
        match=re.search(r'JobId=(\d+)',text)
        if match:jobs.add(match[1])
    return sorted(jobs)

def reduce(root,out):
    runs=sorted(p for p in (root/'runs').glob('segment_*') if (p/'checkpoint_manifest.json').exists())
    if not runs:raise RuntimeError('no verified production segment')
    status=load_csvs(runs,'*.gi_status.csv',['cycle'],priority='valid_bits')
    events=load_csvs(runs,'*.ah_events_*.csv',['state_cycle','horizon'])
    particles=load_csvs(runs,'*.gi_particles.csv',['cycle','component','shell','mask'],'rest_mass')
    grid=load_csvs(runs,'*.gi_grid.csv',['cycle','shell','mask'],'proper_volume')
    transfer=load_csvs(runs,'*.gi_shell_transfer.csv',None)
    transfer=transfer.groupby(['cycle','time','component','shell']).sum(numeric_only=True).reset_index()
    accepted=events[events.published==1].copy()
    status.to_csv(out/'state_index.csv',index=False)
    accepted.to_csv(out/'accepted_horizons.csv',index=False)
    latest=json.loads((root/'control/latest_checkpoint.json').read_text())
    if status.empty or abs(status.time.max()-latest['time'])>1e-9:raise RuntimeError('diagnostics do not reach verified endpoint')
    # Resolve duplicate restart-boundary status in favor of an actual measurement
    # at that exact state; gaps on subsequent steps remain visible in event records.
    vector=status[['x1','y1','z1']].to_numpy()-status[['x0','y0','z0']].to_numpy()
    valid=(status.tracker0_live==1)&(status.tracker1_live==1)&(status.strict_common==0)
    valid &= np.isfinite(vector).all(axis=1)
    valid &= (status.ever_bits.to_numpy(dtype=int)&3&~status.valid_bits.to_numpy(dtype=int))==0
    phase=np.full(len(status),np.nan);omega=phase.copy();turns=0.;post_turns=0.;gaps=[];post_valid=False
    first=[float(accepted.loc[accepted.horizon==h,'state_time'].min())
           if (accepted.horizon==h).any() else None for h in (0,1)]
    both=max(first) if all(x is not None for x in first) else math.inf
    for indices in intervals(status.cycle.to_numpy(),valid.to_numpy()):
        # Do not differentiate across a segment boundary/requalification gap.
        split=np.flatnonzero(np.diff(status.time.to_numpy()[indices])>.06)+1
        for ids in np.split(indices,split):
            if len(ids)<2:continue
            p=np.unwrap(np.arctan2(vector[ids,1],vector[ids,0]));phase[ids]=p
            omega[ids]=np.gradient(p,status.time.to_numpy()[ids])
            turns+=(p[-1]-p[0])/(2*np.pi)
            ids_post=np.flatnonzero(status.time.to_numpy()[ids]>=both)
            if len(ids_post)>1:
                post_turns+=(p[ids_post[-1]]-p[ids_post[0]])/(2*np.pi);post_valid=True
            gaps.append(dict(start=float(status.time.iloc[ids[0]]),end=float(status.time.iloc[ids[-1]])))
    orbit=status[['cycle','time','separation','proper_exterior_line']].copy()
    orbit['phase']=phase;orbit['Omega']=omega
    orbit['radial_over_tangential']=np.nan
    for ids in intervals(status.cycle.to_numpy(),np.isfinite(omega)):
        if len(ids)>2:
            dd=np.gradient(status.separation.to_numpy()[ids],status.time.to_numpy()[ids])
            orbit.loc[ids,'radial_over_tangential']=np.abs(dd)/np.maximum(np.abs(status.separation.to_numpy()[ids]*omega[ids]),1e-15)
    orbit.to_csv(out/'orbit.csv',index=False)
    common_rows=status[status.strict_common==1]
    common=float(common_rows.time.min()) if len(common_rows) else None
    # Exact event time takes priority over compact status cadence.
    for run in runs:
        for path in run.glob('*.gi_milestones.csv'):
            with path.open() as f:
                for row in csv.DictReader(f):
                    if 'strict_common' in row['event']:common=float(row['time'])
    fig,axs=plt.subplots(2,1,figsize=(8,6),sharex=True)
    for h,label in ((0,'BH 1'),(1,'BH 2'),(2,'Common')):
        a=accepted[accepted.horizon==h]
        if h==2:a=a[a.state_time>=common] if common is not None else a.iloc[:0]
        color=COLORS[h+1] if h<2 else 'black'
        # Scatter does not conceal rejected-search gaps with connecting lines.
        axs[0].plot(a.state_time,a.mass,'.',ms=1,color=color,label=label)
        axs[1].plot(a.state_time,a.spin,'.',ms=1,color=color)
    axs[0].set_ylabel('Horizon mass / Mref');axs[1].set_ylabel('Coordinate-rotation spin')
    axs[1].set_xlabel('State time / Mref');axs[0].legend();savefig(fig,out/'horizon_mass_spin')
    fig,axs=plt.subplots(3,1,figsize=(8,8),sharex=True)
    axs[0].plot(orbit.time,orbit.separation,label='Coordinate');axs[0].plot(orbit.time,orbit.proper_exterior_line,label='Exterior proper line');axs[0].legend()
    axs[0].set_ylabel('Separation / Mref');axs[1].plot(orbit.time,orbit.Omega);axs[1].set_ylabel('Omega Mref')
    axs[2].plot(orbit.time,orbit.radial_over_tangential);axs[2].set_ylabel('|d_dot| / |d Omega|');axs[2].set_xlabel('State time / Mref');savefig(fig,out/'orbit_diagnostics')
    sums=aggregate_particles(particles)
    sums=complete_ledgers(sums,status)
    sums.to_csv(out/'component_ledgers.csv',index=False)
    norms=grid[grid['mask']==1].groupby(['cycle','time']).sum(numeric_only=True,min_count=1).reset_index()
    norms['H_RMS']=np.sqrt(norms.H2_volume/norms.proper_volume);norms['M_RMS']=np.sqrt(norms.M2_volume/norms.proper_volume)
    for cut,last_shell in ((8,64),(32,88)):
        inner=grid[(grid['mask']==1)&(grid.shell<last_shell)].groupby(['cycle','time']).sum(numeric_only=True,min_count=1).reset_index()
        inner[f'H_RMS_R{cut}']=np.sqrt(inner.H2_volume/inner.proper_volume)
        inner[f'M_RMS_R{cut}']=np.sqrt(inner.M2_volume/inner.proper_volume)
        norms=norms.merge(inner[['cycle',f'H_RMS_R{cut}',f'M_RMS_R{cut}']],on='cycle',how='left')
    norms.to_csv(out/'exterior_constraints.csv',index=False)
    fig,axs=plt.subplots(2,1,figsize=(8,6),sharex=True)
    for c,label in enumerate(('Envelope','Clump 1','Clump 2')):
        p=sums[(sums.component==c)&(sums['mask']==0)]
        axs[0].plot(p.time,p.rest_mass,label=label,color=COLORS[c]);axs[1].plot(p.time,p.Jz,color=COLORS[c])
    axs[0].set_ylabel('Surviving rest mass');axs[1].set_ylabel('Surviving matter Jz(origin)');axs[1].set_xlabel('State time / Mref');axs[0].legend();savefig(fig,out/'matter_ledgers')
    fig,ax=plt.subplots(figsize=(8,4));ax.semilogy(norms.time,norms.H_RMS,label='H RMS');ax.semilogy(norms.time,norms.M_RMS,label='M RMS');ax.legend();ax.set(xlabel='State time / Mref',ylabel='Proper-volume exterior norm');savefig(fig,out/'exterior_constraints')
    # Store full shell observables for model-dependent wake/potential analysis.
    particles[particles.component==0].to_csv(out/'envelope_shells.csv',index=False)
    grid.to_csv(out/'metric_shells.csv',index=False)
    transfer.to_csv(out/'shell_transfers.csv',index=False)
    metrics=dict(endpoint=latest['time'],cycle=latest['cycle'],first_individual_times=first,
                 strict_common_time=common,observed_turns=turns if gaps else None,
                 post_individual_turns=post_turns if post_valid else None,
                 valid_orbit_intervals=gaps,particles_alive=latest['particles'],removed=latest['removed'],
                 horizon_accepted_counts={str(h):int((accepted.horizon==h).sum()) for h in range(3)},
                 source_runs=[str(p) for p in runs],endpoint_checkpoint=latest)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    return runs,status,events,metrics

def render(runs,events,out,common,make_movies):
    candidates={}
    for run in runs:
        for path in sorted((run/'trk').glob('*.trk2')):
            cycle,time,_=trk(path);candidates[cycle]=(time,path)
    frames=sorted((c,t,p) for c,(t,p) in candidates.items())
    if not frames:raise RuntimeError('no exact cohort frames')
    horizons={(int(c),int(h)):dict(r.to_dict(),state_cycle=int(c),horizon=int(h))
              for (c,h),r in events[events.published==1].set_index(['state_cycle','horizon']).iterrows()}
    records=[]
    for cycle,time,path in frames:
        hs=[horizons[cycle,h] for h in range(3) if (cycle,h) in horizons and (common is None or time<common or h==2)]
        if any(abs(float(h['state_time'])-time)>1e-10 for h in hs):
            raise RuntimeError('horizon and cohort physical-state timestamps disagree')
        records.append(dict(cycle=cycle,time=time,path=str(path),horizons=hs))
    wanted=[0];observed=[True]
    for label in ('first_individual_pair','first_inward_d_lt_1','strict_common'):
        found=[]
        for run in runs:
            for path in run.glob('*.gi_milestones.csv'):
                with path.open() as f:
                    found += [int(r['cycle']) for r in csv.DictReader(f) if label in r['event']]
        wanted.append(min(found) if found else records[-1]['cycle']);observed.append(bool(found))
    fig=plt.figure(figsize=(16,9),facecolor=BG)
    qa=[]
    for i,cycle in enumerate(wanted):
        item=next((r for r in records if r['cycle']==cycle),None)
        if item is None:raise RuntimeError('mandatory event lacks an exact-cycle cohort frame')
        _,time,data=trk(Path(item['path']))
        ax=fig.add_subplot(2,2,i+1,projection='3d')
        qa.append(draw_state(ax,data,item['horizons'],24 if i==0 else 4,time))
        label=('Initialization','Individual horizons','First inward d<1','Strict common')[i]
        if not observed[i]:label+=' not observed; final saved state'
        ax.text2D(.025,.87,label,transform=ax.transAxes,color='white',fontsize=10)
    fig.subplots_adjust(0,0,1,1,0,0);fig.savefig(out/'four_states.png',dpi=150,facecolor=BG);fig.savefig(out/'four_states.pdf',facecolor=BG);plt.close(fig)
    (out/'snapshot_manifest.json').write_text(json.dumps(dict(selected_cycles=wanted,events_observed=observed,frames=qa),indent=2)+'\n')
    if not make_movies:return
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    for name,limit in (('wide',24),('close',4)):
        directory=out/f'frames_{name}';directory.mkdir()
        concat=[];manifest=[]
        for i,item in enumerate(records):
            _,time,data=trk(Path(item['path']))
            fig=plt.figure(figsize=(16,9),dpi=120,facecolor=BG)
            ax=fig.add_axes([0,0,1,1],projection='3d',facecolor=BG)
            details=draw_state(ax,data,item['horizons'],limit,time)
            ax.text2D(.025,.035,'Fixed particle IDs; exact-cycle AH geometry overlaid above particles',transform=ax.transAxes,color='white',fontsize=10)
            path=directory/f'frame_{i:05d}.png';fig.savefig(path,facecolor=BG);plt.close(fig)
            duration=(records[i+1]['time']-time) if i+1<len(records) else .25
            if duration<=0:raise RuntimeError('nonmonotonic cohort times')
            concat.extend([f"file '{path.resolve()}'",f'duration {duration:.17g}'])
            manifest.append(dict(cycle=item['cycle'],source=item['path'],hold_seconds=duration,**details))
        concat.append(f"file '{path.resolve()}'")
        listing=directory/'concat.txt';listing.write_text('\n'.join(concat)+'\n')
        subprocess.run([ffmpeg,'-nostdin','-v','error','-f','concat','-safe','0','-i',str(listing),'-vf','fps=24','-c:v','libx264','-threads','8','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/f'{name}_1080p.mp4')],check=True)
        (out/f'{name}_movie_manifest.json').write_text(json.dumps(dict(physical_Mref_per_second=1,camera=dict(elev=22,azim=-58,roll=0),frames=manifest),indent=2)+'\n')

def ffi(time,psi,f0,taper=True):
    """FFI for rPsi4=d²[r(h+ - i hx)]/dt², regularized below |f|=f0."""
    delta=float(time[1]-time[0]);n=len(time)
    if not np.allclose(np.diff(time),delta,rtol=1e-8,atol=1e-12):raise ValueError('FFI needs uniform samples')
    window=np.ones(n)
    if taper:
        edge=max(2,int(.1*n));ramp=.5*(1-np.cos(np.linspace(0,np.pi,edge)))
        window[:edge]=ramp;window[-edge:]=ramp[::-1]
    source=(psi-np.mean(psi))*window
    omega=2*np.pi*np.fft.fftfreq(n,delta);denominator=np.maximum(np.abs(omega),2*np.pi*f0)**2
    spectrum=np.fft.fft(source);h=np.fft.ifft(-spectrum/denominator)
    filtered=np.fft.ifft(spectrum*omega**2/denominator)
    derivative=np.gradient(np.gradient(h,delta),delta)
    valid=slice(max(4,int(n*.15)),min(n-4,int(n*.85)))
    rms=lambda x:float(np.sqrt(np.mean(np.abs(x)**2)))
    return h,dict(f0=f0,derivative_filtered_relative_rms=rms((derivative-filtered)[valid])/max(rms(filtered[valid]),1e-300),
                  derivative_original_relative_rms=rms((derivative-source)[valid])/max(rms(source[valid]),1e-300))

def waves(runs,out):
    records=[];checks=[]
    fig,ax=plt.subplots(figsize=(8,4))
    for rad in (36,40,44):
        chunks=[]
        for run in runs:
            real=run/'waveforms'/f'rpsi4_real_{rad:04d}.txt'
            imag=run/'waveforms'/f'rpsi4_imag_{rad:04d}.txt'
            times=run/'waveforms/state_times.csv'
            if not real.exists():continue
            a=np.loadtxt(real,ndmin=2);b=np.loadtxt(imag,ndmin=2);meta=pd.read_csv(times)
            if len(a)!=len(b) or len(a)!=len(meta) or not np.allclose(a[:,0],b[:,0],rtol=0,atol=1e-12) or not np.allclose(a[:,0],meta.legacy_time,rtol=0,atol=1e-12):
                raise RuntimeError('wave state provenance mismatch')
            chunks.append(pd.DataFrame(dict(cycle=meta.state_cycle,time=meta.state_time,re=a[:,5],im=b[:,5])))
        if not chunks:continue
        data=pd.concat(chunks).drop_duplicates('cycle',keep='last').sort_values('cycle')
        data.to_csv(out/f'wave22_R{rad}.csv',index=False)
        time=data.time.to_numpy();psi=data.re.to_numpy()+1j*data.im.to_numpy()
        ax.plot(time-rad,np.abs(psi),label=f'R={rad}')
        records.append(dict(radius=rad,samples=len(time),first=float(time[0]),last=float(time[-1]),
                            max_gap=float(np.diff(time).max()) if len(time)>1 else None,
                            peak_state_time=float(time[np.argmax(np.abs(psi))])))
        if rad!=40 or len(time)<100:continue
        splits=np.flatnonzero(np.diff(time)>.0375)+1
        fig_h,ax_h=plt.subplots(figsize=(8,4))
        for index,ids in enumerate(np.split(np.arange(len(time)),splits)):
            if len(ids)<100:continue
            uniform=np.arange(time[ids[0]],time[ids[-1]],.025)
            values=np.interp(uniform,time[ids],psi[ids].real)+1j*np.interp(uniform,time[ids],psi[ids].imag)
            for f0 in (.01,.02,.04):
                h,check=ffi(uniform,values,f0);check.update(interval=index,start=float(uniform[0]),end=float(uniform[-1]));checks.append(check)
                pd.DataFrame(dict(time=uniform,rh_plus=h.real,rh_cross=-h.imag)).to_csv(out/f'provisional_strain_interval{index}_f{f0:.2f}.csv',index=False)
                ax_h.plot(uniform-40,h.real,label=f'f0={f0:.2f}' if index==0 else None)
        ax_h.set(xlabel='Approximate coordinate retarded time t-R / Mref',ylabel='Provisional r h+ (2,2)')
        ax_h.set_title('Finite-window FFI sensitivity; no complete burst/ringdown claim');ax_h.legend();savefig(fig_h,out/'provisional_strain')
    ax.set(xlabel='Approximate coordinate retarded time t-R / Mref',ylabel='|r Psi4 (2,2)|');ax.legend();savefig(fig,out/'wave_radius_comparison')
    result=dict(convention='Psi4=d2(h_plus-i*h_cross)/dt2; saved modes are rPsi4',
                modes=records,ffi_checks=checks,taper='10% cosine at each edge; mean removed',
                gap_policy='split at dt>0.0375; interpolate only within each interval',
                trust='exclude tapered edges; t75 precedes the historical R40 burst near115; provisional strain only')
    (out/'wave_method.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def report(root,out,metrics,wave):
    ledgers=pd.read_csv(out/'component_ledgers.csv');env=pd.read_csv(out/'envelope_shells.csv')
    orbit=pd.read_csv(out/'orbit.csv');constraints=pd.read_csv(out/'exterior_constraints.csv')
    # Expose the non-spherical envelope moment in the instantaneous orbital frame.
    e=ledgers[(ledgers.component==0)&(ledgers['mask']==1)].merge(orbit[['cycle','phase']],on='cycle',how='left')
    quad=(e.mxx-e.myy+2j*e.mxy).to_numpy()*np.exp(-2j*e.phase.to_numpy())
    e['wake_quadrupole_real']=quad.real;e['wake_quadrupole_imag']=quad.imag
    e[['cycle','time','rest_mass','Jz','wake_quadrupole_real','wake_quadrupole_imag']].to_csv(out/'envelope_wake_moments.csv',index=False)
    # Newtonian spherical support proxy; coordinate radius and rest-mass profile,
    # never presented as a relativistic circular-orbit calculation.
    shell_radius=np.r_[np.arange(64)*.125+.0625,np.arange(64)+8.5,np.inf]
    initial=env[(env.cycle==env.cycle.min())&(env['mask']==0)].set_index('shell').rest_mass.reindex(range(129),fill_value=0).to_numpy()
    model=[]
    for row in orbit.itertuples():
        if not np.isfinite(row.Omega) or row.separation<=0:continue
        p=env[(env.cycle==row.cycle)&(env['mask']==1)]
        if p.empty or not np.isfinite(p.rest_mass).all():continue
        mass=p.set_index('shell').rest_mass.reindex(range(129),fill_value=0).to_numpy()
        radius0=row.separation/2;within=shell_radius<radius0
        model.append(dict(cycle=row.cycle,time=row.time,radius=radius0,
              evolving_env_omega2_proxy=float(mass[within].sum()/radius0**3),
              frozen_env_omega2_proxy=float(initial[within].sum()/radius0**3),measured_omega2=row.Omega**2))
    pd.DataFrame(model).to_csv(out/'envelope_support_proxy.csv',index=False)
    # Record actual allocations; scheduler elapsed time includes build/check/final I/O.
    jobs=amd_job_ids(root);node_hours=None
    if jobs:
        scheduler=subprocess.run(['sacct','-X','-P','-n','-j',','.join(sorted(set(jobs))),
                                  '--format=JobID,State,ElapsedRaw,AllocNodes,AllocCPUS,MaxRSS'],text=True,capture_output=True)
        (out/'scheduler_usage.txt').write_text(scheduler.stdout+scheduler.stderr)
        if scheduler.returncode==0:
            node_hours=0.
            for line in scheduler.stdout.splitlines():
                fields=line.split('|')
                if len(fields)>=4 and fields[0].isdigit():node_hours+=float(fields[2])*int(fields[3])/3600
    segment_rates=[];previous_time=0.;peak_gpu=0.;max_storage=0
    for run in sorted((root/'runs').glob('segment_*')):
        if (run/'checkpoint_manifest.json').exists():
            end=max(r['time'] for r in json.loads((run/'checkpoint_manifest.json').read_text()))
            if all((run/f).exists() for f in ('START_EPOCH','END_EPOCH','provenance.txt')):
                elapsed=float((run/'END_EPOCH').read_text())-float((run/'START_EPOCH').read_text())
                match=re.search(r'NumNodes=(\d+)',(run/'provenance.txt').read_text())
                if match and end>previous_time:
                    segment_rates.append(dict(segment=run.name,delta_time=end-previous_time,
                        node_hours=elapsed*int(match[1])/3600,node_hours_per_Mref=elapsed*int(match[1])/3600/(end-previous_time)))
            previous_time=end
        for path in run.glob('telemetry_*.jsonl'):
            for line in path.open():
                row=json.loads(line)
                peak_gpu=max([peak_gpu]+[d['used']/d['total'] for d in row['devices']])
        if (run/'storage.jsonl').exists():
            for line in (run/'storage.jsonl').open():max_storage=max(max_storage,json.loads(line)['bytes'])
    recent=[r['node_hours_per_Mref'] for r in segment_rates[-3:]]
    forecast=dict(central_node_hours_75_to_150=75*float(np.median(recent)) if recent else None,
                  sensitivity_range=[75*.5*min(recent),75*2*max(recent)] if recent else None,
                  method='last three application segments, half to twice measured rates; excludes unknown post75 AMR/physics and queue time')
    resource=dict(allocation_node_hours=node_hours,application_segments=segment_rates,
                  peak_gpu_fraction=peak_gpu,whole_user_peak_bytes=max_storage,forecast=forecast,
                  cost_limit='No independent overhead benchmark; application time includes diagnostics and native output. Scheduler rows include allocation finalization.')
    (out/'resource_summary.json').write_text(json.dumps(resource,indent=2)+'\n')
    saved_check=root/'control/SAVED_AH_CHECKS.json'
    ah_resolution=json.loads(saved_check.read_text()) if saved_check.exists() else dict(status='unavailable; inspect endpoint AH job log')
    (out/'saved_AH_resolution_checks.json').write_text(json.dumps(ah_resolution,indent=2)+'\n')
    inventory=[dict(path=str(p),bytes=p.stat().st_size) for p in sorted((root/'runs').rglob('*')) if p.is_file()]
    raw_bytes=sum(r['bytes'] for r in inventory)
    (out/'raw_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    latest_audit=sorted((root/'runs').glob('segment_*/population_audit.json'))[-1]
    audit=json.loads(latest_audit.read_text())
    accepted=pd.read_csv(out/'accepted_horizons.csv')
    quantitative=diagnose(root,out,metrics,accepted,orbit)
    last=[]
    for h in range(3):
        a=accepted[accepted.horizon==h]
        if len(a):
            r=a.iloc[-1];last.append(f'H{h}: last accepted t={r.state_time:.8g}, mass={r.mass:.8g}, coordinate spin={r.spin:.8g}')
    text=f'''# Session 011 endpoint readout

Verified state: t={metrics['endpoint']:.12g}, cycle={metrics['cycle']}.
First individual acceptance: {metrics['first_individual_times']}.
Strict common enclosure: {metrics['strict_common_time']}.
Observed turns after both individual horizons, excluding gaps: {metrics['post_individual_turns']}.
Observed valid orbit intervals: {len(metrics['valid_orbit_intervals'])}.
Median pre-common |d_dot|/|d Omega|: {quantitative['median_radial_ratio']}.
Median post-formation mass-growth / orbital-period ratios: {quantitative['mass_growth_to_orbit']}.
Accepted-search coverage during each horizon's relevant interval: {quantitative['coverage']}.
Actual AMD node-hours from available allocation accounting: {node_hours}.
Peak measured GPU fraction: {peak_gpu:.4f}; whole-user peak: {max_storage/1024**3:.3f} GiB.
Provisional additional75Mref application cost: {forecast['central_node_hours_75_to_150']} node-hours; sensitivity range {forecast['sensitivity_range']}.

{chr(10).join(last)}

This is one production trajectory at the inherited grid/RK4/CFL and particle
realization. There are no independent grid, timestep, seed, or physics controls.
Accepted measurements and rejected-search gaps remain separate in the event tables.
In particle figures and movies, {AH_RENDERING}. Surface centers, coefficients and
coordinate sizes are unmodified. This visibility convention is not ray tracing.

## Five hypotheses and available evidence

A — Accretion: horizon mass/spin histories, per-horizon exact-tag moving-surface
crossings, and full surviving/removed rest-mass ledgers are retained. Initial
enclosure, common reassignment, requalification, advection and surface sweep have
distinct labels. Horizon mass growth is not identified with particle rest-mass
flux; a causal accretion fraction is not inferred from their difference.

B — Envelope interaction: `envelope_wake_moments.csv` gives the exterior
quadrupole in the instantaneous orbital frame; `shell_transfers.csv` records
origin-J transport and push changes. These are coordinate diagnostics. Individual
and common fluxes overlap and must not be added into a second accretion torque.

C — Evolving support: `envelope_support_proxy.csv` compares evolving and frozen
envelope rest-mass support Menv(<d/2)/(d/2)^3 with measured Omega^2. It uses
coordinate shells and a spherical Newtonian approximation, omits non-spherical
relativistic forces, and does not provide an exact force decomposition.
`metric_shells.csv` retains proper volumes and metric averages for further checks.

D — Noncircular motion: `orbit.csv` provides separation, valid-interval phase,
Omega, |d_dot|/|d Omega| and the exterior proper line length. Binary interpretation
ends at strict common enclosure; no phase is unwrapped through missing intervals.
The proper distance is slice- and path-dependent.

Stored native particle energy uses high-order metric interpolation, which can
produce invalid geometry near collapse. Such energies remain unknown; they are
never replaced by zero or silently omitted from a component total.
`native_energy_unknown_shells` flags affected aggregate rows; checkpoint audits
record affected counts/rest mass, and movie manifests record affected cohort
counts. The independently labelled trilinear energies have their own metric
validity mask. Finite mass, position, momentum and exact IDs remain mandatory.

E — Numerical evidence: `exterior_constraints.csv`, compact AMR catalogs, exact
particle partition/rest-weight closure, restart checks and all finder reasons
are retained. This supports diagnostics and failure detection, not a convergence
claim or an isolation of the numerical contribution to the short orbit.

The binned timescales are in `timescales.csv`; entering rest mass/energy/origin-J
and surface-sweep contributions are in `crossing_fluxes.csv`. A ratio near or
above one in |d_dot|/|d Omega| indicates substantial radial motion during the
observed orbit. Mass-growth times shorter than an orbital period indicate that a
fixed-mass binary approximation is changing rapidly. These observations identify
relevant processes; one trajectory cannot assign unique causal percentages to A–E.

## Resources, storage and continuation

Actual scheduler rows are in `scheduler_usage.txt`; include failed checks and all
production allocations when calculating node-hours. Per-node GPU/host memory and
whole-user storage telemetry remain in each run. Current retained run data totals
{raw_bytes/1024**3:.3f} GiB at `{root/'runs'}`. The complete path/byte inventory is
`raw_inventory.json`; protected science states and the verified latest two recovery
states remain on AMD. Only explicitly authorized redundant recoveries are rotated.

The endpoint particle audit reports exact alive/removed ID partition and per-tag
rest-weight closure: {audit['exact_partition']}. No raw cleanup is requested by
this report. The product manifest supports the one-way checksum-verified transfer to Perseus;
transfer completion is recorded separately.

Saved-state angular checks at ntheta10 and14 (same lmax6) are recorded in
`saved_AH_resolution_checks.json`, including failures and non-mutation verdicts.
They test finder quadrature sensitivity on the same geometry; they do not test
evolution-grid convergence. `resource_summary.json` records per-segment rates and
the 75-to150 extrapolation; merger and AMR changes can invalidate that estimate.
Diagnostic and I/O overhead are included in application timing, not isolated by
an additional trajectory.

The t75 endpoint is a stop for scientific/resource review. Extension to150 is not
submitted. The historical R40 burst near115 lies beyond this endpoint; any strain
products here are finite-window FFI sensitivity diagnostics, not a completed burst
or validated ringdown result. Cutoffs, taper, gaps and differentiation checks are
recorded in `wave_method.json`.
'''
    (out/'REPORT.md').write_text(text)

def diagnose(root,out,metrics,accepted,orbit):
    """Binned rates and transfer evidence, preserving unknown/missing intervals."""
    def median(values):
        values=np.asarray(values,dtype=float);values=values[np.isfinite(values)]
        return float(np.median(values)) if len(values) else None
    orbital=orbit.copy();orbital['bin']=np.floor(orbital.time/.25).astype(int)
    scales=orbital.groupby('bin').mean(numeric_only=True).reset_index()
    scales['orbital_period']=2*np.pi/np.abs(scales.Omega)
    if len(scales)>2:
        rate=np.gradient(scales.separation,scales.time)
        scales['d_over_inward_ddot']=np.where(rate<0,scales.separation/-rate,np.nan)
    ratios={};coverage={}
    common=metrics['strict_common_time']
    common_cycle=int(accepted[(accepted.horizon==2)&(accepted.state_time>=common)].state_cycle.min()) if common is not None else metrics['cycle']+1
    for h in range(3):
        a=accepted[accepted.horizon==h].copy()
        if a.empty:
            coverage[str(h)]=None;ratios[str(h)]=None;continue
        begin=int(a.state_cycle.min()) if h<2 else common_cycle
        end=common_cycle-1 if h<2 else metrics['cycle']
        coverage[str(h)]=float(a.state_cycle.between(begin,end).sum()/(end-begin+1)) if end>=begin else None
        a['bin']=np.floor(a.state_time/.25).astype(int)
        b=a.groupby('bin').agg(time=('state_time','mean'),mass=('mass','mean')).reset_index()
        if len(b)>2:
            rate=np.gradient(b.mass,b.time)
            gap=np.r_[True,np.diff(b['bin'])==1]&np.r_[np.diff(b['bin'])==1,True]
            tau=np.where((rate>0)&gap,b.mass/rate,np.nan)
            b[f'tau_M_H{h}']=tau
            scales=scales.merge(b[['bin',f'tau_M_H{h}']],on='bin',how='left')
            use=np.isfinite(scales.Omega)
            ratios[str(h)]=median((scales[f'tau_M_H{h}']/scales.orbital_period)[use])
        else:ratios[str(h)]=None
    # Coordinate orbital-angular-momentum proxy from measured AH centers and P.
    left=accepted[accepted.horizon==0];right=accepted[accepted.horizon==1]
    pair=left.merge(right,on='state_cycle',suffixes=('_0','_1'))
    if len(pair):
        pair=pair[pair.state_time_0<common] if common is not None else pair
        pair['Lz_center_P']=pair.x_0*pair.Py_0-pair.y_0*pair.Px_0+pair.x_1*pair.Py_1-pair.y_1*pair.Px_1
        pair['bin']=np.floor(pair.state_time_0/.25).astype(int)
        b=pair.groupby('bin').agg(time=('state_time_0','mean'),Lz_center_P=('Lz_center_P','mean')).reset_index()
        if len(b)>2:
            rate=np.gradient(b.Lz_center_P,b.time)
            gap=np.r_[True,np.diff(b['bin'])==1]&np.r_[np.diff(b['bin'])==1,True]
            b['tau_L_center_P']=np.where((abs(rate)>1e-15)&gap,abs(b.Lz_center_P/rate),np.nan)
            scales=scales.merge(b[['bin','Lz_center_P','tau_L_center_P']],on='bin',how='left')
    scales.to_csv(out/'timescales.csv',index=False)
    accum=[]
    for path in sorted((root/'runs').glob('segment_*/*.gi_crossings.rank*.csv')):
        for data in pd.read_csv(path,chunksize=200_000):
            data['bin']=np.floor(data.time/.25).astype(int)
            regular=data.kind==1
            adv=np.where(regular,data.advected_inside-data.old_inside,0)
            sweep=np.where(regular,data.new_inside-data.advected_inside,0)
            data['advected_rest_mass']=adv*data.mass
            data['swept_rest_mass']=sweep*data.mass
            data['advected_Jz']=adv*data.old_Jz_origin
            data['swept_Jz']=sweep*data.old_Jz_origin
            data['enclosure_mass']=np.where(~regular,data.mass,0)
            data['advected_En']=adv*data.En_trilinear
            data['advected_Ecoord']=adv*data.Ecoord_trilinear
            data['invalid_metric_mass']=np.where(data.metric_valid==0,data.mass,0)
            fields=['advected_rest_mass','swept_rest_mass','advected_Jz','swept_Jz','enclosure_mass','advected_En','advected_Ecoord','invalid_metric_mass']
            accum.append(data.groupby(['bin','horizon','component','kind'])[fields].sum().reset_index())
    if accum:
        flux=pd.concat(accum).groupby(['bin','horizon','component','kind']).sum().reset_index()
        flux.loc[flux.invalid_metric_mass>0,['advected_En','advected_Ecoord']]=np.nan
        flux['time_bin_center']=(flux['bin']+.5)*.25
        flux.to_csv(out/'crossing_fluxes.csv',index=False)
    else:pd.DataFrame(columns=['bin','horizon','component','kind']).to_csv(out/'crossing_fluxes.csv',index=False)
    # Exact deletion state, independent of AH availability or movie membership.
    removed=[]
    for path in sorted((root/'runs').glob('segment_*/*.prtcl_destroy.csv')):
        for d in pd.read_csv(path,chunksize=200_000):
            d['bin']=np.floor(d.time/.25).astype(int)
            d['Jz']=d.mass*(d.x1*d.u2_cov-d.x2*d.u1_cov)
            d['invalid_metric_mass']=np.where(d.metric_valid==0,d.mass,0)
            d['count']=1
            removed.append(d.groupby(['bin','component','reason'])[['count','mass','Jz','En_trilinear','Ecoord_trilinear','invalid_metric_mass']].sum().reset_index())
    if removed:
        d=pd.concat(removed).groupby(['bin','component','reason']).sum().reset_index()
        d.loc[d.invalid_metric_mass>0,['En_trilinear','Ecoord_trilinear']]=np.nan
        d.to_csv(out/'removal_ledgers.csv',index=False)
    else:pd.DataFrame(columns=['bin','component','reason','count','mass','Jz','En_trilinear','Ecoord_trilinear','invalid_metric_mass']).to_csv(out/'removal_ledgers.csv',index=False)
    result=dict(median_radial_ratio=median(orbit.radial_over_tangential),mass_growth_to_orbit=ratios,coverage=coverage,
                rate_bin_width=.25,angular_momentum_proxy='sum (AH_center cross measured_AH_P)_z; coordinate dependent')
    (out/'quantitative_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    require_compute()
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--movies',action='store_true');parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    if not args.partial and not (args.root/'control/ENDPOINT_75.json').exists():raise RuntimeError('endpoint not verified')
    args.out.mkdir(parents=True,exist_ok=False)
    runs,status,events,metrics=reduce(args.root,args.out)
    wave=waves(runs,args.out)
    report(args.root,args.out,metrics,wave)
    render(runs,events,args.out,metrics['strict_common_time'],args.movies)
    manifest=[dict(path=str(p.relative_to(args.out)),bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(args.out.iterdir()) if p.is_file()]
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
