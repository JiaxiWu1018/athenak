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
    if len(np.unique(records['tag']))!=n or not np.isfinite(records['q']).all():raise ValueError('invalid cohort records')
    return cycle,time,records

def surface(ax,row):
    theta,phi=np.meshgrid(np.linspace(0,np.pi,37),np.linspace(0,2*np.pi,73),indexing='ij')
    r=radius(row,theta,phi)
    x=row['x']+r*np.sin(theta)*np.cos(phi);y=row['y']+r*np.sin(theta)*np.sin(phi);z=row['z']+r*np.cos(theta)
    ax.plot_surface(x,y,z,color='black',linewidth=0,edgecolor='none',shade=False,alpha=1)

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
    for c in range(3):
        use=visible&(group==c)
        ax.scatter(*xyz[use].T,s=1.45 if c==0 else 2.1,alpha=.34 if c==0 else .82,
                   color=COLORS[c],edgecolors='none',linewidths=0,depthshade=False,rasterized=True)
    for row in horizons:surface(ax,row)
    ax.set(xlim=(-limit,limit),ylim=(-limit,limit),zlim=(-limit,limit));ax.set_box_aspect((1,1,1));ax.set_axis_off()
    if caption:ax.text2D(.025,.95,f'Saved state  t = {time:.4f} Mref',transform=ax.transAxes,color='white',fontsize=12)
    return dict(saved_time=time,alive_cohort=[int((group==c).sum()) for c in range(3)],visible=[int((visible&(group==c)).sum()) for c in range(3)],accepted_horizons=[int(r['horizon']) for r in horizons])

def intervals(cycles,valid):
    selected=np.flatnonzero(valid)
    if not len(selected):return []
    # Compact diagnostic samples normally span >1 integration cycle. Break on
    # invalid observations and segment boundaries, supplied by the caller.
    return np.split(selected,np.flatnonzero(np.diff(selected)>1)+1)

def savefig(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=200,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)

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
    phase=np.full(len(status),np.nan);omega=phase.copy();turns=0.;post_turns=0.;gaps=[]
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
            if len(ids_post)>1:post_turns+=(p[ids_post[-1]]-p[ids_post[0]])/(2*np.pi)
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
    sums=particles.groupby(['cycle','time','component','mask'],dropna=False).sum(numeric_only=True,min_count=1).reset_index()
    sums.to_csv(out/'component_ledgers.csv',index=False)
    norms=grid[grid['mask']==1].groupby(['cycle','time']).sum(numeric_only=True,min_count=1).reset_index()
    norms['H_RMS']=np.sqrt(norms.H2_volume/norms.proper_volume);norms['M_RMS']=np.sqrt(norms.M2_volume/norms.proper_volume)
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
                 strict_common_time=common,observed_turns=turns,post_individual_turns=post_turns,
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
    if not shutil.which('ffmpeg'):raise RuntimeError('ffmpeg unavailable')
    for name,limit in (('wide',24),('close',4)):
        directory=out/f'frames_{name}';directory.mkdir()
        concat=[];manifest=[]
        for i,item in enumerate(records):
            _,time,data=trk(Path(item['path']))
            fig=plt.figure(figsize=(16,9),dpi=120,facecolor=BG)
            ax=fig.add_axes([0,0,1,1],projection='3d',facecolor=BG)
            details=draw_state(ax,data,item['horizons'],limit,time)
            ax.text2D(.025,.035,'Fixed particle IDs; accepted AH geometry at this saved cycle',transform=ax.transAxes,color='white',fontsize=10)
            path=directory/f'frame_{i:05d}.png';fig.savefig(path,facecolor=BG);plt.close(fig)
            duration=(records[i+1]['time']-time) if i+1<len(records) else .25
            if duration<=0:raise RuntimeError('nonmonotonic cohort times')
            concat.extend([f"file '{path.resolve()}'",f'duration {duration:.17g}'])
            manifest.append(dict(cycle=item['cycle'],source=item['path'],hold_seconds=duration,**details))
        concat.append(f"file '{path.resolve()}'")
        listing=directory/'concat.txt';listing.write_text('\n'.join(concat)+'\n')
        subprocess.run(['ffmpeg','-nostdin','-v','error','-f','concat','-safe','0','-i',str(listing),'-vf','fps=24','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/f'{name}_1080p.mp4')],check=True)
        (out/f'{name}_movie_manifest.json').write_text(json.dumps(dict(physical_Mref_per_second=1,camera=dict(elev=22,azim=-58,roll=0),frames=manifest),indent=2)+'\n')

def main():
    require_compute()
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--movies',action='store_true');parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    if not args.partial and not (args.root/'control/ENDPOINT_75.json').exists():raise RuntimeError('endpoint not verified')
    args.out.mkdir(parents=True,exist_ok=False)
    runs,status,events,metrics=reduce(args.root,args.out)
    render(runs,events,args.out,metrics['strict_common_time'],args.movies)
    manifest=[dict(path=str(p.relative_to(args.out)),bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(args.out.iterdir()) if p.is_file()]
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
