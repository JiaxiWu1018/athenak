#!/usr/bin/env python3
"""Restart-aware assessment plots and fixed-cohort central/context particle movies."""
import argparse,csv,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
from validate_initial_s7 import read_particles
from horizon_review_20261005 import accepted_measurements

def orbital_series(times,vec,live):
 """Unwrap/differentiate within contiguous live intervals only."""
 live=live & np.isfinite(vec).all(axis=1)
 d=np.full(len(times),np.nan);phase=d.copy();omega=d.copy();radial=d.copy()
 d[live]=np.linalg.norm(vec[live],axis=1)
 dt=np.diff(times);cadence=np.median(dt[dt>0]) if np.any(dt>0) else np.inf
 groups=[];active=[]
 for k in range(len(times)):
  if not live[k] or (active and times[k]-times[active[-1]]>1.5*cadence):
   if active:groups.append(np.asarray(active,dtype=int));active=[]
  if live[k]:active.append(k)
 if active:groups.append(np.asarray(active,dtype=int))
 for q in groups:
  phase[q]=np.unwrap(np.arctan2(vec[q,1],vec[q,0]))
  if len(q)>=3:
   omega[q]=np.gradient(phase[q],times[q]);radial[q]=np.gradient(d[q],times[q])
 den=d*np.abs(omega)
 ratio=np.divide(np.abs(radial),den,out=np.full_like(d,np.nan),where=den>1e-8)
 return d,phase,omega,radial,ratio,groups

def series(runs,pattern):
 data=[]
 for run in runs:
  for p in run.glob(pattern):
   rows=np.loadtxt(p,comments='#',ndmin=2)
   if len(rows):data.append(rows)
 if not data:return np.empty((0,0))
 out=np.concatenate(data);out=out[np.argsort(out[:,1],kind='stable')]
 # Last segment observation wins at a checkpoint boundary. Missing live diagnostics
 # stay NaN; values are never replaced by physical zeros.
 _,idx=np.unique(out[:,1][::-1],return_index=True)
 return out[len(out)-1-idx][np.argsort(out[len(out)-1-idx,1])]

def accepted(runs,index):
 rows=[]
 for run in runs:
  for p in run.glob('*.horizon_consumer_'+str(index)+'.csv'):
   with p.open() as f:
    for r in csv.DictReader(f):
     if all(r.get(k)=='1' for k in ('published_this_candidate','association_ok','quality_geometry_ok','quality_persist_ok')):
      rows.append({k:float(v) for k,v in r.items()})
 return sorted(rows,key=lambda r:r['time'])

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--milestone',required=True);a=p.parse_args();root=a.root
 sys.path.insert(0,str(root/'python_deps'));import imageio_ffmpeg
 ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
 out=root/'analysis'/('update_'+a.milestone+'_'+os.environ.get('SLURM_JOB_ID','manual'));out.mkdir(parents=True,exist_ok=True)
 preparation=root/'notes/REPORT_AGENT_PREPARATION.md'
 provenance='\n\n## Preparation record (historical status at submission)\n\n'+preparation.read_text() if preparation.exists() else ''
 # The accepted uninterrupted reference supplies t0 through cycle2, then gate_output
 # supplies cycle2..5 on the matched restart branch. Split/restart/stop test repeats
 # are excluded; identical boundary samples are assembled once.
 runs=[root/'runs/gate_reference',root/'runs/gate_output']+sorted((root/'runs').glob('segment_*'))
 runs=[r for r in runs if (r/'ARCHIVE_VERIFIED.json').exists()]
 if not runs:
  (out/'NO_SCIENCE_DATA.md').write_text('No verified evolution outputs are available. No results inferred.\n')
  from report9 import write_reports
  write_reports(root,out,{},a.milestone)
  return
 cutoff=float(a.milestone) if a.milestone not in ('final','0') else (float('inf') if a.milestone=='final' else .05)
 tracks=[series(runs,'*.co_'+str(i)+'.txt') for i in (0,1)]
 tracks=[t[t[:,1]<=cutoff] if len(t) else t for t in tracks]
 ah_cache=[[r for r in accepted(runs,i) if r["time"]<=cutoff] for i in (0,1,2)]
 from science9 import common_enclosure
 common_review=common_enclosure(runs,tracks)
 binary_end=min((r['time'] for r in common_review[0]),default=np.inf)
 fig,axs=plt.subplots(1,3,figsize=(13,4));metrics={"analysis_cutoff":cutoff if np.isfinite(cutoff) else None}
 if all(len(t) for t in tracks):
  times=np.intersect1d(tracks[0][:,1],tracks[1][:,1])
  left=tracks[0][np.searchsorted(tracks[0][:,1],times)];right=tracks[1][np.searchsorted(tracks[1][:,1],times)]
  vec=right[:,2:5]-left[:,2:5]
  live=np.isfinite(left[:,15]) & np.isfinite(right[:,15]) & ((left[:,13]>0)|(left[:,16]>0)) & ((right[:,13]>0)|(right[:,16]>0))
  live=live & (times<binary_end) & (np.linalg.norm(vec,axis=1)>4/256)
  d,phase,omega,ddot,ratio,groups=orbital_series(times,vec,live)
  for t,c,label in zip(tracks,['tab:blue','tab:orange'],['left','right']):
   q=t[:,2:5].copy();good=np.isfinite(t[:,15]) & ((t[:,13]>0)|(t[:,16]>0));q[~good]=np.nan
   axs[0].plot(q[:,0],q[:,1],c=c,label=label)
  axs[0].set(xlabel='x/M_ref',ylabel='y/M_ref',aspect='equal');axs[0].legend()
  axs[1].plot(times,d);axs[1].set(xlabel='t/M_ref',ylabel='coordinate separation / M_ref')
  for q in groups:axs[2].plot(times[q],(phase[q]-phase[q[0]])/(2*np.pi),color='tab:blue')
  axs[2].set(xlabel='t/M_ref',ylabel='coordinate revolutions in each live interval')
  accepted_rows=ah_cache
  formation=max((r[0]['time'] if r else np.inf) for r in accepted_rows[:2])
  post_groups=[q[times[q]>=formation] for q in groups];post_groups=[q for q in post_groups if len(q)]
  observed=sum((phase[q[-1]]-phase[q[0]])/(2*np.pi) for q in post_groups)
  metrics=dict(analysis_cutoff=cutoff if np.isfinite(cutoff) else None,final_time=float(times[-1]),tracker_invalid_rows=int((~live).sum()),both_individual_horizons_formed=bool(np.isfinite(formation)),
   both_formation_time=float(formation) if np.isfinite(formation) else None,
   post_formation_revolutions=float(observed) if len(post_groups)==1 else None,
   observed_post_formation_arc_revolutions=float(observed) if post_groups else None,post_formation_live_intervals=len(post_groups),
   common_candidates_published=len(accepted_rows[2]),merger_claim=False,
   limitations='Coordinate diagnostics. Common candidate publication alone does not establish enclosure/merger. No radiation-driven inspiral claim.')
  np.savetxt(out/'orbit.csv',np.column_stack([times,d,phase,omega,ddot,ratio]),delimiter=',',header='time,separation,unwrapped_phase_within_live_intervals,Omega,d_dot,abs_d_dot_over_d_abs_Omega')
  fig_motion,axes_motion=plt.subplots(1,3,figsize=(13,4))
  axes_motion[0].plot(times,omega);axes_motion[0].set(xlabel='t/M_ref',ylabel='coordinate Omega')
  axes_motion[1].plot(times,ddot,label='d_dot');axes_motion[1].plot(times,d*np.abs(omega),label='d |Omega|');axes_motion[1].legend();axes_motion[1].set(xlabel='t/M_ref',ylabel='coordinate relative speeds')
  axes_motion[2].plot(times,ratio);axes_motion[2].set(xlabel='t/M_ref',ylabel='|d_dot| / (d |Omega|)')
  fig_motion.tight_layout();fig_motion.savefig(out/'radial_tangential.png',dpi=170);plt.close(fig_motion)
 fig.suptitle('Session 009 assessment; M_ref=1 inherited source unit');fig.tight_layout();fig.savefig(out/'orbit.png',dpi=170);plt.close(fig)
 for i in (0,1):
  t=tracks[i]
  if len(t):
   valid=np.isfinite(t[:,15]) & ((t[:,13]>0)|(t[:,16]>0))
   np.savetxt(out/('tracker_'+str(i)+'_validity.csv'),np.c_[t[:,1],valid.astype(int)],delimiter=',',header='time,live_diagnostic_valid')
 # The summary's first column is the finder iteration, not the evolution cycle.
 # Match its same-run candidate by printed time AND surface geometry, uniquely,
 # and require strict consumer publication, persistence and association.
 fig,axes=plt.subplots(1,2,figsize=(10,4));horizon_counts={}
 for index in (0,1,2):
  rows=[]
  for run in runs:
   consumers=[]
   for consumer in run.glob('*.horizon_consumer_'+str(index)+'.csv'):
    with consumer.open() as stream:consumers.extend(csv.DictReader(stream))
   for file in run.glob('*.horizon_summary_'+str(index)+'.txt'):
    if not any(x.strip() and not x.startswith('#') for x in file.read_text().splitlines()):continue
    data=np.loadtxt(file,ndmin=2)
    matched,counts=accepted_measurements(consumers,sorted(data.tolist(),key=lambda row:row[1]))
    if counts['unmatched'] or counts['ambiguous']:raise RuntimeError('Accepted horizon lacks a unique summary match: '+str(counts))
    for candidate,row in matched:
     if float(candidate['time'])>cutoff:continue
     row=np.asarray(row);row[0]=int(candidate['cycle']);row[1]=float(candidate['time']);rows.append(row)
  horizon_counts[str(index)]=len(rows)
  if rows:
   data=np.asarray(rows);np.savetxt(out/('accepted_horizon_'+str(index)+'.csv'),data,delimiter=',',header='cycle,time,M_BH,Sx,Sy,Sz,S,area,hrms,hmean,meanradius,minradius,M_irr,chi_BH,Px,Py,Pz,P,center_x,center_y,center_z')
   axes[0].plot(data[:,1],data[:,2],'.',ms=1,label=str(index));axes[1].plot(data[:,1],data[:,13],'.',ms=1,label=str(index))
 axes[0].set(xlabel='t/M_ref',ylabel='accepted M_BH/M_ref');axes[1].set(xlabel='t/M_ref',ylabel='accepted chi_BH (coordinate spin prescription)')
 if any(horizon_counts.values()):axes[0].legend();axes[1].legend()
 fig.tight_layout();fig.savefig(out/'accepted_horizons.png',dpi=170);plt.close(fig);metrics['accepted_horizon_rows']=horizon_counts
 fig,ax=plt.subplots(figsize=(8,4))
 for run in runs:
  for file in run.glob('*.z4c.user.hst'):
   if not any(x.strip() and not x.startswith('#') for x in file.read_text().splitlines()):continue
   data=np.loadtxt(file,ndmin=2);valid=(data[:,10]>0)&np.isfinite(data).all(axis=1)&(data[:,0]>0)&(data[:,0]<=cutoff)
   ax.semilogy(data[valid,0],np.sqrt(data[valid,3]/data[valid,10]),color='tab:blue',label='H RMS' if run==runs[0] else None)
   ax.semilogy(data[valid,0],np.sqrt(data[valid,4]/data[valid,10]),color='tab:orange',label='M RMS' if run==runs[0] else None)
 ax.set(xlabel='t/M_ref',ylabel='proper-volume RMS; chi >= 0.0625, not AH exterior');ax.legend();fig.tight_layout();fig.savefig(out/'constraints.png',dpi=170);plt.close(fig)
 # Raw multipoles are primary evidence. The assessment precedes central signals at
 # extraction spheres; FFI strain is deliberately deferred until causal coverage exists.
 fig,axes_wave=plt.subplots(2,1,figsize=(8,7),sharex=True)
 for radius in (40,):
  chunks=[]
  for run in runs:
   rp=run/'waveforms'/('rpsi4_real_'+str(radius).zfill(4)+'.txt');ip=run/'waveforms'/('rpsi4_imag_'+str(radius).zfill(4)+'.txt')
   if not rp.exists() or not ip.exists():continue
   r=np.loadtxt(rp,ndmin=2);im=np.loadtxt(ip,ndmin=2)
   if r.shape!=im.shape or not np.array_equal(r[:,0],im[:,0]):raise RuntimeError('waveform real/imag mismatch')
   q=r[:,0]<=cutoff
   if np.any(q):chunks.append(np.c_[r[q,0],r[q,1],im[q,1],r[q,5],im[q,5]])
  if chunks:
   z=np.concatenate(chunks);z=z[np.argsort(z[:,0],kind='stable')]
   for t in np.unique(z[:,0]):
    overlap=z[z[:,0]==t]
    if len(overlap)>1 and not np.allclose(overlap[:,1:],overlap[0,1:],rtol=1e-6,atol=1e-12):raise RuntimeError('conflicting waveform overlap')
   _,ix=np.unique(z[:,0],return_index=True);z=z[ix]
   for ax,m,re_col,im_col in zip(axes_wave,(-2,2),(1,3),(2,4)):
    np.savetxt(out/('rpsi4_2'+str(m)+'_r'+str(radius)+'.csv'),np.c_[z[:,0],z[:,0]-radius,z[:,re_col],z[:,im_col]],delimiter=',',header='coordinate_time,approx_retarded_time_t_minus_r,rPsi4_real,rPsi4_imag')
    ax.plot(z[:,0],np.hypot(z[:,re_col],z[:,im_col]),label='r='+str(radius));ax.set(ylabel='|r Psi4(2,'+str(m)+')|');ax.legend()
 axes_wave[-1].set(xlabel='t/M_ref');fig.suptitle('Raw complex extraction at coordinate r=40; M_ref=1');fig.tight_layout();fig.savefig(out/'raw_waveform.png',dpi=170);plt.close(fig)
 snapshots={}
 initial=next(iter(sorted((root/'runs/gate_reference').rglob('*.part.vtk'))),None)
 if initial:snapshots[0.]=initial
 for run in runs:
  for path in sorted(run.rglob('*.part.vtk')):
   with path.open('rb') as f:head=f.read(256)
   import re
   m=re.search(rb'time=\s*([-+0-9.eE]+)',head)
   if m and float(m.group(1))<=cutoff:snapshots[float(m.group(1))]=path
 colors=['#888888','tab:blue','tab:orange'];cohort=None;particle_ledger=[];matter_shell=[]
 for number,(tm,path) in enumerate(sorted(snapshots.items())):
  pt=read_particles(path)
  rad=np.linalg.norm(pt['position'],axis=1);shell=(rad>=38)&(rad<=42)
  matter_shell.append([tm,int(shell.sum()),float(pt['mass'][shell].sum()),float(pt['mass'].sum())])
  tag=np.rint(pt['tag_float']).astype(np.int64)
  if not np.isfinite(pt['position']).all():raise RuntimeError('nonfinite movie particle coordinates')
  for component,(lo,hi) in enumerate([(0,3000000),(3000000,4000000),(4000000,5000000)]):
   q=(tag>=lo)&(tag<hi);xyz_full=pt['position'][q].astype(float);mom=pt['momentum'][q].astype(float);mu=pt['mass'][q].astype(float)
   if len(mu):
    center=np.sum(mu[:,None]*xyz_full,axis=0)/np.sum(mu);P=np.sum(mu[:,None]*mom,axis=0);J=np.sum(mu[:,None]*np.cross(xyz_full,mom),axis=0)
   else:center=np.full(3,np.nan);P=np.zeros(3);J=np.zeros(3)
   particle_ledger.append([tm,component,len(mu),mu.sum(),*center,*P,*J])
  if cohort is None:
   cohort=set()
   for lo,hi,n in [(0,3000000,40000),(3000000,4000000,25000),(4000000,5000000,25000)]:
    ids=np.sort(tag[(tag>=lo)&(tag<hi)]);cohort.update(ids[np.linspace(0,len(ids)-1,min(n,len(ids)),dtype=int)].tolist())
  selected=np.isin(tag,np.asarray(sorted(cohort)));xyz=pt['position'][selected];tg=tag[selected]
  for label,lim in [('central',6),('context',35)]:
   frames=root/'analysis/frame_cache'/(label+'_frames');frames.mkdir(parents=True,exist_ok=True)
   if (frames/f'{number:05d}.png').exists():continue
   fig,ax=plt.subplots(figsize=(7,7))
   for i,(lo,hi) in enumerate([(0,3000000),(3000000,4000000),(4000000,5000000)]):
    q=xyz[(tg>=lo)&(tg<hi)];ax.scatter(q[:,0],q[:,1],s=.4,alpha=.45,c=colors[i],rasterized=True)
   for i,c in enumerate(colors[1:]):
    t=tracks[i];q=t[t[:,1]<=tm] if len(t) else t
    if len(q):
     q=q.copy();valid=np.isfinite(q[:,15])&((q[:,13]>0)|(q[:,16]>0));q[~valid,2:5]=np.nan
     ax.plot(q[:,2],q[:,3],c=c,lw=1)
    ah=[r for r in ah_cache[i] if 0<=tm-r['time']<=.1]
    if ah:
     r=ah[-1];ax.add_patch(Circle((r['center_x'],r['center_y']),r['rmin'],fill=False,color=c,ls='--'))
   common=[r for r in ah_cache[2] if 0<=tm-r['time']<=.1]
   if common:
    r=common[-1];ax.add_patch(Circle((r['center_x'],r['center_y']),r['rmin'],fill=False,color='green',ls='--'))
   ax.text(.02,.97,'Gray envelope; blue left; orange right; green common AH',transform=ax.transAxes,va='top',fontsize=7)
   ax.set(xlim=(-lim,lim),ylim=(-lim,lim),aspect='equal',xlabel='x/M_ref',ylabel='y/M_ref',title=f'Session 009 {label}; t/M_ref={tm:.3f}')
   ax.text(.02,.02,'Fixed tagged rendering cohort; dashed AH circles show accepted rmin only',transform=ax.transAxes,fontsize=7)
   fig.tight_layout();fig.savefig(frames/f'{number:05d}.png',dpi=120);plt.close(fig)
  # Fixed Cartesian bins of full-particle rest weights avoid native-AMR seams.
  # This is coordinate surface density in a slab, not proper energy density.
  xyz_all=pt['position'];slab=np.abs(xyz_all[:,2])<.7
  edges=np.linspace(-6,6,257);dx=edges[1]-edges[0]
  density,_,_=np.histogram2d(xyz_all[slab,0],xyz_all[slab,1],bins=(edges,edges),weights=pt['mass'][slab]);density/=dx*dx
  frames=root/'analysis/frame_cache/density_frames';frames.mkdir(parents=True,exist_ok=True)
  if (frames/f'{number:05d}.png').exists():continue
  fig,ax=plt.subplots(figsize=(7,7))
  im=ax.imshow(np.ma.masked_less_equal(density.T,0),origin='lower',extent=(-6,6,-6,6),norm=LogNorm(1e-6,1),cmap='magma',interpolation='nearest')
  fig.colorbar(im,ax=ax,label='coordinate rest-mass surface density; |z| < 0.7 M_ref')
  ax.set(xlabel='x/M_ref',ylabel='y/M_ref',title=f'Full-particle fixed-grid deposition; t/M_ref={tm:.3f}')
  ax.text(.02,.02,'256 x 256 fixed bins; fixed color limits; no proper-density claim',transform=ax.transAxes,fontsize=7,color='white')
  fig.tight_layout();fig.savefig(frames/f'{number:05d}.png',dpi=120);plt.close(fig)
 for label in ('central','context','density'):
  frames=root/'analysis/frame_cache'/(label+'_frames')
  if frames.exists():
   subprocess.run([ffmpeg,'-y','-loglevel','error','-framerate','6','-i',str(frames/'%05d.png'),'-frames:v',str(len(snapshots)),'-c:v','libx264','-pix_fmt','yuv420p',str(out/(label+'.mp4'))],check=True)
   subprocess.run([ffmpeg,'-v','error','-i',str(out/(label+'.mp4')),'-f','null','-'],check=True)
 if particle_ledger:
  ledger=np.asarray(particle_ledger);np.savetxt(out/'particle_components.csv',ledger,delimiter=',',header='time,component,count,rest_mass,center_x,center_y,center_z,Px_cov,Py_cov,Pz_cov,Jx_cov,Jy_cov,Jz_cov')
  fig,axes=plt.subplots(1,2,figsize=(10,4))
  for i in (0,1,2):
   q=ledger[ledger[:,1]==i];axes[0].plot(q[:,0],q[:,2],label=str(i));axes[1].plot(q[:,0],q[:,12],label=str(i))
  axes[0].set(xlabel='t/M_ref',ylabel='alive component particle count');axes[1].set(xlabel='t/M_ref',ylabel='alive matter J_z from covariant u_i');axes[0].legend();axes[1].legend();fig.tight_layout();fig.savefig(out/'particle_components.png',dpi=170);plt.close(fig)
  env=ledger[ledger[:,1]==0];good=np.isfinite(env[:,4:7]).all(axis=1);env=env[good]
  fig,ax=plt.subplots(figsize=(6,6))
  if len(env)>=2:
   for t,c,label in zip(tracks,colors[1:],('left','right')):
    if not len(t):continue
    live=np.isfinite(t[:,15]) & ((t[:,13]>0)|(t[:,16]>0)) & (t[:,1]>=env[0,0]) & (t[:,1]<=env[-1,0])
    relative=t[:,2:5]-np.column_stack([np.interp(t[:,1],env[:,0],env[:,k]) for k in (4,5,6)])
    relative[~live]=np.nan;ax.plot(relative[:,0],relative[:,1],color=c,label=label)
    np.savetxt(out/('envelope_relative_'+label+'.csv'),np.c_[t[:,1],relative,live.astype(int)],delimiter=',',header='time,x_relative,y_relative,z_relative,live_tracker_valid')
   ax.legend()
  ax.set(xlabel='(x - sampled envelope center)/M_ref',ylabel='(y - sampled envelope center)/M_ref',aspect='equal',title='Coordinate trajectories relative to rest-weighted envelope center')
  fig.tight_layout();fig.savefig(out/'envelope_relative.png',dpi=170);plt.close(fig)
 (out/'summary.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
 np.savetxt(out/'matter_near_r40.csv',np.asarray(matter_shell).reshape(-1,4),delimiter=',',header='time,particle_count_in_r38_to42,sampled_rest_mass_in_shell,total_alive_rest_mass')
 from removal9 import assess_removals
 assess_removals(runs,out,cutoff)
 from science9 import extended_analysis
 extended_analysis(root,runs,out,metrics,tracks,common_review)
 from report9 import write_reports
 write_reports(root,out,metrics,a.milestone)

if __name__=='__main__':main()
