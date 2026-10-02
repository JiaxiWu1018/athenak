#!/usr/bin/env python3
"""Restart-aware assessment plots and fixed-cohort central/context particle movies."""
import argparse,csv,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from validate_initial_s7 import read_particles

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
     if r.get('published_this_candidate')=='1' and r.get('association_ok')=='1':
      rows.append({k:float(v) for k,v in r.items()})
 return sorted(rows,key=lambda r:r['time'])

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
 sys.path.insert(0,str(root/'python_deps'));import imageio_ffmpeg
 ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
 out=root/'analysis'/('assessment_'+os.environ.get('SLURM_JOB_ID','manual'));out.mkdir(parents=True,exist_ok=True)
 # Only gate_output supplies the initial checkpoint interval. Other gate repetitions
 # are validation evidence, never multiple physical evolution segments.
 runs=[root/'runs/gate_output']+sorted((root/'runs').glob('segment_*'))
 runs=[r for r in runs if (r/'ARCHIVE_VERIFIED.json').exists()]
 if not runs:
  (out/'NO_SCIENCE_DATA.md').write_text('No verified evolution outputs are available. No results inferred.\n')
  state=json.loads((root/'evidence/amd_state.json').read_text())
  (root/'REPORT_Jeans8.md').write_text('# Session 008 assessment halted\n\nNo verified evolution outputs available; no scientific outcome or plots inferred. Workflow status: '+state['status']+'.\n')
  (root/'REPORT_AGENT.md').write_text('# Session 008 failure record\n\n```json\n'+json.dumps(state,indent=2)+'\n```\n')
  return
 tracks=[series(runs,'*.co_'+str(i)+'.txt') for i in (0,1)]
 fig,axs=plt.subplots(1,3,figsize=(13,4));metrics={}
 if all(len(t) for t in tracks):
  times=np.intersect1d(tracks[0][:,1],tracks[1][:,1])
  left=tracks[0][np.searchsorted(tracks[0][:,1],times)];right=tracks[1][np.searchsorted(tracks[1][:,1],times)]
  vec=right[:,2:5]-left[:,2:5];d=np.linalg.norm(vec,axis=1);phase=np.unwrap(np.arctan2(vec[:,1],vec[:,0]))
  live=np.isfinite(left[:,15]) & np.isfinite(right[:,15]) & ((left[:,13]>0)|(left[:,16]>0)) & ((right[:,13]>0)|(right[:,16]>0))
  d[~live]=np.nan;phase[~live]=np.nan
  for t,c,label in zip(tracks,['tab:blue','tab:orange'],['left','right']):axs[0].plot(t[:,2],t[:,3],c=c,label=label)
  axs[0].set(xlabel='x/M_ref',ylabel='y/M_ref',aspect='equal');axs[0].legend()
  axs[1].plot(times,d);axs[1].set(xlabel='t/M_ref',ylabel='coordinate separation / M_ref')
  phase0=phase[np.flatnonzero(live)[0]] if live.any() else 0.
  axs[2].plot(times,(phase-phase0)/(2*np.pi));axs[2].set(xlabel='t/M_ref',ylabel='accumulated coordinate revolutions')
  if len(times)>2:
   omega=np.gradient(phase,times);ddot=np.gradient(d,times);den=d*np.abs(omega)
   ratio=np.divide(np.abs(ddot),den,out=np.full_like(d,np.nan),where=den>1e-8)
  else:omega=ddot=ratio=np.full_like(d,np.nan)
  accepted_rows=[accepted(runs,i) for i in (0,1,2)]
  formation=max((r[0]['time'] if r else np.inf) for r in accepted_rows[:2])
  post=times>=formation
  metrics=dict(final_time=float(times[-1]),tracker_invalid_rows=int((~live).sum()),both_individual_horizons_formed=bool(np.isfinite(formation)),
   both_formation_time=float(formation) if np.isfinite(formation) else None,
   post_formation_revolutions=float((phase[post & live][-1]-phase[post & live][0])/(2*np.pi)) if (post & live).any() else None,
   common_candidates_published=len(accepted_rows[2]),merger_claim=False,
   limitations='Coordinate diagnostics. Common candidate publication alone does not establish enclosure/merger. No radiation-driven inspiral claim.')
  np.savetxt(out/'orbit.csv',np.column_stack([times,d,phase,omega,ddot,ratio]),delimiter=',',header='time,separation,unwrapped_phase,Omega,d_dot,abs_d_dot_over_d_abs_Omega')
 fig.suptitle('Session 008 assessment; M_ref=1 inherited source unit');fig.tight_layout();fig.savefig(out/'orbit.png',dpi=170);plt.close(fig)
 for i in (0,1):
  t=tracks[i]
  if len(t):
   valid=np.isfinite(t[:,15]) & ((t[:,13]>0)|(t[:,16]>0))
   np.savetxt(out/('tracker_'+str(i)+'_validity.csv'),np.c_[t[:,1],valid.astype(int)],delimiter=',',header='time,live_diagnostic_valid')
 # A summary row is a measurement only when its same-run/same-cycle candidate was
 # actually published by the strict consumer with association acceptance.
 fig,axes=plt.subplots(1,2,figsize=(10,4));horizon_counts={}
 for index in (0,1,2):
  rows=[]
  for run in runs:
   good=accepted([run],index)
   for file in run.glob('*.horizon_summary_'+str(index)+'.txt'):
    if not any(x.strip() and not x.startswith('#') for x in file.read_text().splitlines()):continue
    data=np.loadtxt(file,ndmin=2)
    for row in data:
     if np.isfinite(row).all() and row[2]>0 and any(int(a['cycle'])==int(row[0]) and abs(a['time']-row[1])<=5e-5 for a in good):rows.append(row)
  horizon_counts[str(index)]=len(rows)
  if rows:
   data=np.asarray(rows);np.savetxt(out/('accepted_horizon_'+str(index)+'.csv'),data,delimiter=',',header='cycle,time,M_BH,Sx,Sy,Sz,S,area,hrms,hmean,meanradius,minradius,M_irr,chi_BH,Px,Py,Pz,P,center_x,center_y,center_z')
   axes[0].plot(data[:,1],data[:,2],label=str(index));axes[1].plot(data[:,1],data[:,13],label=str(index))
 axes[0].set(xlabel='t/M_ref',ylabel='accepted M_BH/M_ref');axes[1].set(xlabel='t/M_ref',ylabel='accepted chi_BH (coordinate spin prescription)')
 if any(horizon_counts.values()):axes[0].legend();axes[1].legend()
 fig.tight_layout();fig.savefig(out/'accepted_horizons.png',dpi=170);plt.close(fig);metrics['accepted_horizon_rows']=horizon_counts
 fig,ax=plt.subplots(figsize=(8,4))
 for run in runs:
  for file in run.glob('*.z4c.user.hst'):
   if not any(x.strip() and not x.startswith('#') for x in file.read_text().splitlines()):continue
   data=np.loadtxt(file,ndmin=2);valid=(data[:,10]>0)&np.isfinite(data).all(axis=1)&(data[:,0]>0)
   ax.semilogy(data[valid,0],np.sqrt(data[valid,3]/data[valid,10]),color='tab:blue',label='H RMS' if run==runs[0] else None)
   ax.semilogy(data[valid,0],np.sqrt(data[valid,4]/data[valid,10]),color='tab:orange',label='M RMS' if run==runs[0] else None)
 ax.set(xlabel='t/M_ref',ylabel='proper-volume RMS; chi >= 0.0625, not AH exterior');ax.legend();fig.tight_layout();fig.savefig(out/'constraints.png',dpi=170);plt.close(fig)
 # Raw multipoles are primary evidence. The assessment precedes central signals at
 # extraction spheres; FFI strain is deliberately deferred until causal coverage exists.
 fig,ax=plt.subplots(figsize=(8,4))
 for radius in (40,50,60,70):
  chunks=[]
  for run in runs:
   rp=run/'waveforms'/('rpsi4_real_'+str(radius).zfill(4)+'.txt');ip=run/'waveforms'/('rpsi4_imag_'+str(radius).zfill(4)+'.txt')
   if not rp.exists() or not ip.exists():continue
   r=np.loadtxt(rp,ndmin=2);im=np.loadtxt(ip,ndmin=2)
   if r.shape!=im.shape or not np.array_equal(r[:,0],im[:,0]):raise RuntimeError('waveform real/imag mismatch')
   chunks.append(np.c_[r[:,0],r[:,5],im[:,5]])
  if chunks:
   z=np.concatenate(chunks);z=z[np.argsort(z[:,0],kind='stable')]
   for t in np.unique(z[:,0]):
    overlap=z[z[:,0]==t]
    if len(overlap)>1 and not np.allclose(overlap[:,1:],overlap[0,1:],rtol=1e-6,atol=1e-12):raise RuntimeError('conflicting waveform overlap')
   _,ix=np.unique(z[:,0],return_index=True);z=z[ix]
   np.savetxt(out/('rpsi4_22_r'+str(radius)+'.csv'),z,delimiter=',',header='coordinate_time,rPsi4_real,rPsi4_imag')
   ax.plot(z[:,0],np.hypot(z[:,1],z[:,2]),label='r='+str(radius))
 ax.set(xlabel='t/M_ref',ylabel='|r Psi4(2,+2)|',title='Early raw extraction: initialization/outer-field response');ax.legend();fig.tight_layout();fig.savefig(out/'raw_waveform.png',dpi=170);plt.close(fig)
 snapshots={}
 initial=next(iter(sorted((root/'runs/gate_reference').rglob('*.part.vtk'))),None)
 if initial:snapshots[0.]=initial
 for run in runs:
  for path in sorted(run.rglob('*.part.vtk')):
   with path.open('rb') as f:head=f.read(256)
   import re
   m=re.search(rb'time=\s*([-+0-9.eE]+)',head)
   if m:snapshots[float(m.group(1))]=path
 colors=['#888888','tab:blue','tab:orange'];cohort=None;particle_ledger=[]
 for number,(tm,path) in enumerate(sorted(snapshots.items())):
  pt=read_particles(path);tag=np.rint(pt['tag_float']).astype(np.int64)
  if not np.isfinite(pt['position']).all():raise RuntimeError('nonfinite movie particle coordinates')
  for component,(lo,hi) in enumerate([(0,3000000),(3000000,4000000),(4000000,5000000)]):
   q=(tag>=lo)&(tag<hi);xyz_full=pt['position'][q].astype(float);mom=pt['momentum'][q].astype(float);mu=pt['mass'][q].astype(float)
   if len(mu):
    center=np.sum(mu[:,None]*xyz_full,axis=0)/np.sum(mu);P=np.sum(mu[:,None]*mom,axis=0);J=np.sum(mu[:,None]*np.cross(xyz_full,mom),axis=0)
    particle_ledger.append([tm,component,len(mu),mu.sum(),*center,*P,*J])
  if cohort is None:
   cohort=set()
   for lo,hi,n in [(0,3000000,40000),(3000000,4000000,25000),(4000000,5000000,25000)]:
    ids=np.sort(tag[(tag>=lo)&(tag<hi)]);cohort.update(ids[np.linspace(0,len(ids)-1,min(n,len(ids)),dtype=int)].tolist())
  selected=np.isin(tag,np.asarray(sorted(cohort)));xyz=pt['position'][selected];tg=tag[selected]
  for label,lim in [('central',6),('context',35)]:
   frames=out/(label+'_frames');frames.mkdir(exist_ok=True);fig,ax=plt.subplots(figsize=(7,7))
   for i,(lo,hi) in enumerate([(0,3000000),(3000000,4000000),(4000000,5000000)]):
    q=xyz[(tg>=lo)&(tg<hi)];ax.scatter(q[:,0],q[:,1],s=.4,alpha=.45,c=colors[i],rasterized=True)
   for i,c in enumerate(colors[1:]):
    t=tracks[i];q=t[t[:,1]<=tm] if len(t) else t
    if len(q):
     q=q.copy();valid=np.isfinite(q[:,15])&((q[:,13]>0)|(q[:,16]>0));q[~valid,2:5]=np.nan
     ax.plot(q[:,2],q[:,3],c=c,lw=1)
    ah=[r for r in accepted(runs,i) if 0<=tm-r['time']<=.1]
    if ah:
     r=ah[-1];ax.add_patch(Circle((r['center_x'],r['center_y']),r['rmin'],fill=False,color=c,ls='--'))
   ax.set(xlim=(-lim,lim),ylim=(-lim,lim),aspect='equal',xlabel='x/M_ref',ylabel='y/M_ref',title=f'Session 008 {label}; t/M_ref={tm:.3f}')
   ax.text(.02,.02,'Fixed tagged rendering cohort; dashed AH circles show accepted rmin only',transform=ax.transAxes,fontsize=7)
   fig.tight_layout();fig.savefig(frames/f'{number:05d}.png',dpi=120);plt.close(fig)
 for label in ('central','context'):
  frames=out/(label+'_frames')
  if frames.exists():
   subprocess.run([ffmpeg,'-y','-loglevel','error','-framerate','6','-i',str(frames/'%05d.png'),'-c:v','libx264','-pix_fmt','yuv420p',str(out/(label+'.mp4'))],check=True)
   subprocess.run([ffmpeg,'-v','error','-i',str(out/(label+'.mp4')),'-f','null','-'],check=True)
 if particle_ledger:
  ledger=np.asarray(particle_ledger);np.savetxt(out/'particle_components.csv',ledger,delimiter=',',header='time,component,count,rest_mass,center_x,center_y,center_z,Px_cov,Py_cov,Pz_cov,Jx_cov,Jy_cov,Jz_cov')
  fig,axes=plt.subplots(1,2,figsize=(10,4))
  for i in (0,1,2):
   q=ledger[ledger[:,1]==i];axes[0].plot(q[:,0],q[:,2],label=str(i));axes[1].plot(q[:,0],q[:,12],label=str(i))
  axes[0].set(xlabel='t/M_ref',ylabel='alive component particle count');axes[1].set(xlabel='t/M_ref',ylabel='alive matter J_z from covariant u_i');axes[0].legend();axes[1].legend();fig.tight_layout();fig.savefig(out/'particle_components.png',dpi=170);plt.close(fig)
 (out/'summary.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
 (out/'CAPTIONS.md').write_text('Orbit: coordinate separation, trajectories and phase; invalid live tracker rows are separately flagged. Raw waveform: t<=12 cannot contain central collapse waves at r>=40. Movies: fixed 90,000-tag cohort, xy projection, no replacement after removal; simulated initial count remains five million. Dashed circles are accepted coordinate rmin illustrations, not full horizon surfaces. No strain or merger waveform claim.\n')
 state=json.loads((root/'evidence/amd_state.json').read_text())
 report='''# Jeans-in-cluster Session 008 assessment\n\nFresh companion-supported initial local boosts: left -y, right +y, magnitude 0.133215. This approximate prescription is not exact GR binary equilibrium. Five million initial particles, source masses 0.76/0.12/0.12; M_ref=1 is the inherited source unit. Approximate K_ij=0 leaves the local momentum constraint unsolved.\n\n'''
 report+=f"Workflow status: **{state['status']}**. Last verified checkpoint time: **{state.get('time')}**. The t=12 target was bounded by three evolution allocations and 48 raw AMD node-hours including preparation.\n\n"
 report+=f"Both individual horizons detected: {metrics.get('both_individual_horizons_formed')}; first shared formation time: {metrics.get('both_formation_time')}; measured coordinate revolutions after both form: {metrics.get('post_formation_revolutions')}. These are coordinate diagnostics; gaps are flagged. No merger or radiation-driven inspiral claim is made.\n\n"
 report+='The assessment precedes central collapse signals at r=40–70. Raw complex r*Psi4 is retained; strain and merger/ringdown interpretation are deferred. The inherited propagation mesh does not establish high-frequency waveform accuracy.\n\n## Review products\n\n'
 for name,caption in [('orbit.png','Coordinate trajectories, separation and phase; inspect gaps and orbital arc.'),('accepted_horizons.png','Strictly published same-candidate masses and coordinate spin; absent measurements are gaps.'),('particle_components.png','Alive component counts and covariant matter angular momentum; no conserved total is implied.'),('constraints.png','Proper-volume history norms with the code chi mask, not an AH exterior.'),('raw_waveform.png','Early outer-field/initialization response, not a merger waveform.'),('central.mp4','Central fixed-tag particle projection and accepted rmin illustrations.'),('context.mp4','Envelope/context fixed-tag view.')]:
  report+=f'- `{out/name}` — {caption}\n'
 report+='\nMovies passed encoder/decode checks. Representative frame and plot visual review by a person or agent remains pending. Source, logs, complex multipoles, masks, normalization, cadence and restart validity are preserved with checksum manifests.\n'
 (root/'REPORT_Jeans8.md').write_text(report)
 (root/'REPORT_AGENT.md').write_text('# Session 008 terminal workflow record\n\nAutomatically generated on Anta Slurm; visual QA remains pending. Full configuration, checkpoint, executable digest and per-run archive manifests are in evidence/ and runs/.\n\n```json\n'+json.dumps(state,indent=2)+'\n```\n\nAnalysis output: '+str(out)+'\n')

if __name__=='__main__':main()
