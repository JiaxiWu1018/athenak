"""Conservative waveform/strain, strict enclosure, separation and numerical-health review."""
import csv,json,subprocess,sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from runtime import atomic
from horizon_review_20261005 import accepted_measurements

def ffi(t,z,f0):
 if len(t)<16 or np.max(np.diff(t))>1.5*np.median(np.diff(t)):raise ValueError('waveform gap/insufficient samples')
 dt=.025;q=np.arange(t[0],t[-1]+dt*.1,dt)
 y=np.interp(q,t,z.real)+1j*np.interp(q,t,z.imag)
 # Linear detrend and 5%-edge cosine taper. No arbitrary time/phase alignment.
 y-=np.linspace(y[0],y[-1],len(y));window=np.ones(len(y));n=max(2,int(.05*len(y)))
 edge=.5*(1-np.cos(np.linspace(0,np.pi,n)));window[:n]=edge;window[-n:]=edge[::-1]
 freq=np.fft.fftfreq(len(y),dt);den=(2*np.pi*np.maximum(np.abs(freq),f0))**2
 h=np.fft.ifft(-np.fft.fft(y*window)/den)
 return q,h

def load_wave(runs):
 chunks=[]
 for run in runs:
  r=run/'waveforms/rpsi4_real_0050.txt';i=run/'waveforms/rpsi4_imag_0050.txt'
  if not r.exists() or not i.exists():continue
  real=np.loadtxt(r,ndmin=2);imag=np.loadtxt(i,ndmin=2)
  if real.shape!=imag.shape or real.shape[1]!=78 or not np.array_equal(real[:,0],imag[:,0]):raise RuntimeError('raw multipoles real/imag contract failed')
  chunks.append((real[:,0],real[:,1:]+1j*imag[:,1:]))
 if not chunks:return np.array([]),np.empty((0,77),complex)
 t=np.concatenate([x[0] for x in chunks]);z=np.concatenate([x[1] for x in chunks]);order=np.argsort(t,kind='stable');t=t[order];z=z[order]
 for k in np.where(np.diff(t)==0)[0]:
  if not np.allclose(z[k],z[k+1],rtol=1e-6,atol=1e-12):raise RuntimeError('conflicting waveform restart overlap')
 _,indices=np.unique(t,return_index=True);return t[indices],z[indices]

def common_enclosure(runs,tracks):
 good=[];failures=[]
 for run in runs:
  for p in run.glob('*.horizon_consumer_2.csv'):
   with p.open() as f:consumers=list(csv.DictReader(f))
   summaries=[]
   for q in run.glob('*.horizon_summary_2.txt'):
    summaries.extend([list(map(float,l.split())) for l in q.read_text().splitlines() if l.strip() and not l.startswith('#')])
   accepted,counts=accepted_measurements(consumers,sorted(summaries,key=lambda r:r[1]))
   for c,s in accepted:
    centers=[]
    for track in tracks:
     match=track[(track[:,0]==float(c['cycle'])) & (track[:,1]==float(c['time']))] if len(track) else []
     if len(match)==1 and np.isfinite(match[0][15]) and (match[0][13]>0 or match[0][16]>0):centers.append(match[0][2:5])
    distance=max((np.linalg.norm(v-np.array([float(c['center_'+a]) for a in 'xyz'])) for v in centers),default=np.inf)
    row=dict(run=run.name,cycle=int(c['cycle']),time=float(c['time']),mass=s[2],spin=s[13],rmin=float(c['rmin']),maximum_tracker_distance=float(distance) if np.isfinite(distance) else None,encloses_both=bool(len(centers)==2 and distance<float(c['rmin'])))
    (good if row['encloses_both'] else failures).append(row)
 return good,failures

def ringdown_review(t,z,common,wave_gate):
 result=dict(strict_common_accepted=False,common_encloses_both=False,ringdown_usable=False,gaps_checked=False,frequency_band_valid=False,reason='No accepted common surface proven to enclose both live trackers.')
 if not common or len(t)<10:return result
 c=common[0];result.update(strict_common_accepted=True,common_encloses_both=True,common=c)
 mass=c['mass'];spin=c['spin']
 if not .0<=spin<.99 or mass<=0:result['reason']='Unsupported remnant mass/spin';return result
 f=(1.5251-1.1568*(1-spin)**.1292)/(2*np.pi*mass);Q=.7+1.4187*(1-spin)**(-.499);tau=Q/(np.pi*f)
 result.update(expected_f220=f,expected_tau220=tau,qnm_reference='https://pages.jh.edu/eberti2/ringdown/fitcoeffsWEB.dat')
 if not wave_gate.get('passed') or f>.4:result['reason']='Expected ringdown outside validated frequency band';return result
 use=np.where((t>=c['time']+40)&(t<=c['time']+80))[0]
 if not len(use) or t[-1]<c['time']+80:result['reason']='Post-merger propagation interval incomplete';return result
 peak=use[np.argmax(np.abs(z[use,4]))];peak_time=float(t[peak]);result['wave_peak_time']=peak_time
 q=np.where((t>=peak_time+.5*tau)&(t<=peak_time+3.5*tau))[0]
 if len(q)<32:result['reason']='Insufficient damping samples';return result
 amplitude=np.abs(z[q,4]);phase=np.unwrap(np.angle(z[q,4]));x=t[q]-t[q][0]
 if np.any(amplitude<=0):result['reason']='Nonpositive waveform amplitude';return result
 la=np.log(amplitude);b=np.polyfit(x,la,1);p=np.polyfit(x,phase,1);fit=b[0]*x+b[1]
 r2=float(1-np.sum((la-fit)**2)/max(np.sum((la-la.mean())**2),1e-30));frequency=abs(p[0])/(2*np.pi);measured_tau=-1/b[0] if b[0]<0 else None
 contiguous=t[(t>=peak_time)&(t<=peak_time+100)]
 gaps=bool(len(contiguous)>2 and np.max(np.diff(contiguous))<=.0376)
 usable=bool(measured_tau is not None and r2>.9 and abs(frequency/f-1)<.2 and abs(measured_tau/tau-1)<.3 and np.std(np.diff(phase)/np.diff(t[q]))<.2*abs(p[0]))
 result.update(fit_f=frequency,fit_tau=measured_tau,log_amplitude_R2=r2,ringdown_usable=usable,gaps_checked=gaps,frequency_band_valid=bool(frequency<=.4),tail_complete=bool(t[-1]>=peak_time+100),reason='Conservative damped-mode fit; finite-radius approximate QNM agreement is qualitative, not convergence evidence.')
 return result

def extended_analysis(root,runs,out,metrics,tracks):
 from health9 import assess
 assess(root,runs,out)
 t,z=load_wave(runs)
 cutoff=metrics.get('analysis_cutoff')
 if cutoff is not None:q=t<=cutoff;t=t[q];z=z[q]
 np.savez(out/'raw_complex_all_modes_r50.npz',time=t,modes=z,lm=np.array([(l,m) for l in range(2,9) for m in range(-l,l+1)]))
 calibration=json.loads((root/'evidence/wave_gate.json').read_text())
 common,rejected=common_enclosure(runs,tracks)
 if cutoff is not None:common=[c for c in common if c['time']<=cutoff];rejected=[c for c in rejected if c['time']<=cutoff]
 atomic(out/'common_enclosure.json',dict(accepted_enclosing=common,published_without_proven_enclosure=rejected))
 review=ringdown_review(t,z,common,calibration)
 shell=np.loadtxt(out/'matter_near_r50.csv',delimiter=',',ndmin=2)
 near=shell[(shell[:,0]>=review.get('wave_peak_time',float('inf')))&(shell[:,0]<=review.get('wave_peak_time',0)+4*review.get('expected_tau220',1))] if shell.size else np.empty((0,4))
 matter_clean=bool(len(near) and np.all(near[:,2]/np.maximum(near[:,3],1e-30)<1e-4))
 review['matter_shell_clean']=matter_clean
 if not matter_clean:review['ringdown_usable']=False
 atomic(out/'ringdown_review.json',review)
 if review.get('tail_complete') and all(review[k] for k in ('strict_common_accepted','common_encloses_both','ringdown_usable','gaps_checked','frequency_band_valid')):
  from anta_archive import AMD,SSH
  cmd="import json,sys;sys.path.insert(0,'"+AMD+"/scripts');from runtime import atomic;atomic('"+AMD+"/control/ringdown_stop_receipt.json',json.load(sys.stdin))"
  subprocess.run(SSH+['python3 -c '+__import__('shlex').quote(cmd)],input=json.dumps(review),text=True,check=True)
 if len(t)>2:
  fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
  cadence=np.median(np.diff(t));groups=np.split(np.arange(len(t)),np.where(np.diff(t)>1.5*cadence)[0]+1)
  for col,label in ((0,'(2,-2)'),(4,'(2,2)'),(5,'(3,-3)'),(11,'(3,3)'),(20,'(4,4)')):
   for q in groups:
    amp=np.abs(z[q,col]);phase=np.unwrap(np.angle(z[q,col]));valid=amp>max(1e-15,1e-4*amp.max())
    phase[~valid]=np.nan
    axes[0].plot(t[q]-50,phase,label=label if q[0]==0 else None)
    if len(q)>2:axes[1].plot(t[q]-50,np.gradient(phase,t[q])/(2*np.pi),label=label if q[0]==0 else None)
  axes[0].set(ylabel='raw multipole phase [rad]');axes[1].set(xlabel='approximate retarded time (t-50)/M_ref',ylabel='signed phase frequency [cycles/M_ref]',ylim=(-.6,.6));axes[0].legend();axes[1].axhline(.4,ls=':',c='gray');axes[1].axhline(-.4,ls=':',c='gray');fig.tight_layout();fig.savefig(out/'wave_phase_frequency.png',dpi=170);plt.close(fig)
  convention=calibration.get('strain_convention_verified',False)
  if t[-1]>100 and convention and np.max(np.diff(t))<=1.5*cadence:
   fig,ax=plt.subplots(figsize=(9,4));cutoffs=(.003,.006,.012)
   for cutoff in cutoffs:
    q,h=ffi(t,z[:,4],cutoff);np.savetxt(out/f'strain_22_f0_{cutoff}.csv',np.c_[q,q-50,h.real,h.imag],delimiter=',',header='time,t_minus_50,r_hplus,r_minus_hcross')
    ax.plot(q-50,h.real,label=f'f0={cutoff} cycles/M_ref')
   ax.set(xlabel='approximate retarded time (t-50)/M_ref',ylabel='r h_22 real; finite radius');ax.legend();fig.tight_layout();fig.savefig(out/'strain_cutoff_sensitivity.png',dpi=170);plt.close(fig)
   atomic(out/'strain_method.json',dict(convention='H=r(hplus-i hcross), Hddot=rPsi4 verified by source TT limit and linear-wave calibration.',method='Fixed-frequency double integration -FFT(rPsi4)/(2pi max(|f|,f0))^2; uniformdt.025 linear interpolation, linear detrend,5% cosine edge taper; cutoffs.003/.006/.012. Edges/transients and finite-radius strain not precision waveform.',gaps=False))
  else:atomic(out/'strain_method.json',dict(deferred=True,reason='Requires causal coverage>100, uninterrupted waveform and verified sign/normalization calibration; rawPsi4 remains primary.'))
 # Simultaneous accepted AH distances and full surviving-particle centroid distances.
 subprocess.run([sys.executable,str(root/'scripts/horizon_review_20261005.py'),'--runs-root',str(root/'runs'),'--output',str(out/'horizon_review'),'--max-time',str(cutoff if cutoff is not None else 'inf')],check=True)
 if (out/'horizon_review/accepted_horizon_0.csv').exists() and (out/'horizon_review/accepted_horizon_1.csv').exists() and (out/'particle_components.csv').exists():
  subprocess.run([sys.executable,str(root/'scripts/plot_separation_20261005.py'),'--tables',str(out),'--horizons',str(out/'horizon_review'),'--output',str(out/'separation')],check=True)
 metrics['accepted_enclosing_common_rows']=len(common);metrics['ringdown_review']=review
 atomic(out/'summary.json',metrics)
