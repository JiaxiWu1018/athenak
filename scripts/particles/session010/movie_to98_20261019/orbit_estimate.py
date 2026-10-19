#!/usr/bin/env python3
"""Analytic reference scales and existing-data diagnostics, not a new evolution."""
import argparse,hashlib,json,os,datetime as dt
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--session',type=Path,required=True);args=ap.parse_args();s=args.session.resolve()
 assert os.environ.get('SLURM_JOB_ID')
 out=s/'plots/orbit_duration_20261019_v1';out.mkdir(parents=True,exist_ok=True)
 assert not (out/'ESTIMATES.json').exists()
 data=s/'session010_summary_20261019/workspace/runtime_s10/analysis/through_segment002_20261018'
 profile=s/'session010_summary_20261019/workspace/runtime_s10/runs/20261014_gh_retry_01/segments/segment_001/prefix_seed/out/gi_profile_M076_two_clump_s10.txt'
 p=np.loadtxt(profile);R=p[:,0];psi0=p[:,2];alpha0=p[:,3]
 dq=4*np.pi*R**2*p[:,6];mq=np.r_[0,np.cumsum((dq[:-1]+dq[1:])*np.diff(R)/2)]
 dp=np.gradient(psi0,R,edge_order=2);dla=np.gradient(np.log(alpha0),R,edge_order=2)
 trials=[]
 for d in (6.,8.,10.,12.,14.,16.):
  r=d/2;p0=float(np.interp(r,R,psi0));a0=float(np.interp(r,R,alpha0));p0p=float(np.interp(r,R,dp))
  # Hold both clump centers fixed while differentiating the field at one center.
  # A Gaussian's own gradient is zero at its center; the other is at distance d.
  psi=p0+.12/(np.sqrt(2*np.pi)*.70)+.12/(2*d)
  pp=p0p-.12/(2*d*d)
  alpha=a0*(p0/psi)**2
  dlogalpha=float(np.interp(r,R,dla))+2*(p0p/p0-pp/psi)
  v=np.sqrt(r*dlogalpha/(1+2*r*pp/psi));vc=alpha/psi**2*v
  menv=float(np.interp(r,R,mq))
  trials.append(dict(separation=d,center_radius=r,envelope_conformal_source_inside_radius=menv,
    local_instantaneous_static_metric_circular_speed=float(v),coordinate_speed=float(vc),
    coordinate_period=float(2*np.pi*r/vc),newtonian_monopole_speed=float(np.sqrt(menv/r+.12/(2*d))),
    vacuum_kepler_period_Mbin024=float(2*np.pi*np.sqrt(d**3/.24)),
    vacuum_reference_turns_to_6Mbin=float(((d/.24)**2.5-6**2.5)/(16*np.pi)),
    unvalidated_empirical_turns=float(.6704238477656899*(d/6)**2.5)))
 toy=[]
 for mass in (.24,.28,.36):
  toy.append(dict(binary_mass=mass,separation_over_binary_mass=6/mass,
    kepler_period=float(2*np.pi*np.sqrt(6**3/mass)),
    vacuum_time_to_formal_zero=float(5/64*6**4/mass**3),
    vacuum_turns_to_6Mbin=float(((6/mass)**2.5-6**2.5)/(16*np.pi))))
 orbit=np.genfromtxt(data/'orbit.csv',delimiter=',',names=True)
 keep=orbit['valid_pre_common_binary'].astype(bool);arc=orbit['observed_arc_excluding_gaps']/(2*np.pi)
 measured=[]
 h=[np.genfromtxt(data/f'accepted_horizon_{i}.csv',delimiter=',',names=True) for i in (0,1)]
 for tm in (10,20,40,60):
  ii=np.argmin(abs(orbit['time']-tm));m=[]
  for a in h:
   j=np.argmin(abs(a['time']-tm));m.append(dict(time=float(a[j]['time']),mass=float(a[j]['M_BH'])))
  measured.append(dict(target_time=tm,tracker_time=float(orbit[ii]['time']),separation=float(orbit[ii]['coordinate_separation']),turns=float(arc[ii]),individual_masses=m))
 baseline_vco=.133215*.7889962/(1.124482509**2)
 result=dict(job=os.environ['SLURM_JOB_ID'],checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
  source_commit='386870f88f60e25e1a8fac76fe04a071b87a01cf',observed_turns=.6704238477656899,
  observed_post_individual_turns=.6259206743506687,initial_coordinate_period_from_actual_boost=float(2*np.pi*3/baseline_vco),
  source_mass_binary=.24,source_mass_envelope=.76,reference_eta=.25,
  formulae={'T':'2*pi*sqrt(a^3/Mbin)','da_dt':'-(64/5)*eta*Mbin^3/a^3',
  'N':'[(a0/Mbin)^(5/2)-(af/Mbin)^(5/2)]/(64*pi*eta)',
  'vlocal_static':'sqrt(R*dln(alpha)/dR/[1+2*R*dln(psi)/dR])'},
  vacuum_baselines=toy,separation_trials=trials,measured_samples=measured,
  empirical_unvalidated_solutions={str(n):float(6*(n/.6704238477656899)**.4) for n in (2,3)},
  empirical_if_literal_half_turn={str(n):float(6*(n/.5)**.4) for n in (2,3)},
  caveat='Vacuum constant-mass circular quadrupole and static inherited-metric test-particle estimates only. No new constraint solve or dynamical friction/accretion/eccentricity/gauge model. Not predictions or input approvals.',
  sources={str(path.relative_to(s)):sha(path) for path in (profile,data/'orbit.csv',data/'accepted_horizon_0.csv',data/'accepted_horizon_1.csv')})
 (out/'ESTIMATES.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 fig,axes=plt.subplots(1,2,figsize=(11,4.4))
 ix=np.flatnonzero(keep);breaks=np.r_[0,np.flatnonzero((np.diff(ix)>1)|(np.diff(orbit['time'][ix])>.04))[0]+1,len(ix)]
 for lo,hi in zip(breaks[:-1],breaks[1:]):
  j=ix[lo:hi];axes[0].plot(arc[j],orbit['coordinate_separation'][j],color='#1768ac',lw=1.8)
 axes[0].set(xlabel='Observed turns from t=0 (gaps excluded)',ylabel='Coordinate separation / Mref',title='Measured short inward orbit and plunge')
 ds=[r['separation'] for r in trials];vs=[r['local_instantaneous_static_metric_circular_speed'] for r in trials]
 axes[1].plot(ds,vs,'o-',label='Instantaneous force balance in inherited t=0 metric')
 axes[1].axhline(.133215,color='black',ls='--',label='Current local boost = 0.133215')
 axes[1].set(xlabel='Trial initial separation / Mref',ylabel='Local orthonormal speed',title='Envelope changes the speed needed at larger radius')
 axes[1].legend(fontsize=7)
 for ax in axes:ax.grid(alpha=.2)
 fig.text(.5,.01,'Static test-particle estimates are initial guesses only; no new input or simulation. No guarantee of 2–3 orbits.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.04,1,1))
 for ext in ('png','pdf'):fig.savefig(out/f'orbit_duration_assessment.{ext}',dpi=170)
 plt.close(fig)
 print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
