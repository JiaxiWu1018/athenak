"""Pure postprocessing utilities; importing does not analyze any simulation data.

Native diagnostics are coordinate quantities. No covariance/causality/convergence
claim follows from these utilities. Movies and final visual review are post-run work.
"""
import numpy as np

def discrete_jz_terms(x0,u0,x1,u1,mass):
 x0,u0,x1,u1=map(np.asarray,(x0,u0,x1,u1))
 midx=(x0+x1)/2;midu=(u0+u1)/2
 momentum=mass*(midx[...,0]*(u1-u0)[...,1]-midx[...,1]*(u1-u0)[...,0])
 position=mass*(midu[...,1]*(x1-x0)[...,0]-midu[...,0]*(x1-x0)[...,1])
 return momentum,position

def gap_derivative(t,y,max_gap=0.05):
 t=np.asarray(t);y=np.asarray(y);out=np.full(y.shape,np.nan)
 breaks=np.r_[0,np.where(np.diff(t)>max_gap+1e-9)[0]+1,len(t)]
 for a,b in zip(breaks[:-1],breaks[1:]):
  if b-a>=3:out[a:b]=np.gradient(y[a:b],t[a:b])
 return out

def orbit(t,left,right,valid):
 t=np.asarray(t);rel=np.asarray(right)-np.asarray(left);d=np.linalg.norm(rel,axis=1)
 phase=np.full(len(t),np.nan);arc=np.full(len(t),np.nan);offset=0.0
 groups=[];current=[]
 for k in range(len(t)):
  if not valid[k] or (current and t[k]-t[current[-1]]>0.05+1e-9):
   if current:groups.append(current);current=[]
  if valid[k]:current.append(k)
 if current:groups.append(current)
 omega=np.full(len(t),np.nan);vr=omega.copy()
 for group in groups:
  q=np.array(group);p=np.unwrap(np.arctan2(rel[q,1],rel[q,0]))
  phase[q]=p;arc[q]=offset+p-p[0];offset=arc[q[-1]]
  if len(q)>=3:omega[q]=np.gradient(p,t[q]);vr[q]=np.gradient(d[q],t[q])
 return dict(time=t,separation=d,phase=phase,observed_arc=arc,Omega=omega,
             radial_speed=vr,tangential_relative_speed=d*omega,
             orbital_timescale=np.divide(2*np.pi,np.abs(omega),out=np.full(len(t),np.nan),where=np.abs(omega)>0),
             inward_timescale=np.divide(d,-vr,out=np.full(len(t),np.nan),where=vr<0))

def ffi(time,rpsi4,cutoff,taper=True):
 """Finite-radius r(hplus-i hcross); no infinity extrapolation or DC recovery.

 Requires a contiguous uniform series. Select a supported band and review cutoff,
 edge/window and radius sensitivity before publishing. Never bridge restart gaps.
 """
 t=np.asarray(time);psi=np.asarray(rpsi4,dtype=complex)
 if len(t)<8 or psi.shape!=(len(t),) or cutoff<=0 or not np.isfinite(psi).all():
  raise ValueError('Invalid FFI input')
 dt=np.diff(t)
 if np.any(dt<=0) or not np.allclose(dt,dt[0],rtol=1e-6,atol=1e-10):
  raise ValueError('FFI requires reviewed uniform contiguous input; no silent gap interpolation')
 w=np.ones(len(t))
 if taper:
  n=max(2,int(len(t)*0.1));edge=np.sin(np.linspace(0,np.pi/2,n))**2
  w[:n]=edge;w[-n:]=edge[::-1]
 angular=2*np.pi*np.fft.fftfreq(len(t),dt[0]);den=np.maximum(np.abs(angular),2*np.pi*cutoff)**2
 spectrum=-np.fft.fft(psi*w)/den;spectrum[0]=0
 return np.fft.ifft(spectrum)

def removal_corrected_jz(alive,removed_cumulative,momentum_change,position_change):
 """Interval closure, including momentum changes before later removal.

 It is not solely a surviving-tag torque and not total gravitational Jz.
 """
 return np.diff(np.asarray(alive)+np.asarray(removed_cumulative))-np.asarray(momentum_change)[1:]-np.asarray(position_change)[1:]
