#!/usr/bin/env python3
"""Existing linear TT generator: two wavelengths across static AMR interfaces."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from runtime import ROOT,atomic

def generate():
 out=ROOT/'inputs/wave_tests';out.mkdir(exist_ok=True)
 for wavelength in (2.5,5.):
  name='lambda_'+str(wavelength)
  text='''<job>
basename = linear_gw
<mesh>
nghost = 4
nx1 = 320
x1min = -40
x1max = 40
ix1_bc = periodic
ox1_bc = periodic
nx2 = 32
x2min = -4
x2max = 4
ix2_bc = periodic
ox2_bc = periodic
nx3 = 32
x3min = -4
x3max = 4
ix3_bc = periodic
ox3_bc = periodic
<meshblock>
nx1 = 32
nx2 = 32
nx3 = 32
<mesh_refinement>
refinement = static
num_levels = 3
max_nmb_per_rank = 240
<refined_region1>
level = 1
x1min = -16
x1max = 16
x2min = -4
x2max = 4
x3min = -4
x3max = 4
<refined_region2>
level = 2
x1min = -8
x1max = 8
x2min = -2
x2max = 2
x3min = -2
x3max = 2
<time>
evolution = dynamic
integrator = rk4
cfl_number = 0.4
nlim = 100000000
ndiag = 100
'''
  text+=f'tlim = {80/wavelength}\n'
  text+='''<z4c>
lapse_harmonicf = 1.0
lapse_harmonic = 0.0
lapse_oplog = 2.0
lapse_advect = 1.0
shift_eta = 2.0
shift_advect = 1.0
diss = 0.1
chi_div_floor = 0.00001
damp_kappa1 = 0.02
damp_kappa2 = 0.0
nrad_wave_extraction = 0
<problem>
pgen_name = z4c_linear_wave
amp = 0.00001
kx2 = 0
kx3 = 0
'''
  text+=f'kx1 = {1/wavelength}\n'
  text+='''<output1>
file_type = bin
variable = z4c_gzz
id = gw
slice_x3 = 0
dt = 80
<output2>
file_type = bin
variable = weyl
id = weyl
slice_x3 = 0
dt = 80
'''
  (out/(name+'.athinput')).write_text(text)

def coefficients(data,wavelength):
 # Area-weighted projection over ALL leaf cells of the plane: no native-AMR seams.
 matrix=np.zeros((3,3));rhs=np.zeros(3);norm=0.;vol=0.
 geometry=np.asarray(data['mb_geometry']);nx=int(data['nx1_out_mb']);ny=int(data['nx2_out_mb'])
 for block,g in zip(data['mb_data']['z4c_gzz'],geometry):
  x=g[0]+(np.arange(nx)+.5)*(g[1]-g[0])/nx
  signal=np.asarray(block).reshape(ny,nx)-1
  if not np.isfinite(signal).all():raise RuntimeError('nonfinite linear GW state')
  a=np.stack([np.broadcast_to(f,(ny,nx)).ravel() for f in (np.sin(2*np.pi*x/wavelength),np.cos(2*np.pi*x/wavelength),np.ones(nx))],axis=1)
  y=signal.ravel();weight=(g[1]-g[0])*(g[3]-g[2])/(nx*ny)
  matrix+=weight*a.T@a;rhs+=weight*a.T@y;norm+=weight*np.dot(y,y);vol+=weight*len(y)
 b=np.linalg.solve(matrix,rhs)
 return dict(amplitude=float(np.hypot(b[0],b[1])),phase=float(np.arctan2(b[1],b[0])),offset=float(b[2]),rms=float(np.sqrt(norm/vol)))

def assess():
 sys.path.insert(0,str(ROOT/'athenak/vis/python'));import bin_convert
 cases=[]
 for w in (2.5,5.):
  run=ROOT/'runs'/('wave_lambda_'+str(w));files=sorted(run.rglob('*.gw.*.bin'))
  if len(files)<2:raise RuntimeError('missing initial/final plane')
  initial=bin_convert.read_binary(str(files[0]));final=bin_convert.read_binary(str(files[-1]))
  a=coefficients(initial,w);b=coefficients(final,w)
  phase=float(np.angle(np.exp(1j*(b['phase']-a['phase']+2*np.pi*(final['time']-initial['time'])/w))))
  error=abs(b['amplitude']/a['amplitude']-1)
  case=dict(wavelength=w,time_initial=float(initial['time']),time_final=float(final['time']),initial=a,final=b,amplitude_fractional_error=error,phase_error_rad=abs(phase),passed=bool(abs(final['time']-80)<.1 and error<=.1 and abs(phase)<=.2))
  weyl_files=sorted(run.rglob('*.weyl.*.bin'))
  if weyl_files:
   wd=bin_convert.read_binary(str(weyl_files[0]));xs=[];ys=[];rs=[];ims=[]
   nx=int(wd['nx1_out_mb']);ny=int(wd['nx2_out_mb'])
   for g,real,imag in zip(wd['mb_geometry'],wd['mb_data']['weyl_rpsi4'],wd['mb_data']['weyl_ipsi4']):
    x=g[0]+(np.arange(nx)+.5)*(g[1]-g[0])/nx;y=g[2]+(np.arange(ny)+.5)*(g[3]-g[2])/ny
    xx,yy=np.meshgrid(x,y);q=(xx>20)&(xx<36)&(np.abs(yy)<.26)
    xs.extend(xx[q]);ys.extend(yy[q]);rs.extend(np.asarray(real).reshape(ny,nx)[q]);ims.extend(np.asarray(imag).reshape(ny,nx)[q])
   x=np.asarray(xs);y=np.asarray(ys);r=np.hypot(x,y)
   expected=-(2*np.pi/w)**2*1e-5*np.sin(2*np.pi*x/w)
   observed=np.asarray(rs)/r
   scale=float(np.dot(expected,observed)/np.dot(expected,expected))
   residual=float(np.linalg.norm(observed-scale*expected)/np.linalg.norm(expected))
   cross=float(np.linalg.norm(np.asarray(ims)/r)/np.linalg.norm(expected))
   case['weak_field_weyl_calibration']=dict(expected='Psi4=ddot(hplus-i hcross); hplus=gzz-1 on+x near axis',scale=scale,relative_residual=residual,cross_leakage=cross,passed=bool(abs(scale-1)<.1 and residual<.1 and cross<.1))
  cases.append(case)
 result=dict(strain_convention_verified=False,passed=all(c['passed'] for c in cases),cases=cases,method='Area-weighted sine/cosine/constant fit of gzz-1 over every leaf cell on z0; periodic plane travels80M across dx.25/.125/.0625 static AMR; same RK4/CFL/gauge/dissipation. Finite sampling heuristic, not nonlinear convergence evidence.',limitations='Axis-aligned linear wave only; no spherical/nonlinear/ringdown convergence proof.')
 result['strain_convention_verified']=all(c.get('weak_field_weyl_calibration',{}).get('passed',False) for c in cases)
 atomic(ROOT/'evidence/wave_gate.json',result)
 if not result['passed']:raise RuntimeError('linear propagation thresholds failed; inspect receipt; no mesh change or retries')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','assess']);a=p.parse_args();generate() if a.action=='generate' else assess()
