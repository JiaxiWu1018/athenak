#!/usr/bin/env python3
"""Eight-cycle TT calibration with Weyl enabled; preserve the 80M gate receipt."""
import argparse,json,sys
import numpy as np
from runtime import ROOT,atomic,digest

def generate():
 for w in (2.5,5.):
  source=ROOT/'inputs/wave_tests'/('lambda_'+str(w)+'.athinput')
  text=source.read_text().replace('nlim = 100000000','nlim = 8')
  text=text.replace('nrad_wave_extraction = 0',
   'nrad_wave_extraction = 1\nextraction_radius_0 = 1\nextraction_nlev = 3\nwaveform_dt = 0.000001')
  target=ROOT/'inputs/wave_tests'/('weyl_lambda_'+str(w)+'.athinput')
  target.write_text(text)

def assess():
 sys.path.insert(0,str(ROOT/'athenak/vis/python'));import bin_convert
 cases=[]
 for w in (2.5,5.):
  run=ROOT/'runs'/('wave_weyl_lambda_'+str(w))
  paths=sorted(run.rglob('*.weyl.*.bin'))
  if len(paths)<2:raise RuntimeError('missing evolved Weyl snapshot')
  path=paths[-1];data=bin_convert.read_binary(str(path))
  nx=int(data['nx1_out_mb']);ny=int(data['nx2_out_mb'])
  expected=[];real_values=[];imag_values=[]
  for g,real,imag in zip(data['mb_geometry'],data['mb_data']['weyl_rpsi4'],data['mb_data']['weyl_ipsi4']):
   x=g[0]+(np.arange(nx)+.5)*(g[1]-g[0])/nx
   y=g[2]+(np.arange(ny)+.5)*(g[3]-g[2])/ny
   xx,yy=np.meshgrid(x,y)
   # Finest existing TT cells near +x: tetrad theta=-z, phi=+y.
   q=(xx>4)&(xx<7)&(np.abs(yy)<.04)
   zz=(g[4]+g[5])/2
   radius=np.sqrt(xx[q]**2+yy[q]**2+zz**2)
   expected.extend(-(2*np.pi/w)**2*1e-5*np.sin(2*np.pi*(xx[q]-float(data['time']))/w))
   real_values.extend(np.asarray(real).reshape(ny,nx)[q]/radius)
   imag_values.extend(np.asarray(imag).reshape(ny,nx)[q]/radius)
  e=np.asarray(expected);r=np.asarray(real_values);im=np.asarray(imag_values)
  if not len(e) or not np.isfinite(np.r_[e,r,im]).all():raise RuntimeError('invalid TT calibration cells')
  scale=float(np.dot(e,r)/np.dot(e,e))
  residual=float(np.linalg.norm(r-scale*e)/np.linalg.norm(e))
  cross=float(np.linalg.norm(im)/np.linalg.norm(e))
  cases.append(dict(wavelength=w,time=float(data['time']),cells=len(e),scale=scale,
   relative_residual=residual,cross_leakage=cross,input_sha256=digest(ROOT/'inputs/wave_tests'/('weyl_lambda_'+str(w)+'.athinput')),
   snapshot=str(path),snapshot_sha256=digest(path),passed=bool(abs(scale-1)<.1 and residual<.1 and cross<.1)))
 receipt=dict(passed=all(c['passed'] for c in cases),cases=cases,
  convention='Psi4=ddot(hplus-i*hcross), with the implemented orthonormal theta/phi tetrad; saved fields and multipoles are r*Psi4.',
  method='Eight evolved RK4 cycles, extraction enabled, direct analytic TT comparison at dx1/16 near +x; plus polarization tests scale/sign, negligible cross leakage. Imaginary sign is checked from source tetrad/contraction, not a second polarized numerical test.',
  reason='The original 80M propagation inputs disabled extraction, so their Weyl arrays were zero. Their valid metric propagation measurements are preserved unchanged.',
  job=__import__('os').environ.get('SLURM_JOB_ID'))
 atomic(ROOT/'evidence/weyl_convention.json',receipt)
 gate_path=ROOT/'evidence/wave_gate.json'
 gate=json.loads(gate_path.read_text())
 original=ROOT/'evidence/wave_gate_before_weyl_calibration.json'
 if not original.exists():atomic(original,gate)
 gate['strain_convention_verified']=receipt['passed']
 gate['weyl_convention_receipt']='evidence/weyl_convention.json'
 gate['original_disabled_extraction_calibration']='Nonphysical zero arrays; superseded only for strain convention by the separately recorded enabled-extraction test.'
 atomic(gate_path,gate)
 if not receipt['passed']:raise RuntimeError('Weyl convention calibration failed; strain stays deferred; no automatic retry')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','assess']);a=p.parse_args()
 generate() if a.action=='generate' else assess()
