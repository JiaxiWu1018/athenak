#!/usr/bin/env python3
"""Preparation/provenance only. No native execution or Slurm submission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

HERE=Path(__file__).resolve().parent
CODE=HERE.parents[2]
SESSION=CODE.parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
 return h.hexdigest()

def atomic(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 temp=p.with_suffix(p.suffix+'.tmp')
 temp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');os.replace(temp,p)

def source_files():
 return sorted([*CODE.glob('src/**/*.cpp'),*CODE.glob('src/**/*.hpp'),
                *CODE.glob('src/**/CMakeLists.txt'),CODE/'CMakeLists.txt',
                HERE/'kokkos_4_7_2_arm_sve_compat.cmake'])

def source_hash():
 h=hashlib.sha256()
 for p in source_files(): h.update(str(p.relative_to(CODE)).encode()+b'\0'+sha(p).encode())
 return h.hexdigest()

def parse(s):
 out={};block=None
 for line in s.splitlines():
  line=line.split('#',1)[0].strip()
  if re.fullmatch(r'<[^>]+>',line): block=line[1:-1];out.setdefault(block,{})
  elif block and block!='comment' and '=' in line:
   k,v=line.split('=',1);out[block][k.strip()]=v.strip()
 return out

def change(text,overrides,added=None):
 block=None;lines=[]
 for line in text.splitlines():
  m=re.fullmatch(r'<([^>]+)>',line.strip())
  if m:
   if block and added:
    for k,v in added.get(block,{}).items():lines.append(f'{k} = {v}')
   block=m[1]
  key=line.split('=',1)[0].strip() if '=' in line else None
  if (block,key) in overrides:line=f'{key} = {overrides[block,key]}'
  lines.append(line)
 if block and added:
  for k,v in added.get(block,{}).items():lines.append(f'{k} = {v}')
 return '\n'.join(lines)+'\n'

def generate():
 original=(HERE/'inputs/session010_reference.athinput').read_text()
 overrides={('job','basename'):'s11_orbit_gw',('time','tlim'):'150.0',
 ('time','nlim'):'-1',('z4c','nrad_wave_extraction'):'3',('fastflow','verbose'):'false'}
 for i in range(3):overrides['fastflow',f'stop_time_{i}']='150.0'
 for i in range(2):overrides['z4c',f'co_{i}_out_every']='1'
 for i in range(1,22):
  overrides[f'output{i}','dt']='0.025' if i in (1,8) else '0.5' if i==5 else '10.0' if i in (6,7,18,19,20,21) else '0.25'
 text=change(original,overrides,{'z4c':{'extraction_radius_1':'50.0','extraction_radius_2':'60.0'},
 'fastflow':{'diagnostic_dt':'0.025'},'particles':{'orbit_diagnostic_dt':'0.025'}})
 text=text.replace('Session010 approved exploratory fresh t0','Session011 approved final fresh t0')
 text=text.replace('Resource/termination/output limits require explicit Vista discussion before production.','One 16-node 36h job; hard t=150; validation remains required.')
 text=text.replace('This input is a physics template; tlim400 is provisional, not Session010 authorization.','Prepared input, not permission to bypass GPU validation or submit automatically.')
 before,after=parse(original),parse(text)
 allowed=set(overrides)|{('z4c','extraction_radius_1'),('z4c','extraction_radius_2'),('fastflow','diagnostic_dt'),('particles','orbit_diagnostic_dt')}
 for block in set(before)|set(after):
  for key in set(before.get(block,{}))|set(after.get(block,{})):
   if before.get(block,{}).get(key)!=after.get(block,{}).get(key) and (block,key) not in allowed:
    raise RuntimeError(f'Unapproved input change: {block}/{key}')
 (HERE/'inputs/production.athinput').write_text(text)
 compact=(HERE/'inputs/compact_reference.athinput').read_text()
 compact=change(compact,{('job','basename'):'s11_smoke',('time','nlim'):'2'},
                {'particles':{'orbit_diagnostic_dt':'0.025'},'fastflow':{'diagnostic_dt':'0.025'}})
 (HERE/'inputs/compact_validation.athinput').write_text(compact)
 # Compact, static refinement OS fixture. Distinct from production; no old comments
 # claiming MPI feedback is unsupported are inherited from the historical example.
 common='''<job>\nbasename = os_ah\n<mesh>\nnghost = 4\nnx1 = 64\nx2 = 64\nnx3 = 64\nx1min = -64\nx1max = 64\nx2min = -64\nx2max = 64\nx3min = -64\nx3max = 64\nix1_bc = outflow\nox1_bc = outflow\nix2_bc = outflow\nox2_bc = outflow\nix3_bc = outflow\nox3_bc = outflow\n<meshblock>\nnx1 = 16\nnx2 = 16\nnx3 = 16\n<mesh_refinement>\nrefinement = static\nmax_nmb_per_rank = 400\n<time>\nevolution = dynamic\nintegrator = rk4\ncfl_number = 0.4\nnlim = -1\ntlim = 40\nndiag = 100\nstop_file = REQUEST_STOP\nstop_check_every = 16\n<z4c>\nlapse_harmonic = 0\nlapse_oplog = 2\nshift_eta = 2\ndiss = 0.1\nchi_div_floor = 0.00001\ndamp_kappa1 = 0.02\ndamp_kappa2 = 0\n'''
 for level,r in enumerate([32,16,8,4,2,1],1):
  common+=f'<refined_region{level}>\nlevel = {level}\n'+''.join(f'x{a}min = {-r}\nx{a}max = {r}\n' for a in (1,2,3))
 ff='''<fastflow>\nnpunct = 0\nnum_horizons = 1\nntheta = 10\nlmax = 6\nuse_puncture_0 = -1\ninitial_radius_0 = 1\nexpand_guess = 1\nreuse_last_surface_shape_0 = true\npersist_surface_shape_0 = true\nconsumer_rmin_0 = 0\nconsumer_rmin_cells_0 = 4\nconsumer_rmax_0 = 4\nconsumer_hrel_0 = 0.2\nconsumer_persist_0 = 3\nflow_iterations_0 = 400\nflow_alpha_beta_const_0 = 0.005\nhmean_tol_0 = 100\nmass_tol_0 = 0.0001\nstart_time_0 = 0\nstop_time_0 = 40\ndiagnostic_dt = 0.025\nverbose = false\noutput_ylm = true\noutput_grid = false\n'''
 outputs='''<output1>\nfile_type = rst\ndt = 5\n<output2>\nfile_type = hst\ndt = 0.025\n<output3>\nfile_type = bin\nvariable = z4c\nid = metric\nslice_x3 = 0\ndt = 0.5\n'''
 os_input=common+ff+'''<particles>\nparticle_type = dust\npusher = gr_boris\ninit = pgen\nfeedback = true\nmass = 1\ndebug = 1\ndestroy_log = true\nexcise_radius = 0\nexcise_lapse = 0.05\nexcise_ah = false\ncross_level_deposit = conservative\norbit_diagnostic_dt = 0.025\n<problem>\nos_mass = 1\nos_radius_over_mass = 5\nos_lattice_n = 32\nos_precollapsed_lapse = false\n'''+outputs
 (HERE/'inputs/os_validation.athinput').write_text(os_input)
 spin=common.replace('basename = os_ah','basename = spin_ah').replace('tlim = 40','tlim = 0.25')
 spin+=ff.replace('initial_radius_0 = 1','initial_radius_0 = 2').replace('flow_alpha_beta_const_0 = 0.005','flow_alpha_beta_const_0 = 0.1')
 spin+='''<problem>\npunc_ADM_mass = 1\npunc_spin = 0.5\n'''+outputs
 (HERE/'inputs/spin_validation.athinput').write_text(spin)
 atomic(SESSION/'evidence/PREPARATION.json',dict(created_epoch=time.time(),
  source_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),
  native_source_sha256=source_hash(),production_input_sha256=sha(HERE/'inputs/production.athinput'),
  physics_input_diff_verified=True,production_ready=False,science_runs_performed=False,
  conditional_push_performed=False,validation_pending=['OS collapse/restart','bad guess regression','spin/angular resolution','full output contract','full-size startup']))

def build_receipt(kind):
 exe=SESSION/'build'/kind/'src/athena'
 atomic(SESSION/f'evidence/build/{kind}.json',dict(kind=kind,executable=str(exe),
  executable_sha256=sha(exe),native_source_sha256=source_hash(),job_id=os.environ.get('SLURM_JOB_ID'),
  host=socket.gethostname(),epoch=time.time(),compiled=True,simulation_tested=False,
  kokkos_commit='6739bc623081648af9e752b616d9671527922cbf'))
 print('COMPILED (not simulation-validated):',kind,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','build-receipt']);p.add_argument('kind',nargs='?');a=p.parse_args()
 generate() if a.action=='generate' else build_receipt(a.kind)
