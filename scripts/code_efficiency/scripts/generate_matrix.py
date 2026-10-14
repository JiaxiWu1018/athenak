#!/usr/bin/env python3
import os
"""Deterministic input/matrix authoring. Safe to run as a small metadata task."""
import json
from pathlib import Path
ROOT=Path(os.environ.get('EFFICIENCY_SESSION',Path(__file__).resolve().parents[1]))
cases=[]
def add(name,test,metric='st',feedback=True,n=1000000,grid=64,block=32,ranks=1,half=4,ppc=6,sampling='random',distribution='uniform',frozen=True,mesh_only=False):
    c=dict(name=name,test=test,metric=metric,feedback=feedback,n=n,grid=grid,block=block,ranks=ranks,half=half,ppc=ppc,sampling=sampling,distribution=distribution,frozen=frozen,mesh_only=mesh_only)
    c['n']=ppc*grid**3 if sampling=='replicas' else n
    cases.append(c)
for metric in ['st','flat']:
  for feedback in [False,True]:
    suffix=f'{metric}_dep{int(feedback)}'
    for n in [10000,100000,1000000,10000000,100000000]:
      add(f'count_{suffix}_N{n}',1,metric,feedback,n,32,32)
    for block in [64,32,16]:add(f'blocks_{suffix}_B{block}',2,metric,feedback,block=block)
    for ranks in [2,4]:add(f'ranks_{suffix}_R{ranks}',3,metric,feedback,ranks=ranks)
for ppc in [6,24,96,384]:add(f'dynamic_ppc{ppc}',4,half=12,ppc=ppc,sampling='replicas',frozen=False)
add('dynamic_mesh_only',4,half=12,ppc=6,sampling='replicas',frozen=False,mesh_only=True)
add('dynamic_matched_frozen_ppc24',4,half=12,ppc=24,sampling='replicas',frozen=True)
for distribution in ['uniform','central4','central1','block','point']:
  for ranks in [1,2,4]:
    if distribution=='uniform' and ranks==1:continue
    add(f'sampling_{distribution}_R{ranks}',5,ranks=ranks,half=12,ppc=24,sampling='replicas',distribution=distribution,frozen=False)
def deck(c,profile=False,tlim=5,warmup=.5):
    bc='periodic' if c['half']==4 else 'outflow'
    cfl=.05/(2*c['half']/c['grid'])
    mesh=''.join(f'nx{d} = {c["grid"]}\nx{d}min = {-c["half"]}\nx{d}max = {c["half"]}\nix{d}_bc = {bc}\nox{d}_bc = {bc}\n' for d in [1,2,3])
    return f'''<comment>
problem = Paper Session 02 performance benchmark {c['name']}
<job>
basename = {c['name']}
<mesh>
nghost = 4
{mesh}
<meshblock>
nx1 = {c['block']}
nx2 = {c['block']}
nx3 = {c['block']}
<time>
evolution = dynamic
integrator = rk4
cfl_number = {cfl:.17g}
tlim = {tlim}
nlim = {round(tlim/.05)}
ndiag = 10
<z4c>
frozen_background = {str(c['frozen']).lower()}
use_z4c = true
lapse_harmonic = 0.0
lapse_oplog = 2.0
shift_driver = legacy
shift_Gamma = 1.0
shift_H = 0.0
shift_alpha2Gamma = 0.0
shift_eta = 2.0
diss = 0.1
chi_div_floor = 0.00001
damp_kappa1 = 0.02
damp_kappa2 = 0.0
<particles>
particle_type = dust
pusher = gr_boris
init = pgen
feedback = {str(c['feedback']).lower()}
mass = 1.0
debug = 0
destroy_log = false
excise_radius = 0.0
excise_lapse = 0.0
excise_ah = false
cross_level_deposit = conservative
<fastflow>
num_horizons = 0
<benchmark>
enabled = true
profile = {str(profile).lower()}
mesh_only = {str(c['mesh_only']).lower()}
metric = {c['metric']}
sampling = {c['sampling']}
distribution = {c['distribution']}
yc = 0.819
seed = 1
n = {c['n']}
ppc = {c['ppc']}
warmup = {warmup}
'''
for c in cases:(ROOT/'inputs'/f'{c["name"]}.athinput').write_text(deck(c))
(ROOT/'matrix.json').write_text(json.dumps(cases,indent=2)+'\n')
assert len(cases)==60
# Focused gates: complete 1M liveness; small source matching and disabled-control tests.
gates=[]
for c in cases:
  if c['name'] in ['count_st_dep1_N10000','count_flat_dep1_N10000','blocks_st_dep1_B16','ranks_st_dep1_R2','ranks_st_dep1_R4','dynamic_ppc6','sampling_central4_R1','sampling_central1_R1','sampling_block_R1','sampling_point_R1']:
    g=c.copy();g['name']='gate_'+c['name'];gates.append(g)
    (ROOT/'inputs'/f'{g["name"]}.athinput').write_text(deck(g,tlim=1,warmup=.25))
(ROOT/'gates.json').write_text(json.dumps(gates,indent=2)+'\n')
if __name__=='__main__':print(f'Authored {len(cases)} throughput cases and {len(gates)} gates')
