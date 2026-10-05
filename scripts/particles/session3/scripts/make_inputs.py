"""Lightweight deck generation from the independently validated reference clock."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REF=461.99570043659014
GAUGE='''lapse_harmonic = 0.0
lapse_harmonicf = 1.0
lapse_oplog = 2.0
lapse_advect = 1.0
shift_driver = legacy
shift_eta = 2.0
shift_advect = 1.0
shift_Gamma = 1.0
shift_alpha2Gamma = 0.0
shift_H = 0.0
const_damp = true
diss = 0.1
chi_div_floor = 0.00001
use_z4c = false
hdamp_cH = 0.02
hdamp_par_safety = 0.5
damp_kappa1 = 0.0
damp_kappa2 = 0.0
nrad_wave_extraction = 0
'''

def block(name,**values):
    return '<'+name+'>\n'+''.join(f'{k} = {v}\n' for k,v in values.items())+'\n'

def deck(name,live,cut=1000,n=2113536,levels=10,cfl=.25,periods=5,nlim=100000000):
    text='# Session 03 isotropic Part I; sampling cut is not an orbital boundary.\n'
    text+=block('job',basename=name)
    mesh=dict(nghost=4)
    for d in range(1,4):
        mesh.update({f'nx{d}':128,f'x{d}min':-20480,f'x{d}max':20480,
                     f'ix{d}_bc':'outflow',f'ox{d}_bc':'outflow'})
    text+=block('mesh',**mesh)+block('meshblock',nx1=32,nx2=32,nx3=32)
    text+=block('mesh_refinement',refinement='static')
    for l in range(1,levels+1):
        region=dict(level=l)
        for d in range(1,4):region.update({f'x{d}min':-20480/2**l,f'x{d}max':20480/2**l})
        text+=block(f'refined_region{l}',**region)
    text+=block('time',evolution='dynamic',integrator='rk4',cfl_number=cfl,
                nlim=nlim,tlim=format(periods*REF,'.17g'),ndiag=10)
    text+=('<z4c>\n'+GAUGE+'\n') if live else '<adm>\n\n'
    text+=block('particles',particle_type='dust',pusher='gr_boris',init='pgen',
      feedback='true' if live else 'false',mass=1,debug=0,destroy_log='true',
      excise_radius=0,excise_x1=0,excise_x2=0,excise_x3=0,excise_lapse=0,excise_ah='false',
      cross_level_deposit='conservative',tmunu_filter_passes=0)
    text+=block('problem',user_hist='true',plummer_model='isotropic',plummer_mass=1,
      plummer_isotropic_a=10,plummer_sampling_radius=cut,plummer_npair=n//2,plummer_seed=1985,
      plummer_shell_nbin=64,plummer_shell_rmin=.3125,plummer_shell_rmax=4000,
      plummer_cohort_nbin=32,plummer_lmax=4,plummer_field_nbin=80,
      plummer_field_rmin=.3125,plummer_field_rmax=30000,plummer_constraint_reference=0,
      plummer_pmom_nrad=2,plummer_pmom_ntheta=32,plummer_pmom_selftest=1,
      plummer_pmom_r1=9000,plummer_pmom_r2=10000)
    output=[dict(file_type='pvtk',variable='prtcl_all',dt=REF/100)]
    if live:
        # Native cell slices preserve the deposited cell averages. Rendering performs
        # area-overlap averaging onto fixed pixels; never interpolate CIC density.
        for variable in ('tmunu','con','z4c_alpha'):
            output.append(dict(file_type='bin',variable=variable,slice_x3=0,dt=REF/50))
    output.append(dict(file_type='cbin',variable='adm',coarsen_factor=2,dt=REF/10))
    # Output types are inserted at the front. Last non-restart declaration makes
    # health/history execute before every other output due at the same time.
    output.append(dict(file_type='hst',dt=REF/200,data_format='%.17e'))
    output.append(dict(file_type='rst',dt=REF))
    for i,params in enumerate(output,1):text+=block(f'output{i}',**params)
    return text

def main():
    validation=json.loads((ROOT/'evidence/profile_agreement.json').read_text())
    if not validation['passed']:raise RuntimeError('profile validation failed')
    matrix=[]
    for label,cut,n,levels,cfl,periods in [('main',1000,2113536,10,.25,5),
       ('tail',2000,2113536,10,.25,5),('coarse',1000,2113536,9,.125,2),
       ('lowN',1000,528384,10,.25,2)]:
        for live in (False,True):
            name=label+('_live' if live else '_frozen')
            (ROOT/'inputs'/f'{name}.athinput').write_text(deck(name,live,cut,n,levels,cfl,periods))
            matrix.append(dict(name=name,live=live,sampling_radius=cut,N=n,levels=levels,
                               expected_leaf_blocks=64+56*levels,CFL=cfl,periods=periods,
                               P_ref=REF,tlim=periods*REF,finest_dx=320/2**levels))
    for live in (False,True):
        for prefix,periods,nlim in [('preflight',5,100),('pilot',.30,100000000),
                                    ('split',.30,740),('short',.025,100000000),
                                    ('halfdt',.025,100000000)]:
            name=prefix+('_live' if live else '_frozen')
            text=deck(name,live,cfl=.125 if prefix=='halfdt' else .25,periods=periods,nlim=nlim)
            if prefix in ('pilot','split'):
                # Quarter-period checkpoint, plus a 0.05-period post-transient
                # interval to establish the constraint reference without t=0.
                text=text.replace('dt = '+format(REF,'.17g'),'dt = '+format(REF/4,'.17g'))
            (ROOT/'inputs'/f'{name}.athinput').write_text(text)
    (ROOT/'state/matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')
if __name__=='__main__':main()
