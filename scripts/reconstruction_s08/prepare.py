"""Generate five source-matched proper-profile LHS samples and exact decks on AMD."""
import sys
import re
import numpy as np
from common import *
sys.path.insert(0,str(ROOT/'src/athenak/scripts/reconstruction_s05'))
import st_migration_lib as profile

def section(text,name,entries):
    pattern=r'(?ms)^<'+re.escape(name)+r'>\n(.*?)(?=^<|\Z)'
    match=re.search(pattern,text)
    body=match.group(1) if match else ''
    for k,v in entries.items():
        body=re.sub(r'(?m)^'+re.escape(k)+r'\s*=.*\n?','',body)
        body+=f'{k} = {v}\n'
    block=f'<{name}>\n'+body.rstrip()+'\n\n'
    return text[:match.start()]+block+text[match.end():] if match else text+'\n'+block

def make_sample(case,n,path):
    p=profile.cluster_profile(case['yc'],dx=1e-4)
    rng=np.random.Generator(np.random.PCG64(1))
    u=np.column_stack([(rng.permutation(n)+rng.random(n))/n for _ in range(5)])
    cdf=p['v0']/p['v0'][-1]
    r=np.interp(u[:,0],cdf,p['riso_over_m'])
    mu=2*u[:,1]-1;phi=2*np.pi*u[:,2];st=np.sqrt(1-mu*mu)
    x=r[:,None]*np.column_stack((st*np.cos(phi),st*np.sin(phi),mu))
    y=np.interp(r,p['riso_over_m'],p['y']);a=np.interp(r,p['riso_over_m'],p['conformal_a'])
    mag=a*np.sqrt(np.maximum(1/y-1,0));mu=2*u[:,3]-1;phi=2*np.pi*u[:,4];st=np.sqrt(1-mu*mu)
    vel=mag[:,None]*np.column_stack((st*np.cos(phi),st*np.sin(phi),mu))
    data=np.column_stack((np.arange(n),x,vel,np.full(n,p['M0_over_M']/n)))
    energy=p['alpha_s']*np.sqrt(y)*np.sqrt(1+(vel*vel).sum(1)/(a*a))
    assert np.isfinite(data).all() and np.max(abs(energy-p['alpha_s']))<1e-12
    assert abs(data[:,7].sum()/p['M0_over_M']-1)<1e-12
    for d in range(5):assert np.array_equal(np.sort(np.floor(n*u[:,d]).astype(int)),np.arange(n))
    assert not path.exists()
    with path.open('wb') as f:
        f.write(f"STSAMP05 {n} 1 1 {case['yc']:.3f} .0001 stratified proper stratified\n".encode())
        data.astype('<f8').tofile(f)
    np.savez_compressed(path.with_suffix('.profile.npz'),**p)
    meta=dict(case=case,n=n,job=job(),rest_mass=float(p['M0_over_M']),Riso=float(p['Riso_over_M']),
              R=float(p['R_over_M']),alpha_c=float(p['alpha_c']),alpha_s=float(p['alpha_s']),
              com=x.mean(0).tolist(),max_energy_error=float(np.max(abs(energy-p['alpha_s']))),
              exact_lhs_strata_pass=True,**receipt(path))
    atomic(ROOT/'evidence/samples'/f'{path.stem}.json',meta)
    return meta

def deck(case,n=2000000,test=False):
    text=(ROOT/'src/athenak/scripts/reconstruction_s08/template.athinput').read_text()
    text=re.sub(r'(?ms)^<refined_region\d+>\n.*?(?=^<|\Z)','',text)
    ident=case['id'] if not test else 'frozen_ah'
    text=section(text,'comment',dict(problem='ST session 8 production' if not test else 'Frozen Schwarzschild AH check'))
    text=section(text,'job',dict(basename=ident))
    L=case['L'] if not test else 8
    root=128 if not test else 64
    text=section(text,'mesh',{**{f'nx{i}':root for i in [1,2,3]},**{f'x{i}min':-L for i in [1,2,3]},**{f'x{i}max':L for i in [1,2,3]}})
    text=section(text,'meshblock',{f'nx{i}':32 if not test else 16 for i in [1,2,3]})
    text=section(text,'mesh_refinement',dict(refinement='adaptive' if not test else 'static',num_levels=case['num_levels'],max_nmb_per_rank=128,ncycle_check=16,refinement_interval=16,log_amr_events='true',capacity_stop_fraction=.9))
    levels=range(1,6) if not test else range(1,3)
    amr=dict(method='loehner',loehner_variable='alp_psi7',loehner_threshold=.2)
    for l in levels:
        half=L/2**l
        text=section(text,f'refined_region{l}',{'level':l,**{f'x{i}min':-half for i in [1,2,3]},**{f'x{i}max':half for i in [1,2,3]}})
        amr[f'radius_{l-1}_rad']=half;amr[f'radius_{l-1}_reflevel']=l
    text=section(text,'z4c_amr',amr)
    text=section(text,'amr_criterion1',dict(method='user'))
    text=section(text,'time',dict(tlim=300 if not test else 0,nlim=100000000 if not test else 0,ndiag=16,stop_file='REQUEST_STOP',stop_check_every=1))
    text=section(text,'particles',dict(excise_lapse=.1 if case['unstable'] and not test else 0,excise_ah='false',destroy_log='true'))
    opts=dict(cluster_n=n,cluster_yc=case['yc'],cluster_sample_file=ROOT/'samples'/f'{ident}.bin',
      cluster_sampling='stratified',cluster_radial_measure='proper',cluster_velocity_sampling='stratified',
      production_diagnostics='true',production_history_dt=.25 if case['unstable'] else .5,
      production_cohort_dt=.5,production_cohort_stride=100)
    if test:opts['production_frozen_ah_check']='true'
    text=section(text,'problem',opts)
    ff=dict(num_horizons=1 if case['unstable'] or test else 0)
    if ff['num_horizons']:
        ff.update(npunct=0,ntheta=10,lmax=6,use_puncture_0=-1,initial_radius_0=.5,center_x_0=0,center_y_0=0,center_z_0=0,
                  expand_guess=1,flow_iterations_0=200,flow_alpha_beta_const_0=1,mass_tol_0=.01,start_time_0=0,
                  stop_time_0=1e30,verbose='false',production_quality_0='true',search_dt_0=.5,
                  quality_hrel_0=.05,quality_cells_0=4)
    text=section(text,'fastflow',ff)
    for i in [1,2,3,4,5,6,7,8]:
        dt=(.25 if case['unstable'] else .5) if i==1 else (1 if case['unstable'] else 2)
        if i in [6,7]:dt=1e30
        if i==8:dt=50
        text=section(text,f'output{i}',dict(dt=dt))
    # Full metric lives in milestone/recovery checkpoints, avoiding redundant 3D Z4c files.
    return text

def main():
    job();(ROOT/'samples').mkdir(exist_ok=True);(ROOT/'inputs').mkdir(exist_ok=True)
    for c in CASES:
        meta=make_sample(c,2000000,ROOT/'samples'/f"{c['id']}.bin")
        (ROOT/'inputs'/f"{c['id']}.athinput").write_text(deck(c))
        print(c['id'],meta['Riso'],meta['rest_mass'],flush=True)
    make_sample(CASES[0],2000,ROOT/'samples/frozen_ah.bin')
    (ROOT/'inputs/frozen_ah.athinput').write_text(deck(CASES[0],2000,True))
    atomic(ROOT/'evidence/SAMPLES_READY.json',dict(utc=now(),job=job(),cases=CASES))
if __name__=='__main__':main()
