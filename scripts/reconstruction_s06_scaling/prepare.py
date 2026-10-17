"""Allocated AMD sample construction and pre-evolution mass/energy/COM checks."""
import os
import subprocess
import numpy as np
from cfg import *
import generate as G
from initial_energy import metric_nodes
from os_gate import queue

def initial_transport(p, pos, momentum):
    """Same initial lapse/gamma Lagrange interpolant and transport formula as GR-Boris.

    Initial shift=0, gamma_ij=gamma delta_ij. All particles start in the dx=.5 region.
    """
    n, gamma, lapse = metric_nodes(p)
    offset = np.arange(8)-3
    result = np.empty_like(momentum)
    for start in range(0, len(pos), 2048):
        xyz=pos[start:start+2048];u=momentum[start:start+2048]
        scaled=xyz/.5-.5;low=np.floor(scaled).astype(int);frac=scaled-low
        index=low[:,:,None]+offset+n+4
        assert index.min()>=0 and index.max()<len(gamma)
        w=np.ones((len(xyz),3,8))
        for i in range(8):
            for j in range(8):
                if i!=j:w[:,:,i]*=(frac-offset[j])/(offset[i]-offset[j])
        ix=index[:,0,:,None,None];iy=index[:,1,None,:,None];iz=index[:,2,None,None,:]
        weights=w[:,0,:,None,None]*w[:,1,None,:,None]*w[:,2,None,None,:]
        g=np.sum(weights*gamma[ix,iy,iz],axis=(1,2,3))
        alp=np.sum(weights*lapse[ix,iy,iz],axis=(1,2,3))
        assert np.all(g>0) and np.all(alp>0)
        result[start:start+len(xyz)]=u*(alp/g/np.sqrt(1+np.sum(u*u,axis=1)/g))[:,None]
    return result

def ordinary(p,n):
    points=G.lhs(n,5,1001)
    pos=G.finish_positions(p,'proper',points[:,:3])
    r=np.linalg.norm(pos,axis=1)
    y=np.interp(r,p['riso_over_m'],p['y']);a=np.interp(r,p['riso_over_m'],p['conformal_a'])
    mu=2*points[:,3]-1;phi=2*np.pi*points[:,4];st=np.sqrt(np.maximum(1-mu*mu,0))
    u=(a*np.sqrt(np.maximum(1/y-1,0)))[:,None]*np.column_stack((st*np.cos(phi),st*np.sin(phi),mu))
    return np.column_stack((np.arange(n),pos,u,np.full(n,float(p['M0_over_M'])/n)))

def main():
    require_job()
    assert not queue(local=True), 'OS jobs still have priority'
    assert load(ROOT/'evidence/OS_PRIORITY_RELEASE.json')['verified_complete']
    assert sha(ROOT/'protected/athena')==EXE_SHA
    parent_path=ROOT/'samples/parent_1p5m.bin'
    assert sha(parent_path)==PARENT_SHA
    _,parent=sample(parent_path)
    assert len(parent)==1500000 and np.array_equal(parent[:,0],np.arange(1500000))
    p=dict(np.load(ROOT/'samples/model.npz'))
    for case in CASES:
        path=ROOT/'samples'/(case['id']+'.bin')
        assert not path.exists(), 'Preserve prior generation; inspect its receipt before any retry'
        if case['paired']:
            data=np.concatenate((np.asarray(parent).copy(),np.asarray(parent).copy()))
            data[:,0]=np.arange(case['n']);data[:,7]*=.5
            data[1500000:,1:7]*=-1
            assert np.array_equal(data[1500000:,1:7],-data[:1500000,1:7])
            assert np.array_equal(data[:1500000,1:7],parent[:,1:7])
        else:
            data=ordinary(p,case['n'])
        assert data.shape==(case['n'],8) and np.isfinite(data).all()
        mass=data[:,7];pos=data[:,1:4];u=data[:,4:7]
        assert np.all(mass>0) and abs(mass.sum()/p['M0_over_M']-1)<1e-12
        r=np.linalg.norm(pos,axis=1);y=np.interp(r,p['riso_over_m'],p['y']);a=np.interp(r,p['riso_over_m'],p['conformal_a'])
        energy=p['alpha_s']*np.sqrt(y)*np.sqrt(1+np.sum(u*u,axis=1)/(a*a))
        assert np.max(abs(energy-p['alpha_s']))<1e-12
        com=np.average(pos,axis=0,weights=mass)
        momentum=np.average(u,axis=0,weights=mass)
        velocity=initial_transport(p,pos,u)
        bulk=np.average(velocity,axis=0,weights=mass)
        pair_v_error=None
        if case['paired']:
            pair_v_error=float(np.max(np.linalg.norm(velocity[:1500000]+velocity[1500000:],axis=1)))
            assert max(np.linalg.norm(com),np.linalg.norm(momentum),np.linalg.norm(bulk),pair_v_error)<1e-12
        write_sample(path,data)
        atomic(ROOT/'evidence/samples'/(case['id']+'.json'),dict(case,utc=now(),job=os.environ['SLURM_JOB_ID'],
            bytes=path.stat().st_size,sha256=sha(path),parent_sha256=PARENT_SHA if case['paired'] else None,
            seed=1,lhs_rng_seed=1001,radial='proper',sampling_family='antipodal_from_stratified' if case['paired'] else 'randomized_Latin_hypercube',
            total_rest_mass=float(mass.sum()),mass_per_particle=float(mass[0]),
            initial_com_M=com.tolist(),initial_mean_covariant_momentum=momentum.tolist(),
            initial_mean_coordinate_velocity=bulk.tolist(),max_pair_native_velocity_defect=pair_v_error,
            analytic_max_monoenergy_error=float(np.max(abs(energy-p['alpha_s']))),numpy=np.__version__,
            pair_mapping='partner(i)=i+1500000; masses halved; original x/u unchanged' if case['paired'] else None))
        print('SAMPLE_OK',case['id'],'COM',com,'bulk_v',bulk,flush=True)
        del data,velocity
    atomic(ROOT/'evidence/SAMPLES_OK.json',dict(utc=now(),job=os.environ['SLURM_JOB_ID'],source=source(),cases=CASES))

if __name__=='__main__':
    main()
