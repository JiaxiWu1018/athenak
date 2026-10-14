"""Source-matched profiles, converged clipped-cell masses and 24 initial samples."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
import st_migration_lib as L
from common import ROOT, N, CASES, atomic, sha, require_job

def model():
    p = L.cluster_profile(.819, dx=1e-4)
    p['legacy'] = L.equilibrium_cdf(p)
    p['proper'] = p['v0']/p['v0'][-1]
    x, y, r = p['x'], p['y'], p['riso_over_m']
    ratio = np.maximum(1/y-1, 0)/(1/.819-1)
    rho0 = np.sqrt(.819)*(y/.819)**-1*np.sqrt(ratio)
    radial = np.ones_like(x)
    radial[1:] = 1/np.sqrt(1-2*p['zeta']*p['v'][1:]/x[1:])
    xr = np.empty_like(x); xr[1:] = x[1:]/r[1:]; xr[0] = xr[1]
    legacy_w = x*x/y*np.sqrt(np.maximum(1/y-1, 0))
    norm = np.trapz(legacy_w, x)
    p['D_proper'] = p['M0_over_M']*xr**3*rho0/(4*np.pi*p['v0'][-1])
    p['D_legacy'] = p['M0_over_M']*xr**3/y*np.sqrt(np.maximum(1/y-1, 0))/(4*np.pi*radial*norm)
    return p

def density(p, radial, r):
    return np.interp(r, p['riso_over_m'], p['D_'+radial], right=0.)

def octant_cells(radius):
    n = int(np.ceil(radius/.5))
    ijk = np.stack(np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij'), -1).reshape(-1, 3)
    lo = .5*ijk
    lo = lo[np.sum(lo*lo, axis=1) < radius*radius]
    return lo, lo+.5

def integrate(p, radial, lo, hi, order):
    """Positive octant cubature, with exact spherical clipping of each z interval."""
    nodes, weights = leggauss(order); u=(nodes+1)/2; w=weights/2
    radius=p['Riso_over_M']; out=[]
    # Bound temporary point arrays, independently of quadrature order.
    chunk=max(1, 1500000//order**3)
    for start in range(0, len(lo), chunk):
        a,b=lo[start:start+chunk],hi[start:start+chunk]
        xhi=np.minimum(b[:,0],np.sqrt(np.maximum(radius**2-a[:,1]**2-a[:,2]**2,0)))
        lx=xhi-a[:,0]; x=a[:,0,None]+lx[:,None]*u
        yhi=np.minimum(b[:,1,None],np.sqrt(np.maximum(radius**2-x*x-a[:,2,None]**2,0)))
        ly=np.maximum(yhi-a[:,1,None],0); y=a[:,1,None,None]+ly[:,:,None]*u
        zhi=np.minimum(b[:,2,None,None],np.sqrt(np.maximum(radius**2-x[:,:,None]**2-y*y,0)))
        lz=np.maximum(zhi-a[:,2,None,None],0)
        z=a[:,2,None,None,None]+lz[:,:,:,None]*u
        r=np.sqrt(x[:,:,None,None]**2+y[:,:,:,None]**2+z*z)
        dz=np.sum(density(p,radial,r)*w,axis=3)*lz
        dy=np.sum(dz*w,axis=2)*ly
        out.extend(np.sum(dy*w,axis=1)*lx)
    return np.asarray(out)

def cells(p, radial):
    path=ROOT/'samples'/('cells_'+radial+'.npz')
    if path.exists(): return dict(np.load(path))
    lo,hi=octant_cells(p['Riso_over_M'])
    mass=integrate(p,radial,lo,hi,8); error=np.full(len(lo),np.inf)
    active=np.arange(len(lo)); history=[]
    for order in (16,32,64,128):
        new=integrate(p,radial,lo[active],hi[active],order)
        error[active]=np.abs(new-mass[active]); mass[active]=new
        keep=error[active] > 2e-5*new+1e-11*p['M0_over_M']
        history.append({'order':order,'evaluated_cells':len(active),'unconverged_cells':int(keep.sum()),
                        'estimated_global_relative_error':float(8*error.sum()/p['M0_over_M'])})
        print(radial,history[-1],flush=True)
        active=active[keep]
        if len(active)==0: break
    if len(active) or not np.all(mass>0): raise ValueError('Clipped cell cubature did not converge')
    total=8*mass.sum(); rel=abs(total/p['M0_over_M']-1)
    if rel>1e-5: raise ValueError('Cell integral disagrees with independent radial mass: '+str(rel))
    # Octant replication makes integration weights identical under reflections.
    signs=np.asarray([[sx,sy,sz] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)])
    low=np.minimum(lo[:,None,:]*signs,hi[:,None,:]*signs).reshape(-1,3)
    high=np.maximum(lo[:,None,:]*signs,hi[:,None,:]*signs).reshape(-1,3)
    cm=np.repeat(mass*p['M0_over_M']/total,8)
    np.savez_compressed(path,lo=low,hi=high,mass=cm)
    atomic(ROOT/'evidence'/('cell_integral_'+radial+'.json'),
           {'cell_count':len(cm),'unnormalized_total':float(total),'relative_mass_difference':float(rel),
            'global_normalization':float(p['M0_over_M']/total),'history':history,'sha256':sha(path)})
    return {'lo':low,'hi':high,'mass':cm}

def lhs(n, dim, seed):
    rng=np.random.Generator(np.random.PCG64(seed)); out=np.empty((n,dim))
    for d in range(dim): out[:,d]=(rng.permutation(n)+rng.random(n))/n
    return out

def finish_positions(p, radial, uniforms):
    r=np.interp(uniforms[:,0],p[radial],p['riso_over_m'])
    mu=2*uniforms[:,1]-1; phi=2*np.pi*uniforms[:,2]
    st=np.sqrt(np.maximum(1-mu*mu,0))
    return r[:,None]*np.column_stack((st*np.cos(phi),st*np.sin(phi),mu))

def cell_positions(p, radial, seed, table):
    rng=np.random.Generator(np.random.PCG64(seed+5000))
    nc=len(table['mass']); counts=np.full(nc,N//nc,dtype=np.int64)
    groups=nc//8; remainder=N%nc
    assert remainder%8==0
    selected=rng.permutation(groups)[:remainder//8]
    counts[(selected[:,None]*8+np.arange(8)).ravel()]+=1
    cell_id=np.repeat(np.arange(nc),counts)
    a,b=table['lo'][cell_id],table['hi'][cell_id]
    rmin=np.sqrt(np.sum(np.where(a>0,a,np.where(b<0,b,0))**2,axis=1))
    near=np.where(a>0,a,np.where(b<0,b,0))
    for axis in range(3):
        other=np.sum(near*near,axis=1)-near[:,axis]**2
        cap=np.sqrt(np.maximum(p['Riso_over_M']**2-other,0))
        a[:,axis]=np.maximum(a[:,axis],-cap)
        b[:,axis]=np.minimum(b[:,axis],cap)
    # Profile monotonicity is checked before using the nearest-radius rejection bound.
    d=p['D_'+radial]
    if np.max(np.diff(d))>1e-10*d.max(): raise ValueError('Density is not monotone')
    bound=density(p,radial,rmin)*(1+1e-12)
    pos=np.empty((N,3)); pending=np.arange(N); draws=0
    while len(pending):
        trial=a[pending]+(b[pending]-a[pending])*rng.random((len(pending),3))
        rr=np.linalg.norm(trial,axis=1); dd=density(p,radial,rr)
        if np.any(dd>bound[pending]*(1+1e-10)): raise ValueError('Rejection bound failed')
        accept=(rr<p['Riso_over_M']) & (rng.random(len(pending))*bound[pending]<dd)
        pos[pending[accept]]=trial[accept]; draws+=len(pending); pending=pending[~accept]
        if draws>100*N: raise ValueError('Cell rejection sampler stalled')
    mass=table['mass'][cell_id]/counts[cell_id]
    return pos,mass,{'cell_count':nc,'count_min':int(counts.min()),'count_max':int(counts.max()),
                    'proposal_draws':draws,'counts_sha256':__import__('hashlib').sha256(counts.tobytes()).hexdigest()}

def generate_one(p, radial, method, seed, table=None, n=N):
    # Stream separation ensures the ablation cannot change positions.
    extra={}
    if method in ('stratified','ablation'):
        points=lhs(n,5,seed+1000)
        pos=finish_positions(p,radial,points[:,:3])
        vel=points[:,3:] if method=='stratified' else np.random.Generator(np.random.PCG64(seed+2000)).random((n,2))
        mass=np.full(n,p['M0_over_M']/n)
    elif method=='cell':
        if n!=N: raise ValueError('Cell sample uses the fixed production budget')
        pos,mass,extra=cell_positions(p,radial,seed,table)
        vel=np.random.Generator(np.random.PCG64(seed+2000)).random((n,2))
    else:
        points=np.random.Generator(np.random.PCG64(seed+3000)).random((n,5))
        pos=finish_positions(p,radial,points[:,:3]); vel=points[:,3:]
        mass=np.full(n,p['M0_over_M']/n)
    rr=np.linalg.norm(pos,axis=1)
    y=np.interp(rr,p['riso_over_m'],p['y']); a=np.interp(rr,p['riso_over_m'],p['conformal_a'])
    umag=a*np.sqrt(np.maximum(1/y-1,0)); mu=2*vel[:,0]-1; phi=2*np.pi*vel[:,1]
    st=np.sqrt(np.maximum(1-mu*mu,0))
    u=umag[:,None]*np.column_stack((st*np.cos(phi),st*np.sin(phi),mu))
    data=np.column_stack((np.arange(n),pos,u,mass))
    if not np.isfinite(data).all() or not np.all(mass>0) or abs(mass.sum()/p['M0_over_M']-1)>1e-12:
        raise ValueError('Invalid sample')
    sampling={'mc':'random','stratified':'stratified','cell':'cell','ablation':'stratified'}[method]
    velocity='stratified' if method=='stratified' else 'random'
    case=f'{radial}_{method}_s{seed}'; path=ROOT/'samples'/(case+'.bin')
    if path.exists(): raise FileExistsError(path)
    with open(path,'wb') as f:
        f.write(f'STSAMP05 {n} {seed} 1 .819 .0001 {sampling} {radial} {velocity}\n'.encode())
        data.astype('<f8').tofile(f)
    com=np.average(pos,weights=mass,axis=0); unit=u/umag[:,None]
    mm=(unit*mass[:,None]).T@unit/mass.sum()
    atomic(ROOT/'evidence/samples'/(case+'.json'),
           dict(case=case,sha256=sha(path),bytes=path.stat().st_size,n=n,
                rest_mass=float(mass.sum()),mass_min=float(mass.min()),mass_max=float(mass.max()),
                global_neff=float(mass.sum()**2/np.sum(mass*mass)),com=com.tolist(),
                velocity_first_moment=np.average(unit,weights=mass,axis=0).tolist(),
                velocity_second_moment=mm.tolist(),radial=radial,method=method,seed=seed,**extra))
    return data

def main():
    require_job(); ap=argparse.ArgumentParser(); ap.add_argument('--profiles-only',action='store_true'); args=ap.parse_args()
    (ROOT/'samples').mkdir(exist_ok=True); p=model()
    np.savez_compressed(ROOT/'samples/model.npz',**p)
    atomic(ROOT/'evidence/model.json',{k:float(p[k]) for k in ('M0_over_M','Riso_over_M','R_over_M','alpha_c','alpha_s')})
    tables={rad:cells(p,rad) for rad in ('legacy','proper')}
    if args.profiles_only:return
    for case,rad,method,seed in CASES:
        if (ROOT/'samples'/(case+'.bin')).exists():continue
        t=time.monotonic(); generate_one(p,rad,method,seed,tables[rad]); print(case,time.monotonic()-t,flush=True)
    for rad in ('legacy','proper'):
        for seed in (1,2,3):
            from common import sample
            _,s=sample(ROOT/'samples'/f'{rad}_stratified_s{seed}.bin')
            _,a=sample(ROOT/'samples'/f'{rad}_ablation_s{seed}.bin')
            if not np.array_equal(s[:,1:4],a[:,1:4]) or not np.array_equal(s[:,7],a[:,7]):
                raise ValueError('Ablation changed positions/masses')
    atomic(ROOT/'evidence/SAMPLES_OK',{'cases':24,'job':os.environ['SLURM_JOB_ID']})

if __name__=='__main__':
    import os
    main()
