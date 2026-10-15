"""Diagnose the genuine native initial-energy gate using a preserved full-N run."""
import json,os,time,traceback
from pathlib import Path
import numpy as np
import reduce_run as R
from common import ROOT,atomic,require_job,sample


def interpolated_energy(p,pos,u):
    """calc_energy.cpp: interpolate covariant metric/lapse, then invert metric."""
    scaled=pos/.5-.5;low=np.floor(scaled).astype(int);frac=scaled-low
    offsets=np.arange(8)-3
    weights=[]
    for axis in range(3):
        w=np.ones((len(pos),8))
        for i in range(8):
            for j in range(8):
                if i!=j:w[:,i]*=(frac[:,axis]-offsets[j])/(offsets[i]-offsets[j])
        weights.append(w)
    nodes=[(low[:,a,None]+offsets+.5)*.5 for a in range(3)]
    rr=np.sqrt(nodes[0][:,:,None,None]**2+nodes[1][:,None,:,None]**2+nodes[2][:,None,None,:]**2)
    a=np.interp(rr,p['riso_over_m'],p['conformal_a'])
    alpha=p['alpha_s']*np.sqrt(np.interp(rr,p['riso_over_m'],p['y']))
    outside=rr>p['Riso_over_M'];q=.5/rr[outside]
    a[outside]=(1+q)**2;alpha[outside]=(1-q)/(1+q)
    w=weights[0][:,:,None,None]*weights[1][:,None,:,None]*weights[2][:,None,None,:]
    g=np.sum(w*a*a,axis=(1,2,3));alp=np.sum(w*alpha,axis=(1,2,3))
    return alp*np.sqrt(1+np.sum(u*u,axis=1)/g)


def main():
    require_job();dest=ROOT/'qa_native_energy_463699';dest.mkdir(exist_ok=True)
    case='legacy_mc_s1';work=ROOT/'preflight_attempts/463699'/case
    path=sorted((work/'out/pvtk').glob('*.part.vtk'))[0]
    t,cycle,n,d=R.read_pvtk(str(path));assert t==0 and cycle==0 and n==1500000
    p=dict(np.load(ROOT/'samples/model.npz'));_,a=sample(ROOT/'samples'/(case+'.bin'))
    e=d['prtcl_energy'];target=float(p['alpha_s']);error=e-target
    indices=np.unique(np.r_[np.argpartition(np.abs(error),-512)[-512:],np.linspace(0,n-1,1024).astype(int)])
    tags=d['ptag'][indices].astype(int);pos=a[tags,1:4];u=a[tags,4:7]
    prediction=interpolated_energy(p,pos,u)
    r=np.linalg.norm(pos,axis=1);y=np.interp(r,p['riso_over_m'],p['y']);ac=np.interp(r,p['riso_over_m'],p['conformal_a'])
    analytic=p['alpha_s']*np.sqrt(y)*np.sqrt(1+np.sum(u*u,axis=1)/(ac*ac))
    log=R.parse_log(str(work/'solver.log'))
    result={'job':os.environ['SLURM_JOB_ID'],'source_run':str(work),'target_alpha_s':target,
       'native_energy_min':float(e.min()),'native_energy_max':float(e.max()),
       'native_error_max_abs':float(np.max(abs(error))),'native_error_rms':float(np.sqrt(np.mean(error**2))),
       'native_error_abs_quantiles':np.quantile(abs(error),[0,.5,.9,.99,.999,1]).tolist(),
       'analytic_prescription_max_abs_error':float(np.max(abs(analytic-target))),
       'native_vs_interpolated_covariant_metric_max_abs_error':float(np.max(abs(e[indices]-prediction))),
       'interpolated_prescription_max_abs_error':float(np.max(abs(prediction-target))),
       'sampled_particles':len(indices),'worst_particles':[{'tag':int(d['ptag'][i]),'energy':float(e[i]),'r':float(np.sqrt(sum(d[k][i]**2 for k in 'xyz')))} for i in np.argsort(abs(error))[-10:]],
       'boris_fallbacks':log['boris_fallbacks'],'boris_fallback_cycles':log['boris_fallback_cycles']}
    atomic(dest/'ENERGY_DIAGNOSTIC.json',result);print(json.dumps(result,indent=2),flush=True)
    from initial_energy import interpolated_energy as cached_energy,validate
    cached=cached_energy(p,pos,u)
    if not np.allclose(cached,prediction,rtol=0,atol=1e-12):
        raise ValueError('Cached-node metric interpolation differs from direct nodal evaluation')
    result['all_particles_energy_gate']=validate(p,a,d['ptag'],e)
    atomic(dest/'ENERGY_DIAGNOSTIC.json',result)
    from analyze import analyze
    begin=time.monotonic()
    try:analyze(case,run_root=ROOT/'preflight_attempts/463699',dest_root=dest/'reduced')
    except Exception:
        atomic(dest/'ANALYSIS_ERROR.txt',traceback.format_exc());raise
    atomic(dest/'NATIVE_REDUCTION_OK.json',{'seconds':time.monotonic()-begin})

if __name__=='__main__':main()
