"""Initial monoenergy prescription and the pinned scalar metric-interpolation diagnostic."""
import numpy as np


def metric_nodes(p):
    n=int(np.ceil(float(p['Riso_over_M'])/.5))
    x=(np.arange(-n-4,n+4)+.5)*.5
    rr=np.sqrt(x[:,None,None]**2+x[None,:,None]**2+x[None,None,:]**2)
    a=np.interp(rr,p['riso_over_m'],p['conformal_a'])
    alpha=np.interp(rr,p['riso_over_m'],p['lapse'])
    exterior=rr>p['Riso_over_M'];q=.5/rr[exterior]
    a[exterior]=(1+q)**2;alpha[exterior]=(1-q)/(1+q)
    return n,a*a,alpha


def interpolated_energy(p,pos,u,chunk=4096):
    # calc_energy.cpp interpolates gamma_ij first, then inverts; deposition
    # instead interpolates node-wise gamma^ij. Keep the two operators distinct.
    n,gamma,lapse=metric_nodes(p);out=np.empty(len(pos));offset=np.arange(8)-3
    for first in range(0,len(pos),chunk):
        xyz=pos[first:first+chunk];v=u[first:first+chunk]
        scaled=xyz/.5-.5;low=np.floor(scaled).astype(int);frac=scaled-low
        index=low[:,:,None]+offset+n+4
        if index.min()<0 or index.max()>=len(gamma):raise ValueError('Initial-energy stencil leaves cached metric')
        w=np.ones((len(xyz),3,8))
        for i in range(8):
            for j in range(8):
                if i!=j:w[:,:,i]*=(frac-offset[j])/(offset[i]-offset[j])
        ix=index[:,0,:,None,None];iy=index[:,1,None,:,None];iz=index[:,2,None,None,:]
        weights=w[:,0,:,None,None]*w[:,1,None,:,None]*w[:,2,None,None,:]
        g=np.sum(weights*gamma[ix,iy,iz],axis=(1,2,3))
        alp=np.sum(weights*lapse[ix,iy,iz],axis=(1,2,3))
        out[first:first+len(xyz)]=alp*np.sqrt(1+np.sum(v*v,axis=1)/g)
    return out


def validate(p,initial,tags,native_energy):
    tags=np.asarray(tags,dtype=np.int64);pos=initial[tags,1:4];u=initial[tags,4:7]
    r=np.linalg.norm(pos,axis=1);y=np.interp(r,p['riso_over_m'],p['y']);a=np.interp(r,p['riso_over_m'],p['conformal_a'])
    target=float(p['alpha_s']);analytic=target*np.sqrt(y)*np.sqrt(1+np.sum(u*u,axis=1)/(a*a))
    prescription=float(np.max(abs(analytic-target)))
    if prescription>1e-12:raise ValueError('Analytic initial monoenergy prescription changed')
    prediction=interpolated_energy(p,pos,u)
    residual=float(np.max(abs(native_energy-prediction)))
    if residual>1e-7:raise ValueError('Native initial energy differs from the pinned metric interpolation: '+str(residual))
    return {'analytic_max_abs_error':prescription,'native_max_abs_error_from_alpha_s':float(np.max(abs(native_energy-target))),
            'expected_interpolation_max_abs_error_from_alpha_s':float(np.max(abs(prediction-target))),
            'native_vs_expected_max_abs_error':residual,'native_vs_expected_atol':1e-7,
            'definition':'Analytic monoenergy plus native scalar lapse/gamma interpolation then matrix inversion; float32 VTK residual'}
