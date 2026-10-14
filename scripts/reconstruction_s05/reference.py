"""Continuum source projected with the production finest-grid CIC kernel."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from generate import density
from common import ROOT, atomic, sha

HALF=16.; DX=.5; SIDE=64
METRIC_CACHE={}

def quadrature(order):
    # CIC has a derivative break at the cell centre; integrate its two halves separately.
    x,w=leggauss(order)
    return np.r_[(x+1)/4,(x+1)/4+.5],np.r_[w/4,w/4]

def interpolation_matrix(order,n):
    """The pusher's eight-point Lagrange stencil on the common dx=.5 grid."""
    nodes=quadrature(order)[0]
    x=(np.arange(-n,n)[:,None]+nodes).ravel()-.5
    low=np.floor(x).astype(int);index=low[:,None]-3+np.arange(8)
    weights=np.ones_like(index,dtype=float)
    for i in range(8):
        for j in range(8):
            if i!=j:weights[:,i]*=(x-index[:,j])/(index[:,i]-index[:,j])
    matrix=np.zeros((len(x),2*n+8))
    matrix[np.arange(len(x))[:,None],index+n+4]=weights
    return matrix

def tensor_interpolate(field,matrix):
    # Tensor separability avoids a 512-node loop at every quadrature point.
    first=np.einsum('ui,ijk->ujk',matrix,field,optimize=True)
    second=np.einsum('vj,ujk->uvk',matrix,first,optimize=True)
    return np.einsum('wk,uvk->uvw',matrix,second,optimize=True)

def metric_quadrature(p,order,n):
    key=(float(p['yc']),float(p['Riso_over_M']),order,n)
    if key not in METRIC_CACHE:
        x=(np.arange(-n-4,n+4)+.5)*DX
        rr=np.sqrt(x[:,None,None]**2+x[None,:,None]**2+x[None,None,:]**2)
        a=np.interp(rr,p['riso_over_m'],p['conformal_a'])
        outside=rr>p['Riso_over_M'];a[outside]=(1+.5/rr[outside])**2
        values=tensor_interpolate(1/(a*a),interpolation_matrix(order,n))
        q=2*order
        METRIC_CACHE[key]=values.reshape(2*n,q,2*n,q,2*n,q).transpose(0,2,4,1,3,5)
    return METRIC_CACHE[key]

def cloud(pos,weights,squared=False):
    scaled=(pos+HALF)/DX-.5; low=np.floor(scaled).astype(int);frac=scaled-low
    result=np.zeros(SIDE**3)
    for i in (0,1):
        for j in (0,1):
            for k in (0,1):
                ix=low+np.array([i,j,k])
                inside=np.all((ix>=0)&(ix<SIDE),axis=1)
                w=weights*np.prod(np.where(np.array([i,j,k]),frac,1-frac),axis=1)
                if squared:w=w*w
                ids=(ix[:,0]*SIDE+ix[:,1])*SIDE+ix[:,2]
                result+=np.bincount(ids[inside],weights=w[inside],minlength=SIDE**3)
    return result

def coordinates():
    x=-HALF+DX*(np.arange(SIDE)+.5)
    return np.stack(np.meshgrid(x,x,x,indexing='ij'),-1).reshape(-1,3)

def project(p,radial,order):
    radius=p['Riso_over_M']; n=int(np.ceil(radius/DX))
    lo=np.stack(np.meshgrid(np.arange(-n,n),np.arange(-n,n),np.arange(-n,n),indexing='ij'),-1).reshape(-1,3)*DX
    near=np.where(lo>0,lo,np.where(lo+DX<0,lo+DX,0))
    lo=lo[np.sum(near*near,axis=1)<radius**2]
    nodes,w=quadrature(order)
    uv=np.stack(np.meshgrid(nodes,nodes,nodes,indexing='ij'),-1).reshape(-1,3)
    qw=np.prod(np.stack(np.meshgrid(w,w,w,indexing='ij'),-1),axis=-1).ravel()*DX**3
    mass=np.zeros(SIDE**3);energy=mass.copy();stress=mass.copy()
    metric=metric_quadrature(p,order,n);max_lorentz_difference=0.
    for start in range(0,len(lo),512):
        pos=(lo[start:start+512,None,:]+DX*uv).reshape(-1,3)
        rr=np.linalg.norm(pos,axis=1);d=density(p,radial,rr)
        y=np.interp(rr,p['riso_over_m'],p['y']);a=np.interp(rr,p['riso_over_m'],p['conformal_a'])
        cell=np.rint(lo[start:start+512]/DX).astype(int)+n
        gi=metric[cell[:,0],cell[:,1],cell[:,2]].ravel()
        usq=a*a*np.maximum(1/y-1,0);ww=np.sqrt(1+gi*usq)
        supported=d>0
        max_lorentz_difference=max(max_lorentz_difference,float(np.max(np.abs(ww[supported]*np.sqrt(y[supported])-1),initial=0)))
        m=d*np.tile(qw,len(lo[start:start+512]))
        mass+=cloud(pos,m);energy+=cloud(pos,m*ww)
        stress+=cloud(pos,m*usq/(3*ww))
    rc=np.linalg.norm(coordinates(),axis=1);ac=np.interp(rc,p['riso_over_m'],p['conformal_a'])
    outside=rc>radius;ac[outside]=(1+.5/rc[outside])**2
    return {'mass':mass,'E':energy/(ac**3*DX**3),'Sdiag':stress/(ac**3*DX**3),'r':rc,
            'max_relative_lorentz_interpolation_difference':max_lorentz_difference}

def ensure(p):
    out=ROOT/'reference';out.mkdir(exist_ok=True)
    for radial in ('legacy','proper'):
        path=out/(radial+'.npz')
        receipt=out/(radial+'.json')
        if path.exists():
            import json
            if not receipt.exists():raise ValueError('Reference cache lacks a validation receipt: '+str(path))
            quality=json.loads(receipt.read_text())
            if quality.get('version')!=3 or quality['sha256']!=sha(path):
                raise ValueError('Reference cache version/hash differs: '+str(path))
            continue
        before=project(p,radial,2);after=project(p,radial,4)
        def differences(a,b):
            return {key:float(np.linalg.norm(b[key]-a[key])/np.linalg.norm(b[key])) for key in ('mass','E','Sdiag')}
        errors=differences(before,after)
        if max(errors.values())>1e-4:
            before=after;after=project(p,radial,8)
            errors=differences(before,after)
        if max(errors.values())>1e-4:raise ValueError('Continuum CIC reference quadrature not converged')
        mass_error=float(abs(after['mass'].sum()/p['M0_over_M']-1))
        if mass_error>1e-4:raise ValueError('Continuum source normalization failed')
        np.savez_compressed(path,**after)
        atomic(receipt,{'version':3,'energy_relative_L2_quadrature_difference':errors['E'],
               'relative_L2_quadrature_differences':errors,
               'mass_relative_error':mass_error,'sha256':sha(path),'cube_half_M':HALF,'dx_M':DX,
               'metric_interpolation':'eight-point Lagrange of nodal inverse metric, matching particles_tmunu.cpp',
               'quadrature':'Gauss-Legendre on each half cell; split at CIC derivative breaks; relative L2 tolerance1e-4',
               'max_relative_lorentz_interpolation_difference':after['max_relative_lorentz_interpolation_difference']})
