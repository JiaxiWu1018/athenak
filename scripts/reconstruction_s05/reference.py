"""Continuum source projected with the production finest-grid CIC kernel."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from generate import density
from common import ROOT, atomic, sha

HALF=16.; DX=.5; SIDE=64

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
    nodes,w=leggauss(order);nodes=(nodes+1)/2;w=w/2
    uv=np.stack(np.meshgrid(nodes,nodes,nodes,indexing='ij'),-1).reshape(-1,3)
    qw=np.prod(np.stack(np.meshgrid(w,w,w,indexing='ij'),-1),axis=-1).ravel()*DX**3
    mass=np.zeros(SIDE**3);energy=mass.copy();stress=mass.copy()
    for start in range(0,len(lo),512):
        pos=(lo[start:start+512,None,:]+DX*uv).reshape(-1,3)
        rr=np.linalg.norm(pos,axis=1);d=density(p,radial,rr)
        y=np.interp(rr,p['riso_over_m'],p['y']);a=np.interp(rr,p['riso_over_m'],p['conformal_a'])
        ww=1/np.sqrt(y);m=d*np.tile(qw,len(lo[start:start+512]))
        mass+=cloud(pos,m);energy+=cloud(pos,m*ww)
        stress+=cloud(pos,m*a*a*(ww*ww-1)/(3*ww))
    rc=np.linalg.norm(coordinates(),axis=1);ac=np.interp(rc,p['riso_over_m'],p['conformal_a'])
    outside=rc>radius;ac[outside]=(1+.5/rc[outside])**2
    return {'mass':mass,'E':energy/(ac**3*DX**3),'Sdiag':stress/(ac**3*DX**3),'r':rc}

def ensure(p):
    out=ROOT/'reference';out.mkdir(exist_ok=True)
    for radial in ('legacy','proper'):
        path=out/(radial+'.npz')
        if path.exists():continue
        before=project(p,radial,4);after=project(p,radial,8)
        rel=float(np.linalg.norm(after['E']-before['E'])/np.linalg.norm(after['E']))
        if rel>1e-3:
            before=after;after=project(p,radial,16)
            rel=float(np.linalg.norm(after['E']-before['E'])/np.linalg.norm(after['E']))
        if rel>1e-3:raise ValueError('Continuum CIC reference quadrature not converged')
        mass_error=float(abs(after['mass'].sum()/p['M0_over_M']-1))
        if mass_error>1e-4:raise ValueError('Continuum source normalization failed')
        np.savez_compressed(path,**after)
        atomic(out/(radial+'.json'),{'energy_relative_L2_quadrature_difference':rel,
               'mass_relative_error':mass_error,'sha256':sha(path),'cube_half_M':HALF,'dx_M':DX})
