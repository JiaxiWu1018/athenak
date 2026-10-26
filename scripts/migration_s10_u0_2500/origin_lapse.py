"""Trilinear origin lapse from the eight central checkpoint cell centres."""
import struct,itertools
import numpy as np
from common import *
from checkpoint import parameters

def extract(path):
    job();path=Path(path)
    with path.open('rb') as f:
        text=b''
        while not text.endswith(b'<par_end>\n'):
            line=f.readline();assert line and len(text)<2**20;text+=line
        off=f.tell();par=parameters(text.decode());nmb,root_level=struct.unpack('<ii',f.read(8));f.seek(off+232);t,dt,cyc=struct.unpack('<ddi',f.read(20));f.seek(off+252);locs=np.fromfile(f,dtype='<i4',count=nmb*4).reshape(nmb,4);f.seek(off+252+nmb*20+8);per=struct.unpack('<Q',f.read(8))[0];grid=f.tell()
    assert per==25*40**3*8
    # Z4c enum: chi0, metric1..6, Khat7, A8..13, Gamma14..16, Theta17, alpha18.
    values=np.memmap(path,dtype='<f8',mode='r',offset=grid,shape=(nmb,25,40,40,40));points=[]
    for b,loc in enumerate(locs):
        level=int(loc[3])-root_level;lo=[];width=[]
        for dim in [1,2,3]:
            mn=float(par['mesh']['x'+str(dim)+'min']);mx=float(par['mesh']['x'+str(dim)+'max']);rootblocks=int(par['mesh']['nx'+str(dim)])//32;w=(mx-mn)/(rootblocks*2**level);lo.append(mn+int(loc[dim-1])*w);width.append(w)
        hi=np.array(lo)+width;lo=np.array(lo);width=np.array(width)
        if not (np.all(lo<=1e-12) and np.all(hi>=-1e-12)):continue
        centers=[lo[k]+(np.arange(32)+.5)*width[k]/32 for k in range(3)]
        # If a block crosses zero, retain both adjacent interior centres.
        nearby=[]
        for arr in centers:
            choices=[]
            for sign in [-1,1]:
                candidates=np.flatnonzero(arr*sign>0)
                if len(candidates):choices.append(int(candidates[np.argmin(abs(arr[candidates]))]))
            nearby.append(choices)
        for i,j,k in itertools.product(*nearby):points.append((float(centers[0][i]),float(centers[1][j]),float(centers[2][k]),float(values[b,18,k+4,j+4,i+4])))
    assert len(points)>=8,(path,t,len(points))
    axes=[]
    for d in range(3):
        coords=np.array([p[d] for p in points]);axes.append([float(coords[coords<0].max()),float(coords[coords>0].min())])
    corners=[];result=0.
    for signs in itertools.product([0,1],repeat=3):
        xyz=np.array([axes[d][signs[d]] for d in range(3)]);matches=[p for p in points if np.allclose(p[:3],xyz,rtol=0,atol=1e-10)];assert len(matches)==1,('Central AMR stencil not Cartesian; do not invent interpolation',path,t,xyz)
        value=matches[0][3];assert np.isfinite(value) and value>=0;weight=np.prod([(-axes[d][0] if signs[d] else axes[d][1])/(axes[d][1]-axes[d][0]) for d in range(3)]);result+=weight*value;corners.append(dict(xyz=xyz.tolist(),alpha=value,weight=float(weight)))
    return dict(time=t,cycle=cyc,central_lapse=float(result),source=receipt(path),corners=corners,definition='Trilinear interpolation at fixed coordinate origin from8adjacent interior cell-centres of native3D Z4c alpha(index18); checkpoint sampled, not a temporal fit.')
