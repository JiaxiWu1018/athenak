"""Full source-layout validation; run only inside a Slurm allocation."""
import argparse
import re
import struct
import numpy as np
from common import *

def parameters(text):
    result={};block=None
    for line in text.splitlines():
        line=line.split('#')[0].strip()
        if line.startswith('<'):block=line.strip('<>');result[block]={}
        elif '=' in line and block:
            k,v=line.split('=',1);result[block][k.strip()]=v.strip()
    return result

def validate(path,output=None):
    job();p=Path(path)
    with p.open('rb') as f:
        text=b''
        while not text.endswith(b'<par_end>\n'):
            line=f.readline()
            if not line or len(text)>2**20:raise ValueError('Missing parameter terminator')
            text+=line
        offset=f.tell();par=parameters(text.decode('ascii'))
        nmb,level=struct.unpack('<ii',f.read(8))
        assert 0<nmb<100000 and 0<=level<32
        f.seek(offset+232);t,dt,cycle=struct.unpack('<ddi',f.read(20))
        assert np.isfinite([t,dt]).all() and 0<=t<=float(os.environ.get("S8_MAX_TIME", str(max(c['target'] for c in ALL_CASES)+1))) and 0<=dt<1 and 0<=cycle<100000000
        assert int(par['fastflow'].get('npunct','0'))==0
        assert par['output8'].get('single_file_per_rank','false') in ['false','0']
        f.seek(offset+252+nmb*20+8)
        data_size=struct.unpack('<Q',f.read(8))[0];grid_start=f.tell()
        # This source writes 25 Z4c variables, full cells including four ghost layers.
        shape=[int(par['meshblock']['nx'+str(i)])+2*int(par['mesh']['nghost']) for i in (1,2,3)]
        assert data_size==int(np.prod(shape))*25*8,(data_size,shape)
        particle_base=grid_start+nmb*data_size
        f.seek(particle_base);hdr=np.array(struct.unpack('<4d',f.read(32)))
        assert np.isfinite(hdr).all() and (hdr>=0).all() and np.equal(hdr,np.floor(hdr)).all()
        n=int(hdr[0]);initial=int(par['problem']['cluster_n'])
        horizon=int(par['particles'].get('ndestroy_horizon_cum','0'))
        assert n+sum(hdr[1:])+horizon==initial
        assert p.stat().st_size==particle_base+32+n*80,'Truncated/extra checkpoint bytes'
        f.seek(grid_start);remain=nmb*data_size+32+n*80
        while remain:
            take=min(remain,8<<20);buf=f.read(take)
            assert len(buf)==take and np.isfinite(np.frombuffer(buf,dtype='<f8')).all(),'Nonfinite payload'
            remain-=take
        f.seek(particle_base+32);mass=np.fromfile(f,dtype='<f8',count=n)
        m0=float(par['problem']['production_initial_rest_mass']);weight=m0/initial
        assert n==0 or np.max(abs(mass/weight-1))<1e-10
        f.seek(particle_base+32+n*8*8);ids=np.fromfile(f,dtype='<f8',count=2*n).reshape(2,n)
        assert np.equal(ids,np.floor(ids)).all()
        tags=np.sort(ids[1].astype(np.int64))
        assert n==len(np.unique(tags)) and (tags>=0).all() and (tags<initial).all()
        assert (ids[0]>=0).all() and (ids[0]<nmb).all()
    d=dict(utc=now(),job=job(),**receipt(p),time=t,dt=dt,cycle=cycle,nblocks=nmb,
           particle_count=n,initial_count=initial,removal_counts=hdr[1:].astype(int).tolist()+[horizon],
           rest_mass_initial=m0,rest_mass_alive=float(mass.sum()),payload_verified=True,
           ah_onset=float(par['problem'].get('production_ah_onset','-1')),parameters=par)
    assert abs((d['rest_mass_alive']+(initial-n)*weight)/m0-1)<1e-10
    if output:atomic(output,d)
    return d

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('path');ap.add_argument('--out');a=ap.parse_args()
    print(json.dumps(validate(a.path,a.out),indent=2))
