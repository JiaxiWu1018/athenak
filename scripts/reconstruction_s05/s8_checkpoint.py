#!/usr/bin/env python3
"""Source-backed checkpoint header + streaming hash (allocated jobs for large files)."""
import argparse, hashlib, json, struct
from pathlib import Path

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def header(path):
    p=Path(path)
    with p.open('rb') as f: b=f.read(65536)
    marker=b.index(b'<par_end>'); off=marker+10
    nmb,root=struct.unpack_from('<ii',b,off)
    # restart.cpp: two ints, RegionSize(9 Real), root+block RegionIndcs(19 ints each).
    t,dt,cyc=struct.unpack_from('<ddi',b,off+232)
    # Session09 starts fresh: the native initial checkpoint has time=cycle=0.
    if not (0<=t<=10000 and 0<=dt<1 and 0<=cyc<100000000 and 0<nmb<100000):
        raise ValueError('Invalid source-backed checkpoint header')
    return {'path':str(p.resolve()),'bytes':p.stat().st_size,'time':t,'dt':dt,
            'cycle':cyc,'nmb':nmb,'root_level':root,'ascii_bytes':off},b[:off]

def payload(path, d, ascii_blob, ids_out=None, expected_ids=None):
    import numpy as np, re
    # Pinned restart.cpp layout for this particles+Z4c-only, no-puncture 3D model.
    text=ascii_blob.decode('ascii')
    if re.search(r'num_horizons\s*=\s*[1-9]',text): raise ValueError('Unexpected horizons')
    loc=d['ascii_bytes']+252+d['nmb']*20+8
    with open(path,'rb') as f:
        f.seek(loc); data_size=struct.unpack('<Q',f.read(8))[0]
        grid_start=loc+8; prtcl_base=grid_start+data_size*d['nmb']
        f.seek(prtcl_base); hdr=struct.unpack('<4d',f.read(32)); n=int(hdr[0])
        if hdr[0]!=n or n<=0 or any(x<0 or x!=int(x) for x in hdr[1:]):raise ValueError(hdr)
        expected_size=prtcl_base+32+n*10*8
        if d['bytes']!=expected_size:raise ValueError('Incomplete checkpoint: size mismatch')
        # All grid state and real particle variables must be finite.
        f.seek(grid_start); remain=data_size*d['nmb']+32+n*8*8
        while remain:
            take=min(remain,8<<20); buf=f.read(take)
            if len(buf)!=take or not np.isfinite(np.frombuffer(buf,dtype='<f8')).all():
                raise ValueError('Nonfinite or incomplete grid/particle payload')
            remain-=take
        ids=np.fromfile(f,dtype='<f8',count=2*n).reshape(2,n)
    tags=ids[1]
    if not np.isfinite(ids).all() or not np.equal(ids,np.floor(ids)).all() or len(np.unique(tags))!=n:
        raise ValueError('Invalid particle integer payload')
    tags=np.sort(tags.astype(np.int64))
    if expected_ids and not np.array_equal(tags,np.load(expected_ids)):raise ValueError('Particle ID set changed')
    if ids_out:np.save(ids_out,tags)
    d.update(data_size_per_mb=data_size,particle_count=n,removal_header=list(hdr[1:]),
             expected_bytes=expected_size,payload_verified=True)
    return d

def main():
    a=argparse.ArgumentParser();a.add_argument('path');a.add_argument('--hash',action='store_true')
    a.add_argument('--expect-time',type=float);a.add_argument('--expect-sha');a.add_argument('--expect-size',type=int)
    a.add_argument('--out');a.add_argument('--deck');a.add_argument('--payload',action='store_true')
    a.add_argument('--ids-out');a.add_argument('--expected-ids');x=a.parse_args();d,b=header(x.path)
    if x.expect_time is not None and abs(d['time']-x.expect_time)>1e-8:raise ValueError(d)
    if x.expect_size is not None and d['bytes']!=x.expect_size:raise ValueError(d)
    if x.hash or x.expect_sha:d['sha256']=sha256(x.path)
    if x.expect_sha and d['sha256']!=x.expect_sha:raise ValueError('SHA256 mismatch')
    if x.payload:payload(x.path,d,b,x.ids_out,x.expected_ids)
    if x.deck:Path(x.deck).write_bytes(b)
    if x.out:Path(x.out).write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps(d,indent=2))
if __name__=='__main__':main()
