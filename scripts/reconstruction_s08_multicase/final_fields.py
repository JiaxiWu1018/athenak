"""Export and validate final 3D fields from a closed checkpoint, without evolution."""
import argparse
import re
import time
import numpy as np
from common import *
from checkpoint import validate


def check_binary(path,checkpoint):
    """Stream every native 3D block; layout matches vis/python/bin_convert.py."""
    with path.open('rb') as stream:
        assert stream.readline().strip()==b'Athena binary output version=1.1'
        count=int(stream.readline().split(b'=')[-1]);header={}
        for _ in range(count-1):
            key,value=stream.readline().decode().strip().split('=',1)
            header[key.strip()]=value.strip()
        assert abs(float(header['time'])-checkpoint['time'])<1e-8
        assert int(header['cycle'])==checkpoint['cycle']
        loc=int(header['size of location']);var=int(header['size of variable'])
        assert loc in [4,8] and var in [4,8]
        nvars=int(stream.readline().split(b'=')[-1])
        names=stream.readline().decode().split()[1:];assert len(names)==nvars>0
        size=int(stream.readline().split(b'=')[-1]);assert len(stream.read(size))==size
        blocks=0
        while stream.tell()<path.stat().st_size:
            raw=stream.read(24);assert len(raw)==24
            index=np.frombuffer(raw,dtype='<i4');dims=index[1::2]-index[::2]+1
            assert np.array_equal(dims,[32,32,32]),'Expected full 3D blocks'
            assert len(stream.read(16))==16
            geometry=np.frombuffer(stream.read(6*loc),dtype='<f'+str(loc))
            assert len(geometry)==6 and np.isfinite(geometry).all()
            values=np.fromfile(stream,dtype='<f'+str(var),count=32**3*nvars)
            assert len(values)==32**3*nvars and np.isfinite(values).all()
            blocks+=1
        assert blocks==checkpoint['nblocks'] and stream.tell()==path.stat().st_size
    return dict(**receipt(path),time=checkpoint['time'],cycle=checkpoint['cycle'],
                nblocks=blocks,variables=names,full_payload_finite=True)


def export(case):
    jid=job();work=ROOT/'runs'/case;state=load(work/'state.json')
    assert state['complete'] and state['latest']['payload_verified']
    marker=work/'FINAL_FIELDS_OK.json'
    if marker.exists():
        previous=load(marker)
        assert previous['checkpoint']['sha256']==state['latest']['sha256']
        assert all(Path(x['path']).stat().st_size==x['bytes'] for x in previous['files'])
        return previous
    if os.environ.get('S8_MACHINE')=='perseus':
        from extension_resources import admission,storage
        admission()
        assert storage()['own']+2*GIB<storage()['limit']
    checkpoint=validate(state['latest']['path']);checkpoint.pop('parameters')
    assert checkpoint['sha256']==state['latest']['sha256']
    destination=work/'final_fields'/f'{jid}_{time.time_ns()}';destination.mkdir(parents=True)
    command=native_launcher()+[str(ROOT/'protected/athena'),
      '--kokkos-map-device-id-by=mpi_rank','-r',checkpoint['path'],'-d',str(destination),
      'time/nlim='+str(checkpoint['cycle']),'time/tlim='+str(checkpoint['time']),
      'problem/production_diagnostics=false','problem/user_hist=false',
      'problem/cluster_sample_file='+str(ROOT/'samples'/f'{case}.bin'),
      'problem/cluster_t0_dump='+str(destination/'initial_particles.bin'),
      'output6/dt=1e30','output7/dt=1e30']
    command += [f'output{i}/dt=0' for i in [1,2,3,4,5,8]]
    ranks=int(os.environ['SLURM_NTASKS'])
    per=max(128,32*((int(np.ceil(checkpoint['nblocks']/ranks))+31)//32))
    assert per<=256, 'Final-field export needs a larger validated GPU layout'
    command += [f'mesh_refinement/max_nmb_per_rank={per}']
    executable=receipt(ROOT/'protected/athena')
    assert executable['sha256']==state['segments'][-1]['executable']['sha256']
    atomic(destination/'launch.json',dict(utc=now(),job=jid,command=command,
           checkpoint=checkpoint,executable=executable,no_evolution_requested=True))
    atomic(ROOT/'evidence/active.json',dict(job=jid,work=str(destination),case=case,
           phase='final_field_export'))
    with (destination/'native.log').open('w') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    assert result.returncode==0
    log=(destination/'native.log').read_text()
    assert '### FATAL' not in log and not re.search(r'search_fail=[1-9]',log)
    files=sorted((destination/'bin').glob('*.bin'))
    assert len(files)==2 and any('.con3d.' in p.name for p in files)
    assert any('.tme3d.' in p.name for p in files)
    checked=[check_binary(path,checkpoint) for path in files]
    assert sha(checkpoint['path'])==checkpoint['sha256'],'Input checkpoint changed'
    record=dict(utc=now(),job=jid,case=case,checkpoint=checkpoint,
      executable=executable,command=command,files=checked,no_evolution=True,
      evidence='Both full 3D native outputs have the exact source cycle/time; nlim equals source cycle. Input checkpoint bytes are unchanged.',
      scope='Final diagnostic fields only; no replacement of trajectory checkpoint, AH history or particle data.',
      source=receipt(Path(__file__)),native_log=receipt(destination/'native.log'))
    atomic(marker,record)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('case',choices=[c['id'] for c in CASES])
    print(json.dumps(export(parser.parse_args().case),indent=2))
