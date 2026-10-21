"""Zero-step cross-backend restart identity, interiors, and particle validation."""
import struct,time
import numpy as np
from common import *
from checkpoint import validate
from run_segment import cohort

def layout(path):
    with Path(path).open('rb') as f:
        text=b''
        while not text.endswith(b'<par_end>\n'):text+=f.readline()
        off=f.tell();nmb=struct.unpack('<i',f.read(4))[0]
        f.seek(off+252+nmb*20+8);block_bytes=struct.unpack('<Q',f.read(8))[0]
        grid=f.tell();base=grid+nmb*block_bytes
        f.seek(base);n=int(struct.unpack('<d',f.read(8))[0])
    return grid,nmb,block_bytes,base+32,n

def main():
    jid=job();assert load(ROOT/'evidence/IMPORT_VERIFIED.json')['files']>0
    assert (ROOT/'evidence/NATIVE_CHECKS_OK.json').exists()
    source=ROOT/'runs/yc0500/rst/yc0500.00005.rst'
    before=validate(source)
    assert before['sha256']=='aaf731dbabb4ff3d676e519871d3caf8667cca508c978dd290f35000f1bac6fc'
    work=ROOT/'tests'/f'migration_{jid}';work.mkdir()
    command=native_launcher()+[str(ROOT/'protected/athena'),'--kokkos-map-device-id-by=mpi_rank',
      '-r',str(source),'-d',str(work),'time/nlim='+str(before['cycle']),
      'mesh_refinement/max_nmb_per_rank=192','problem/cluster_sample_file='+str(ROOT/'samples/yc0500.bin'),
      'problem/cluster_t0_dump='+str(work/'initial_particles.bin')]
    atomic(work/'launch.json',dict(job=jid,utc=now(),command=command,executable=receipt(ROOT/'protected/athena')))
    memory=[]
    with (work/'native.log').open('w') as f:
        process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
        while process.poll() is None:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
            rows=[x.split(',') for x in raw.splitlines() if x.strip()];assert len(rows)==3
            frac=max(float(x[1])/float(x[2]) for x in rows);memory.append(dict(utc=now(),fraction=frac,raw=raw))
            if frac>=.85:(work/'REQUEST_STOP').write_text('gpu_memory_85_percent\n')
            time.sleep(5)
    assert process.returncode==0,process.returncode
    atomic(work/'memory.json',memory)
    assert max(x['fraction'] for x in memory)<.8,'Insufficient production memory margin'
    destination=sorted((work/'rst').glob('*.rst'))[-1];after=validate(destination)
    for key in ['time','cycle','particle_count','initial_count','removal_counts','nblocks','ah_onset']:
        assert before[key]==after[key],key
    a=layout(source);b=layout(destination);assert a[1:3]==b[1:3] and a[4]==b[4]
    pa=np.memmap(source,dtype='<f8',mode='r',offset=a[3],shape=(10,a[4]))
    pb=np.memmap(destination,dtype='<f8',mode='r',offset=b[3],shape=(10,b[4]))
    ia=np.argsort(pa[9]);ib=np.argsort(pb[9]);assert np.array_equal(pa[9,ia],pb[9,ib])
    errors=[]
    for k in range(8):
        aa=pa[k,ia];bb=pb[k,ib]
        assert np.array_equal(aa,bb),('particle field changed without evolution',k)
        errors.append(float(np.max(abs(aa-bb))))
    ga=np.memmap(source,dtype='<f8',mode='r',offset=a[0],shape=(a[1],25,40,40,40))
    gb=np.memmap(destination,dtype='<f8',mode='r',offset=b[0],shape=(b[1],25,40,40,40))
    max_scaled=0.
    for block in range(a[1]):
        aa=ga[block,:,4:-4,4:-4,4:-4];bb=gb[block,:,4:-4,4:-4,4:-4]
        err=float(np.max(abs(aa-bb)/np.maximum(1.,abs(aa))));max_scaled=max(max_scaled,err)
    assert max_scaled<1e-12,max_scaled
    before.pop('parameters');after.pop('parameters')
    atomic(ROOT/'evidence/MIGRATION_RESTART_OK.json',dict(utc=now(),job=jid,source=before,destination=after,
      no_evolution=True,all_particle_fields_exact=True,particle_max_abs_errors=errors,
      interior_grid_max_scaled_difference=max_scaled,max_gpu_fraction=max(x['fraction'] for x in memory),
      ranks=3,max_nmb_per_rank=192,source_retained=True))
if __name__=='__main__':main()
