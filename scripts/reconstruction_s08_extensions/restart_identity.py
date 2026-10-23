"""Zero-step identity on the allocated GPU/rank topology."""
import struct
import numpy as np
from common import *
from checkpoint import validate

def layout(path):
    with Path(path).open('rb') as f:
        text=b''
        while not text.endswith(b'<par_end>\n'):text+=f.readline()
        off=f.tell();nmb=struct.unpack('<i',f.read(4))[0]
        f.seek(off+252+nmb*20+8);b=struct.unpack('<Q',f.read(8))[0]
        grid=f.tell();f.seek(grid+nmb*b);n=int(struct.unpack('<d',f.read(8))[0])
    return grid,nmb,b,grid+nmb*b+32,n

def gate(source):
    before=validate(source);ranks=int(os.environ['SLURM_NTASKS'])
    work=ROOT/'tests'/('restart_'+job());work.mkdir(parents=True)
    args=native_launcher()+[str(ROOT/'protected/athena'),'--kokkos-map-device-id-by=mpi_rank','-r',str(source),'-d',str(work),f'time/nlim={before["cycle"]}',f'mesh_refinement/max_nmb_per_rank={max(128,512//ranks)}','problem/cluster_sample_file='+str(ROOT/'samples/yc0500.bin'),'problem/cluster_t0_dump='+str(work/'initial_particles.bin')]
    atomic(work/'launch.json',dict(command=args,executable=receipt(ROOT/'protected/athena'),job=job()))
    with (work/'native.log').open('w') as out:cmd(args,stdout=out,stderr=subprocess.STDOUT)
    dest=sorted((work/'rst').glob('*.rst'))[-1];after=validate(dest)
    for key in ['time','cycle','particle_count','removal_counts','nblocks','ah_onset']:assert before[key]==after[key],key
    a=layout(source);b=layout(dest);assert a[1:3]==b[1:3] and a[4]==b[4]
    pa=np.memmap(source,dtype='<f8',mode='r',offset=a[3],shape=(10,a[4]));pb=np.memmap(dest,dtype='<f8',mode='r',offset=b[3],shape=(10,b[4]))
    ia=np.argsort(pa[9]);ib=np.argsort(pb[9]);assert np.array_equal(pa[9,ia],pb[9,ib])
    for k in range(8):
        if k==1:assert np.allclose(pa[k,ia],pb[k,ib],rtol=1e-12,atol=1e-12)
        else:assert np.array_equal(pa[k,ia],pb[k,ib]),k
    ga=np.memmap(source,dtype='<f8',mode='r',offset=a[0],shape=(a[1],25,40,40,40));gb=np.memmap(dest,dtype='<f8',mode='r',offset=b[0],shape=(b[1],25,40,40,40))
    err=max(float(np.max(abs(ga[k,:,4:-4,4:-4,4:-4]-gb[k,:,4:-4,4:-4,4:-4])/np.maximum(1,abs(ga[k,:,4:-4,4:-4,4:-4])))) for k in range(a[1]))
    assert err<1e-12,err
    before.pop('parameters');after.pop('parameters')
    record=dict(job=job(),utc=now(),case='yc0500',source=before,destination=after,physical_fields_exact=True,interior_grid_max_scaled_error=err,ranks=ranks,zero_steps=True)
    atomic(ROOT/'evidence/PRODUCTION_ADMISSION.json',record)
    return record
