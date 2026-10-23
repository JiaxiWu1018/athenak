"""Per-platform storage reservations and assigned-device in-job guard."""
import time
import shutil
from common import *

def storage():
    scope=LOCAL if MACHINE=='perseus' else EXT
    own=int(subprocess.check_output(['du','-s','-B1',str(scope)],text=True,timeout=180).split()[0])
    policy=(LOCAL if MACHINE=='perseus' else EXT)/'evidence/RESOURCE_RESERVATIONS.json'
    if policy.exists():
        reservation=load(policy)
        assert sum(reservation['platform_GiB'].values())==reservation['combined_review_GiB']<=350
        limit=reservation['platform_GiB'][MACHINE]*GIB
        reserve=reservation['checkpoint_reserve_GiB']*GIB
    else:
        limit=(120 if MACHINE=='perseus' else 80)*GIB;reserve=16*GIB
    data=dict(own=own,limit=limit,checkpoint_reserve=reserve,policy=str(policy),storage_root=str(scope),filesystem_free=shutil.disk_usage(scope).free)
    if MACHINE=='amd':
        whole=int(subprocess.check_output(['du','-s','-B1','/work1/eliasmost/jiaxiwu'],text=True,timeout=180).split()[0])
        data.update(whole=whole,forecast=whole+max(0,limit-own),whole_storage_warning=whole>=1.5*TIB)
    return data

def admission():
    job();data=storage();data.update(job=job(),utc=now(),machine=MACHINE)
    assert data['own']<data['limit']-data['checkpoint_reserve'],data
    if MACHINE=='amd':assert data['whole']<1.7*TIB and data['forecast']<1.9*TIB,data
    else:assert data['filesystem_free']>2*TIB,data
    atomic(ROOT/'evidence/jobs'/job()/'admission.json',data)
    return data

def stop(reason):
    atomic(ROOT/'evidence/RESOURCE_STOP.json',dict(job=job(),utc=now(),reason=reason))
    path=ROOT/'evidence/active.json'
    if path.exists():
        active=load(path);work=Path(active['work'])
        if active['job']==job() and ROOT in work.parents:(work/'REQUEST_STOP').write_text(reason+'\n')

def watchdog():
    out=ROOT/'evidence/jobs'/job();out.mkdir(parents=True,exist_ok=True)
    last_disk=last_gpu=time.monotonic();next_disk=0
    while True:
        t=time.monotonic();data=dict(job=job(),utc=now())
        if t>=next_disk:
            try:
                disk=storage();data.update(disk);last_disk=t
                if disk['own']>=disk['limit']-disk['checkpoint_reserve']:stop('batch_storage_checkpoint_reserve')
                if MACHINE=='amd' and (disk['whole']>=1.7*TIB or disk['forecast']>=1.9*TIB):stop('whole_AMD_storage_limit')
                if MACHINE=='perseus' and disk['filesystem_free']<2*TIB:stop('Perseus_free_space_reserve')
            except Exception as error:data['storage_error']=repr(error)
            next_disk=t+300
        if t-last_disk>900:stop('storage_telemetry_stale')
        try:
            if MACHINE=='perseus':
                # Perseus device cgroups renumber the assigned devices; NVML
                # sees precisely CUDA_VISIBLE_DEVICES, not physical Slurm IDs.
                ids=os.environ.get('CUDA_VISIBLE_DEVICES')
                assert ids,'Missing assigned-device list'
                raw=subprocess.check_output(['nvidia-smi','-i',ids,'--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=30)
                rows=[r.split(',') for r in raw.splitlines() if r.strip()]
                assert len(rows)==int(os.environ['SLURM_NTASKS']),raw
                ratios=[float(r[1])/float(r[2]) for r in rows];data['assigned_GPU_memory']=raw
            else:
                raw=json.loads(subprocess.check_output(['rocm-smi','--showmeminfo','vram','--json'],text=True,timeout=30));ratios=[]
                for row in raw.values():
                    if not isinstance(row,dict):continue
                    total=[int(v) for k,v in row.items() if 'Total Memory' in k];used=[int(v) for k,v in row.items() if 'Used Memory' in k]
                    if total and used:ratios.append(used[0]/total[0])
                data['assigned_GPU_memory']=raw
            assert ratios;data['vram_max_fraction']=max(ratios);data['memory_warning']=max(ratios)>=.8;last_gpu=t
            if max(ratios)>=.85:stop('gpu_memory_85_percent')
        except Exception as error:data['memory_error']=repr(error)
        if t-last_gpu>300:stop('memory_telemetry_stale')
        atomic(out/'LATEST.json',data)
        with (out/'telemetry.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
        time.sleep(10)

if __name__=='__main__':watchdog()
