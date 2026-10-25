"""Session-wide quota and assigned GPU guards, only in compute allocations."""
import time, shutil, sys
from common import *
def allocated_bytes(path):
    for attempt in range(5):
        r=subprocess.run(['du','-s','-B1',str(path)],capture_output=True,text=True,timeout=180)
        if r.returncode==0:return int(r.stdout.split()[0])
        errors=[v for v in r.stderr.splitlines() if v.strip()]
        if r.returncode!=1 or not errors or not all('No such file or directory' in v for v in errors):raise RuntimeError(r.stderr)
        time.sleep(.2*(attempt+1))
    raise RuntimeError('Storage inventory remained transiently unavailable')
def storage():
    own=allocated_bytes(EXT);whole=allocated_bytes('/work1/eliasmost/jiaxiwu')
    return dict(own=own,limit=400*GIB,checkpoint_reserve=48*GIB,whole=whole,forecast=whole+max(0,400*GIB-own),whole_storage_warning=whole>=1.5*TIB,filesystem_free=shutil.disk_usage(EXT).free)
def admission():
    job();d=storage();d.update(job=job(),utc=now(),machine=MACHINE)
    assert d['own']<d['limit']-d['checkpoint_reserve'] and d['whole']<1.7*TIB and d['forecast']<1.9*TIB,d
    assert d['filesystem_free']>d['checkpoint_reserve'],d
    atomic(ROOT/'evidence/jobs'/job()/'admission.json',d);return d
def stop(reason):
    atomic(ROOT/'evidence/RESOURCE_STOP.json',dict(job=job(),utc=now(),reason=reason))
    f=ROOT/'evidence/active.json'
    if f.exists():
        a=load(f);w=Path(a['work'])
        if a['job']==job() and ROOT in w.parents:(w/'REQUEST_STOP').write_text(reason+'\n')
def watchdog():
    out=ROOT/'evidence/jobs'/job();out.mkdir(parents=True,exist_ok=True)
    last_disk=last_gpu=time.monotonic();next_disk=0
    while True:
        t=time.monotonic();d=dict(job=job(),utc=now())
        if t>=next_disk:
            try:
                disk=storage();d.update(disk);last_disk=t
                if disk['own']>=disk['limit']-disk['checkpoint_reserve']:stop('session_storage_checkpoint_reserve')
                if disk['whole']>=1.7*TIB or disk['forecast']>=1.9*TIB:stop('whole_AMD_storage_limit')
                if disk['filesystem_free']<disk['checkpoint_reserve']:stop('filesystem_checkpoint_reserve')
            except Exception as e:d['storage_error']=repr(e)
            next_disk=t+300
        if t-last_disk>900:stop('storage_telemetry_stale')
        try:
            raw=json.loads(subprocess.check_output(['rocm-smi','--showmeminfo','vram','--json'],text=True,timeout=30));ratios=[]
            for row in raw.values():
                if not isinstance(row,dict):continue
                total=[int(v) for k,v in row.items() if 'Total Memory' in k];used=[int(v) for k,v in row.items() if 'Used Memory' in k]
                if total and used:ratios.append(used[0]/total[0])
            assert len(ratios)>=int(os.environ['SLURM_NTASKS'])
            d.update(assigned_GPU_memory=raw,vram_max_fraction=max(ratios),memory_warning=max(ratios)>=.8);last_gpu=t
            if max(ratios)>=.85:stop('gpu_memory_85_percent')
        except Exception as e:d['memory_error']=repr(e)
        if t-last_gpu>300:stop('memory_telemetry_stale')
        atomic(out/'LATEST.json',d)
        with (out/'telemetry.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
        time.sleep(10)
def cpu_watchdog(once=False):
    out=ROOT/'evidence/jobs'/job();out.mkdir(parents=True,exist_ok=True)
    last_disk=time.monotonic()
    while True:
        started=time.monotonic();d=dict(job=job(),utc=now(),phase='CPU support storage watchdog')
        try:
            disk=storage();d.update(disk);last_disk=time.monotonic()
            if disk['own']>=disk['limit']-disk['checkpoint_reserve']:stop('session_storage_checkpoint_reserve')
            if disk['whole']>=1.7*TIB or disk['forecast']>=1.9*TIB:stop('whole_AMD_storage_limit')
            if disk['filesystem_free']<disk['checkpoint_reserve']:stop('filesystem_checkpoint_reserve')
        except Exception as error:d['storage_error']=repr(error)
        if time.monotonic()-last_disk>900:stop('storage_telemetry_stale')
        atomic(out/'CPU_STORAGE_LATEST.json',d)
        with (out/'cpu_storage_telemetry.jsonl').open('a') as stream:stream.write(json.dumps(d)+'\n')
        if once:return d
        time.sleep(max(1,300-(time.monotonic()-started)))

if __name__=='__main__':
    if '--cpu' in sys.argv:cpu_watchdog()
    else:watchdog()
