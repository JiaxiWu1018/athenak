"""Perseus storage and assigned-GPU interlocks, only inside allocations."""
import shutil,time
from common import *

def storage():
    storage_root=Path(os.environ.get('S8_STORAGE_ROOT',str(ROOT))).resolve()
    if 'S8_STORAGE_ROOT' in os.environ:
        assert storage_root==(LOCAL/'perseus').resolve()
        assert ROOT.resolve()==storage_root or storage_root in ROOT.resolve().parents
    # A single whole-tree du counts cross-root hard links once.
    own=int(subprocess.check_output(['du','-s','-B1',str(storage_root)],text=True,timeout=180).split()[0])
    # The AMD snapshot is conservative while its sole ST trajectory is held/stopped.
    permit=load(ROOT/'evidence/PRODUCTION_ADMISSION.json')
    amd=permit['amd_retained_bytes_upper_bound']
    free=shutil.disk_usage(ROOT).free
    return dict(utc=now(),storage_root=str(storage_root),work_root=str(ROOT),
                perseus_bytes=own,amd_reserved_bytes=amd,
                combined_reserved_bytes=own+amd,filesystem_free_bytes=free)

def admission():
    job()
    assert not (ROOT/'evidence/RESOURCE_STOP.json').exists()
    permit=load(ROOT/'evidence/PRODUCTION_ADMISSION.json')
    assert permit['approved'] and permit['job']==job()
    assert permit['combined_node_hour_upper_bound_with_this_job']<99
    # Reject concurrent AMD ST evolution; pending held jobs do not evolve.
    q=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
      'hpcfund.amd.com','squeue -h -u jiaxiwu -o "%i|%j|%T|%D"'],text=True,timeout=45)
    active=[line for line in q.splitlines() if '|st_s08_run|' in line and '|PENDING|' not in line]
    assert not active,active
    d=storage();d.update(job=job(),amd_queue=q,permit=permit)
    atomic(ROOT/'evidence/jobs'/job()/'admission.json',d)
    assert d['combined_reserved_bytes']<350*GIB,d
    assert d['filesystem_free_bytes']>2*TIB,d

def stop(reason):
    atomic(ROOT/'evidence/RESOURCE_STOP.json',dict(utc=now(),job=job(),reason=reason))
    p=ROOT/'evidence/active.json'
    if p.exists():
        active=load(p)
        work=Path(active['work'])
        if active['job']==job() and ROOT in work.parents:
            (work/'REQUEST_STOP').write_text(reason+'\n')

def main():
    jid=job();out=ROOT/'evidence/jobs'/jid;out.mkdir(parents=True,exist_ok=True)
    last_storage=last_memory=time.monotonic();next_storage=0
    while True:
        d=dict(utc=now(),job=jid)
        if time.monotonic()>=next_storage:
            try:
                s=storage();d.update(s);last_storage=time.monotonic()
                if s['combined_reserved_bytes']>=350*GIB:stop('combined_ST_storage_limit')
                if s['filesystem_free_bytes']<2*TIB:stop('perseus_free_space_reserve')
            except Exception as e:d['storage_error']=repr(e)
            next_storage=time.monotonic()+300
        if time.monotonic()-last_storage>900:stop('storage_telemetry_stale')
        try:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=30)
            rows=[r.split(',') for r in raw.splitlines() if r.strip()]
            assert len(rows)==int(os.environ['SLURM_NTASKS']),raw
            ratios=[float(r[1])/float(r[2]) for r in rows]
            d.update(gpu_memory_csv=raw,vram_max_fraction=max(ratios));last_memory=time.monotonic()
            if max(ratios)>=.8:d['memory_warning']=True
            if max(ratios)>=.85:stop('gpu_memory_85_percent')
        except Exception as e:d['memory_error']=repr(e)
        if time.monotonic()-last_memory>300:stop('memory_telemetry_stale')
        with (out/'telemetry.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
        atomic(out/'LATEST.json',d);time.sleep(5)
if __name__=='__main__':main()
