"""Allocation-only storage, GPU and whole-session cost guards."""
import shutil
import time
import csv
import math
from common import *

def allocated_bytes(path):
    failures = []
    for attempt in range(5):
        try:
            r = subprocess.run(['du', '-s', '-B1', str(path)], capture_output=True, text=True, timeout=180)
            if r.returncode == 0:
                return int(r.stdout.split()[0])
            failures.append(r.stderr[-2000:])
        except Exception as error:
            failures.append(repr(error))
        time.sleep(.2*(attempt+1))
    raise RuntimeError('Storage checks unavailable: '+repr(failures))

def storage():
    own = allocated_bytes(EXT)
    whole = allocated_bytes('/work1/eliasmost/jiaxiwu')
    return dict(own=own, limit=350*GIB, checkpoint_reserve=48*GIB, whole=whole,
        forecast=whole+max(0, 350*GIB-own), whole_storage_warning=whole >= 1.5*TIB,
        filesystem_free=shutil.disk_usage(EXT).free)

def cost_seconds():
    """Use a bounded receipt; absent terminal evidence reserves the full allocation."""
    registry = EXT/'evidence/job_registry.json'
    rows = load(registry) if registry.exists() else []
    total = 0.
    for row in rows:
        end = EXT/'evidence/allocations'/str(row['id'])/'ALLOCATION_END.json'
        total += min(row['requested_seconds'], load(end)['elapsed_seconds']+60) if end.exists() else row['requested_seconds']
    return total

def admission():
    job()
    d = storage()
    d.update(job=job(), utc=now(), conservative_reserved_node_seconds=cost_seconds())
    assert d['own'] < d['limit']-d['checkpoint_reserve'], d
    assert d['whole'] < 1.7*TIB and d['forecast'] < 1.9*TIB, d
    assert d['filesystem_free'] > d['checkpoint_reserve'], d
    assert d['conservative_reserved_node_seconds'] <= 50*3600, d
    atomic(ROOT/'evidence/jobs'/job()/'admission.json', d)
    return d

def stop(reason):
    atomic(ROOT/'evidence/RESOURCE_STOP.json', dict(job=job(), utc=now(), reason=reason))
    active = ROOT/'evidence/active.json'
    if active.exists():
        a = load(active)
        w = Path(a['work'])
        if a['job'] == job() and ROOT in w.parents:
            (w/'REQUEST_STOP').write_text(reason+'\n')

def request_horizon_checkpoint(root=ROOT):
    """Checkpoint promptly after the first qualified AH, without changing evolution."""
    active=root/'evidence/active.json'
    marker=root/'evidence/AH_CHECKPOINT_REQUEST.json'
    if marker.exists() or not active.exists():return None
    a=load(active)
    work=Path(a['work'])
    if a['job']!=job() or a['phase']!='evolution' or root not in work.parents:return None
    path=work/(a['case']+'.ah_quality_0.csv')
    if not path.exists():return None
    with path.open() as f:
        header=f.readline()
        f.seek(max(f.tell(),path.stat().st_size-8192))
        lines=f.read().splitlines()
    if not lines:return None
    row=next(csv.DictReader([header,lines[-1]]))
    if any(v is None for v in row.values()) or float(row['accepted'])!=1:return None
    assert float(row['missing'])==0 and float(row['streak'])>=3
    assert float(row['hrel'])<=.05 and float(row['rmin'])>=4*float(row['dx'])
    assert all(math.isfinite(float(row[k])) and float(row[k])>0 for k in ['mass','mirr','area'])
    request=work/'REQUEST_STOP'
    if request.exists():return None  # A resource/user stop always takes precedence.
    event=dict(utc=now(),job=job(),accepted_native_time=float(row['time']),quality=row,
        source=str(path),scope='First qualified-AH clean-checkpoint request; actual checkpoint time recorded separately')
    atomic(marker,event)
    request.write_text('qualified_AH_checkpoint\n')
    return event

def watchdog():
    out = ROOT/'evidence/jobs'/job()
    out.mkdir(parents=True, exist_ok=True)
    last_disk = last_gpu = time.monotonic()
    previous = None
    next_disk = 0
    while True:
        t = time.monotonic()
        d = dict(job=job(), utc=now())
        if t >= next_disk:
            try:
                disk = storage()
                if previous:
                    elapsed = max(t-previous[0], 1)
                    disk['ten_minute_projected_whole'] = disk['whole']+max(0, disk['whole']-previous[1])*600/elapsed
                previous = (t, disk['whole'])
                d.update(disk)
                last_disk = t
                if disk['own'] >= disk['limit']-disk['checkpoint_reserve']:
                    stop('session_storage_checkpoint_reserve')
                if disk['whole'] >= 1.7*TIB or max(disk['forecast'], disk.get('ten_minute_projected_whole', 0)) >= 1.9*TIB:
                    stop('whole_AMD_storage_limit')
                if disk['filesystem_free'] < disk['checkpoint_reserve']:
                    stop('filesystem_checkpoint_reserve')
                if cost_seconds() > 50*3600:
                    stop('session_cost_limit')
            except Exception as error:
                d['storage_error'] = repr(error)
            next_disk = t+300
        if t-last_disk > 900:
            stop('storage_telemetry_stale')
        if os.environ.get('S10_CPU_ONLY') != '1':
            try:
                event=request_horizon_checkpoint()
                if event:d['horizon_checkpoint_request']=event
            except Exception as error:
                d['horizon_metadata_error']=repr(error)
                stop('AH_quality_metadata_error')
            try:
                raw = json.loads(subprocess.check_output(['rocm-smi', '--showmeminfo', 'vram', '--json'], text=True, timeout=30))
                ratios = []
                for row in raw.values():
                    if not isinstance(row, dict):
                        continue
                    total = [int(v) for k, v in row.items() if 'Total Memory' in k]
                    used = [int(v) for k, v in row.items() if 'Used Memory' in k]
                    if total and used:
                        ratios.append(used[0]/total[0])
                assert len(ratios) >= int(os.environ['SLURM_NTASKS'])
                d.update(assigned_GPU_memory=raw, vram_max_fraction=max(ratios), memory_warning=max(ratios) >= .8)
                last_gpu = t
                if max(ratios) >= .85:
                    stop('gpu_memory_85_percent')
            except Exception as error:
                d['memory_error'] = repr(error)
            if t-last_gpu > 300:
                stop('memory_telemetry_stale')
        atomic(out/'LATEST.json', d)
        with (out/'telemetry.jsonl').open('a') as f:
            f.write(json.dumps(d)+'\n')
        time.sleep(10)

if __name__ == '__main__':
    watchdog()
