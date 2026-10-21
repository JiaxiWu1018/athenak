#!/usr/bin/env python3
"""Session-local finite continuation, verification, telemetry and bounded retention.

Never submit beyond t=75. A failed/ambiguous launch or science diagnostic pause
requires review; it cannot silently start a new t=0 trajectory. Heavy operations
(hashing and population verification) require an allocated Slurm job.
"""
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import socket
import struct
import subprocess
import time
from pathlib import Path

END = 75.0
TIB = 1024**4
LAYOUTS = [(6, 256), (7, 240), (8, 224)]

def atomic(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    os.replace(tmp, path)

def require_compute():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Allocated Slurm compute is required')

def digest(path):
    require_compute()
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()

def checkpoint(path, hashing=True, expected_particles=5_000_000, ntrackers=2, block_n=40):
    """Native Real=double MPI single-file Z4c+particles schema; reject truncation."""
    path = Path(path)
    st = path.stat()
    with path.open('rb') as f:
        header = f.read(4*1024**2)
        end = header.find(b'<par_end>')
        if end < 0:
            raise ValueError('missing bounded parameter header')
        offset = end+10
        f.seek(offset)
        nmb, rootlevel = struct.unpack('<ii', f.read(8))
        if not 0 < nmb <= 100_000 or not 0 <= rootlevel < 20:
            raise ValueError('invalid mesh metadata')
        f.seek(offset+232)
        simtime, delta, cycle = struct.unpack('<ddi', f.read(20))
        if not all(math.isfinite(x) for x in (simtime, delta)) or not 0 <= simtime <= END+1e-10 or delta <= 0 or cycle < 0:
            raise ValueError('invalid checkpoint time/cycle')
        f.seek(offset+252+nmb*20+8+24*ntrackers)
        block_bytes = struct.unpack('<Q', f.read(8))[0]
        if block_bytes != 25*block_n**3*8:
            raise ValueError('unexpected Z4c block layout')
        particle_offset = f.tell()+block_bytes*nmb
        f.seek(particle_offset)
        counts = struct.unpack('<4d', f.read(32))
        if any(not math.isfinite(x) or x < 0 or x != int(x) for x in counts):
            raise ValueError('invalid particle census')
        if sum(counts) != expected_particles:
            raise ValueError('particle count closure failed')
        expected = particle_offset+32+int(counts[0])*10*8
        if st.st_size != expected:
            raise ValueError(f'truncated/incompatible checkpoint: {st.st_size} != {expected}')
    row = dict(path=str(path.resolve()), bytes=st.st_size, time=simtime, dt=delta,
               cycle=cycle, blocks=nmb, particles=int(counts[0]),
               removed=list(map(int, counts[1:])), particle_offset=particle_offset+32)
    if hashing:
        row['sha256'] = digest(path)
    if path.stat().st_size != st.st_size or path.stat().st_mtime_ns != st.st_mtime_ns:
        raise ValueError('checkpoint changed during verification')
    return row

def last_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open('rb') as f:
        header = f.readline().decode().strip().split(',')
        f.seek(max(0, path.stat().st_size-32768))
        lines = f.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        fields = line.split(',')
        if len(fields) == len(header) and fields[0].lstrip('-').isdigit():
            return dict(zip(header, fields))
    return None

def telemetry(run):
    require_compute()
    host = socket.gethostname()
    run = Path(run)
    while True:
        devices = []
        for p in sorted(Path('/sys/class/drm').glob('card[0-9]*/device')):
            try:
                total = int((p/'mem_info_vram_total').read_text())
                used = int((p/'mem_info_vram_used').read_text())
                devices.append(dict(device=str(p.resolve()), total=total, used=used))
            except (OSError, ValueError):
                continue
        # PCI addresses can have duplicate DRM aliases; count each physical device once.
        devices = list({v['device']: v for v in devices}.values())
        host_memory={}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key,value=line.split(':',1)
            if key in ('MemTotal','MemAvailable'):
                host_memory[key]=int(value.split()[0])*1024
        row = dict(epoch=time.time(), host=host, devices=devices,
                   job=os.environ['SLURM_JOB_ID'],host_memory=host_memory)
        atomic(run/f'telemetry_{host}.json', row)
        with (run/f'telemetry_{host}.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        time.sleep(5)

def request_stop(run, reason, kind):
    path = run/'STOP_REASON.json'
    if not path.exists():
        atomic(path, dict(epoch=time.time(), reason=reason, kind=kind))
    (run/'REQUEST_STOP').touch()

def persistent_dropout(status,event,horizon):
    # Failed searches before the first accepted AH are formation evidence, not dropout.
    return bool(int(status['ever_bits'])&(1<<horizon)) and event is not None and int(event['failed_searches'])>=128

def watch(root, run, nodes):
    require_compute()
    started=time.time(); last_du=0; usage=None; usage_time=0
    previous_gpu={}; reserve=2400; horizon_gaps={}; active_started=None
    while True:
        now=time.time()
        status=last_csv(run/'s11_amd.gi_status.csv')
        if status and int(status['cycle'])>0 and active_started is None:active_started=now
        telemetry_files=list(run.glob('telemetry_*.json'))
        healthy=0
        for path in telemetry_files:
            try:
                row=json.loads(path.read_text())
                if now-row['epoch']>60 or len(row['devices'])!=4:
                    continue
                healthy+=1
                hm=row.get('host_memory',{})
                if hm.get('MemTotal',0) and hm.get('MemAvailable',0)<.15*hm['MemTotal']:
                    request_stop(run,'host memory reserve below 15%','capacity')
                for device in row['devices']:
                    fraction=device['used']/device['total']
                    key=device['device']+row['host']
                    old=previous_gpu.get(key)
                    # Growth observed over the monitor interval predicts the next
                    # finalization reserve conservatively; abrupt jumps also stop.
                    growth=max(0,(fraction-old[1])/(now-old[0])) if old and now>old[0] else 0
                    if fraction>=.80:
                        with (run/'warnings.jsonl').open('a') as f:
                            f.write(json.dumps(dict(epoch=now,gpu_fraction=fraction,device=key))+'\n')
                    project_growth=active_started is not None and now-active_started>60 and fraction>=.78
                    if fraction>=.83 or (project_growth and fraction+growth*reserve>=.85):
                        request_stop(run,'GPU memory pressure before 85%','capacity')
                    previous_gpu[key]=(now,fraction)
            except (OSError,ValueError,KeyError):
                continue
        if now-started>600 and healthy<nodes:
            request_stop(run,'GPU telemetry missing or stale','review')
        if now-last_du>=300:
            last_du=now
            try:
                output=subprocess.check_output(['du','-sb','/work1/eliasmost/jiaxiwu'],timeout=180,text=True)
                new_usage=int(output.split()[0])
                growth=max(0,(new_usage-usage)/(now-usage_time)) if usage is not None else 0
                record=dict(epoch=now,bytes=new_usage,projected_bytes=new_usage+growth*reserve,
                            warning=new_usage>=1.5*TIB)
                with (run/'storage.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                usage=new_usage;usage_time=now
                # Reserve two maximum observed ~64GiB checkpoint writes as well as growth.
                if new_usage>=1.7*TIB or new_usage+max(growth*reserve,128*1024**3)>=1.9*TIB:
                    request_stop(run,'whole-user storage reserve exhausted','review')
            except (OSError,ValueError,subprocess.SubprocessError) as e:
                with (run/'warnings.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=now,du_error=str(e)))+'\n')
        if now-started>900 and now-usage_time>900:
            request_stop(run,'whole-user storage telemetry stale','review')
        status=last_csv(run/'s11_amd.gi_status.csv')
        if status:
            if float(status['time'])>15 and (int(status['ever_bits'])&3)!=3:
                request_stop(run,'formation stage passed without both accepted individual horizons','science')
            if int(status['max_rank_blocks'])>=.90*int(status['capacity']):
                request_stop(run,'per-rank block capacity pressure','capacity')
            common=status['strict_common']=='1'
            for h in ([2] if common else [0,1]):
                established=int(status['ever_bits'])&(1<<h)
                current=int(status['valid_bits'])&(1<<h)
                if established and not current:
                    horizon_gaps.setdefault(h,int(status['cycle']))
                    if int(status['cycle'])-horizon_gaps[h]>=128:
                        request_stop(run,f'no current accepted surface for horizon {h}','science')
                else:
                    horizon_gaps.pop(h,None)
                event=last_csv(run/f's11_amd.ah_events_{h}.csv')
                if persistent_dropout(status,event,h):
                    request_stop(run,f'persistent accepted-AH dropout horizon {h}','science')
        time.sleep(15)

def finish(root,run,rc):
    require_compute()
    if rc:
        atomic(root/'control/PAUSED.json',dict(reason='application failed; no automatic replay',returncode=rc,run=str(run)))
        return False
    candidates=[]
    for p in sorted((run/'rst').glob('*.rst')):
        candidates.append(checkpoint(p))
    if not candidates:
        raise RuntimeError('clean application exit without verified checkpoint')
    candidates.sort(key=lambda r:(r['cycle'],r['path']))
    latest=candidates[-1]
    previous_path=root/'control/latest_checkpoint.json'
    previous=json.loads(previous_path.read_text()) if previous_path.exists() else None
    if previous and latest['cycle']<=previous['cycle']:
        raise RuntimeError('segment made no trajectory progress')
    audit_population(root,run,candidates,latest)
    atomic(run/'checkpoint_manifest.json',candidates)
    atomic(previous_path,latest)
    # Retain every milestone, initialization, final, and the last two recovery states.
    history_path=root/'control/checkpoint_inventory.json'
    history=json.loads(history_path.read_text()) if history_path.exists() else []
    history += candidates
    milestone_numbers=set()
    event_file=run/'s11_amd.gi_milestones.csv'
    if event_file.exists():
        with event_file.open() as f:
            milestone_numbers={int(r['restart_number']) for r in csv.DictReader(f)}
    for r in candidates:
        number=int(Path(r['path']).stem.rsplit('.',1)[1])
        r['protected']=r['cycle']==0 or r['time']>=END-1e-10 or number in milestone_numbers
    recoveries=sorted([r for r in history if not r.get('deleted')],key=lambda r:(r['cycle'],r['path']))
    unique_cycles={r['cycle']:r for r in recoveries}
    keep={r['path'] for r in list(unique_cycles.values())[-2:]}
    atomic(history_path,history)  # inventory exists BEFORE any authorized rotation
    for r in history:
        if r.get('deleted') or r.get('protected') or r['path'] in keep:
            continue
        p=Path(r['path']).resolve()
        if root.resolve() not in p.parents or 'runs' not in p.parts:
            raise RuntimeError('refusing rotation outside this session run tree')
        # Both recovery copies are validated before deleting an older redundant state.
        if not all(Path(k).exists() for k in keep) or len(keep)<2:
            continue
        if digest(p)!=r['sha256']:
            raise RuntimeError('checkpoint changed before authorized rotation')
        p.unlink();r['deleted']=dt.datetime.now(dt.timezone.utc).isoformat()
    atomic(history_path,history)
    if latest['time']>=END-1e-10:
        atomic(root/'control/ENDPOINT_75.json',latest)
        from endpoint import submit_endpoint
        submit_endpoint(root)
        return False
    stop=run/'STOP_REASON.json'
    if stop.exists():
        reason=json.loads(stop.read_text())
        if reason['kind']=='capacity':
            layout=int(os.environ['LAYOUT_INDEX'])+1
            if layout<len(LAYOUTS):
                atomic(root/'control/layout.json',dict(index=layout,reason=reason))
                return True
        atomic(root/'control/PAUSED.json',reason)
        return False
    return True

def particle_diagnostic_quality(data):
    """IPEN is derived -u_t, not evolved state; preserve invalid energy as unknown.

    The pusher/deposition use mass, position and covariant momentum. Those and
    both ID fields must remain finite. No energy value is replaced or clipped.
    """
    import numpy as np
    for index in (0,2,3,4,5,6,7,8,9):
        if not np.isfinite(data[index]).all():
            raise RuntimeError(f'nonfinite native physical particle field {index}')
    invalid=~np.isfinite(data[1])
    components=[]
    for lo,hi in ((0,3000000),(3000000,4000000),(4000000,5000000)):
        selected=invalid & (data[9]>=lo) & (data[9]<hi)
        components.append(dict(unknown_count=int(selected.sum()),
                               unknown_rest_mass=float(data[0,selected].sum())))
    return dict(field='IPEN',definition='specific coordinate energy -u_t, high-order metric interpolation',
                unknown_count=int(invalid.sum()),components=components,
                policy='retain nonfinite diagnostic as unknown; reject any nonfinite physical field')

def audit_population(root,run,candidates,latest):
    """Exact all-tag alive/dead partition and per-tag rest-weight closure."""
    require_compute()
    import numpy as np
    def table(row):
        if row['particles']==0:return np.empty((10,0))
        return np.memmap(row['path'],dtype='<f8',mode='r',offset=row['particle_offset'],
                         shape=(10,row['particles']))
    def identities(row):
        data=table(row)
        particle_diagnostic_quality(data)
        raw=data[9];tags=raw.astype(np.int64)
        if np.any(raw!=tags) or np.any(tags<0) or np.any(tags>=5_000_000):
            raise RuntimeError('invalid exact particle IDs')
        counts=np.bincount(tags,minlength=5_000_000)
        if counts.max(initial=0)>1:raise RuntimeError('duplicate surviving particle ID')
        if np.any(data[0]<=0):raise RuntimeError('nonpositive particle rest weight')
        return data,tags,counts.astype(bool)
    weights_path=root/'control/initial_mass.npy'
    if not weights_path.exists():
        initial=next((r for r in candidates if r['cycle']==0),None)
        if initial is None:raise RuntimeError('missing mandatory initialization state')
        data,tags,alive=identities(initial)
        if not alive.all():raise RuntimeError('incomplete initial tag population')
        weights=np.empty(5_000_000);weights[tags]=data[0]
        with weights_path.open('wb') as f:np.save(f,weights)
        atomic(root/'control/initial_mass_manifest.json',dict(path=str(weights_path),
               sha256=digest(weights_path),source_checkpoint=initial))
    weights=np.load(weights_path,mmap_mode='r')
    dead_path=root/'control/dead_tags.npy'
    dead=np.load(dead_path) if dead_path.exists() else np.zeros(5_000_000,dtype=bool)
    death_file=run/'s11_amd.prtcl_destroy.csv'
    new_deaths=0
    if death_file.exists():
        with death_file.open() as f:
            for row in csv.DictReader(f):
                tag=int(row['tag']);mass=float(row['mass'])
                if not 0<=tag<5_000_000 or dead[tag]:raise RuntimeError('duplicate/invalid removed ID')
                if not math.isclose(mass,float(weights[tag]),rel_tol=1e-12,abs_tol=1e-16):
                    raise RuntimeError('removed rest-weight mismatch')
                dead[tag]=True;new_deaths+=1
    data,tags,alive=identities(latest)
    if np.any(alive&dead) or not np.all(alive|dead):
        raise RuntimeError('alive plus removed IDs do not partition initial population')
    if not np.allclose(data[0],weights[tags],rtol=1e-12,atol=1e-16):
        raise RuntimeError('surviving per-tag rest weights changed')
    if int(dead.sum())!=sum(latest['removed']):raise RuntimeError('native/death-log census differs')
    components=[]
    for start,end in ((0,3000000),(3000000,4000000),(4000000,5000000)):
        components.append(dict(initial=end-start,alive=int(alive[start:end].sum()),
          dead=int(dead[start:end].sum()),initial_mass=float(weights[start:end].sum()),
          alive_mass=float(weights[start:end][alive[start:end]].sum()),
          removed_mass=float(weights[start:end][dead[start:end]].sum())))
    profile_file=run/'s11_amd.gi_particles.csv'
    measured={}
    with profile_file.open() as f:
        for row in csv.DictReader(f):
            if int(row['cycle'])==latest['cycle'] and row['mask']=='0':
                measured[int(row['component']),int(row['shell'])]=row
    for c,(start,end) in enumerate(((0,3000000),(3000000,4000000),(4000000,5000000))):
        selected=[v for (component,shell),v in measured.items() if component==c]
        if sum(float(r['count']) for r in selected)!=components[c]['alive']:
            raise RuntimeError('online population count disagrees with native checkpoint')
        online_mass=sum(float(r['rest_mass']) for r in selected)
        if not math.isclose(online_mass,components[c]['alive_mass'],rel_tol=1e-9,abs_tol=1e-12):
            raise RuntimeError('online rest-mass profile disagrees with native checkpoint')
        member=(tags>=start)&(tags<end)
        native_j=float((data[0,member]*(data[2,member]*data[5,member]-data[4,member]*data[3,member])).sum())
        online_j=sum(float(r['Jz']) for r in selected)
        if not math.isclose(online_j,native_j,rel_tol=1e-9,abs_tol=1e-10):
            raise RuntimeError('online Jz profile disagrees with native checkpoint')
        components[c].update(native_Jz=native_j,online_Jz=online_j)
    # Publish state only after all closure checks pass; new segment is not submitted
    # if any check above fails. A failed audit retains its raw evidence for review.
    with dead_path.with_suffix('.tmp').open('wb') as f:np.save(f,dead)
    os.replace(dead_path.with_suffix('.tmp'),dead_path)
    atomic(run/'population_audit.json',dict(cycle=latest['cycle'],time=latest['time'],
           exact_partition=True,new_deaths=new_deaths,components=components,
           native_energy_quality=particle_diagnostic_quality(data)))

def submit(root):
    with (root/'control/submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for name in ('PAUSED.json','ENDPOINT_75.json'):
            if (root/'control'/name).exists():
                return
        if not (root/'control/VALIDATION_PASS.json').exists():
            raise RuntimeError('production validation gate has not been released')
        latest=root/'control/latest_checkpoint.json'
        if latest.exists() and json.loads(latest.read_text())['time']>=END-1e-10:
            return
        state_path=root/'control/submission.json'
        old=json.loads(state_path.read_text()) if state_path.exists() else None
        intent_path=root/'control/submission_intent.json'
        if intent_path.exists():
            intent=json.loads(intent_path.read_text())
            if intent.get('status')=='prepared':
                if old and old['segment']==intent['segment']:
                    atomic(intent_path,dict(intent,status='submitted',job_id=old['job_id']))
                else:
                    raise RuntimeError('ambiguous scheduler receipt; review intent before any further submission')
        if old:
            active=subprocess.check_output(['squeue','-h','-j',str(old['job_id']),'-o','%T'],text=True).strip()
            # The current completed segment may submit its successor while finalizing.
            if active and str(old['job_id'])!=os.environ.get('SLURM_JOB_ID'):
                return
            if not latest.exists():
                raise RuntimeError('fresh production already submitted; refusing another t=0 launch')
        layout_path=root/'control/layout.json'
        layout=json.loads(layout_path.read_text())['index'] if layout_path.exists() else 0
        nodes,cap=LAYOUTS[layout]; segment=old['segment']+1 if old else 0
        export=f'ALL,SESSION_ROOT={root},SEGMENT={segment},LAYOUT_INDEX={layout},BLOCK_CAPACITY={cap}'
        command=['sbatch','--parsable',f'--nodes={nodes}',f'--ntasks={nodes*4}',
                 '--ntasks-per-node=4',f'--export={export}',
                 f'--chdir={root}',
                 f'--output={root}/evidence/production_%j.slurm.log',str(root/'scripts/amd_segment.sbatch')]
        atomic(intent_path,dict(status='prepared',segment=segment,command=command,
                               epoch=time.time(),parent_job=os.environ.get('SLURM_JOB_ID')))
        output=subprocess.check_output(command,text=True).strip()
        job=output.splitlines()[-1].split(';')[0]
        if not job.isdigit():raise RuntimeError('unrecognized scheduler receipt: '+output)
        atomic(state_path,dict(job_id=job,segment=segment,nodes=nodes,capacity=cap,command=command))
        atomic(intent_path,dict(status='submitted',segment=segment,job_id=job,command=command))

def finish_and_submit(root,run,returncode):
    try:
        if finish(root,run,returncode):submit(root)
    except Exception as exc:
        atomic(root/'control/PAUSED.json',dict(kind='review',run=str(run),
               job=os.environ.get('SLURM_JOB_ID'),reason=f'{type(exc).__name__}: {exc}',
               epoch=time.time()))
        raise

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['telemetry','watch','finish','submit','checkpoint'])
    ap.add_argument('--root',type=Path,default=Path(os.environ.get('SESSION_ROOT','.')))
    ap.add_argument('--run',type=Path)
    ap.add_argument('--nodes',type=int)
    ap.add_argument('--returncode',type=int,default=0)
    ap.add_argument('--path',type=Path)
    a=ap.parse_args()
    if a.action=='telemetry':telemetry(a.run)
    elif a.action=='watch':watch(a.root,a.run,a.nodes)
    elif a.action=='checkpoint':print(json.dumps(checkpoint(a.path),indent=2))
    elif a.action=='submit':submit(a.root)
    else:finish_and_submit(a.root,a.run,a.returncode)

if __name__=='__main__':main()
