#!/usr/bin/env python3
"""Session 009 checkpoint verification, retention, telemetry and finite monitoring."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import time

ROOT = Path('/work1/eliasmost/jiaxiwu/gi_s009_amd_20261005')

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()

def checkpoint(path, hash_file=True):
    """Exact native double/MPI single-file layout of the frozen GI source."""
    path = Path(path)
    st = path.stat()
    with path.open('rb') as f:
        header = f.read(1024**2)
        end = header.find(b'<par_end>')
        if end < 0: raise ValueError('missing bounded parameter header')
        offset = end + 10
        f.seek(offset)
        nmb, rootlevel = struct.unpack('<ii', f.read(8))
        if not 0 < nmb <= 11520 or rootlevel != 3:
            raise ValueError('unexpected mesh size/root level')
        f.seek(offset + 232)
        simtime, dt, cycle = struct.unpack('<ddi', f.read(20))
        if not all(math.isfinite(v) for v in (simtime,dt)) or simtime < 0 or dt <= 0 or cycle < 0:
            raise ValueError('invalid checkpoint time/cycle')
        f.seek(offset + 252 + nmb*20 + 56)
        block_bytes = struct.unpack('<Q', f.read(8))[0]
        if block_bytes != 25*40**3*8:
            raise ValueError('unexpected Z4c block layout')
        particle_offset = f.tell() + block_bytes*nmb
        f.seek(particle_offset)
        counts = struct.unpack('<4d', f.read(32))
        if any(not math.isfinite(x) or x < 0 or x != int(x) for x in counts):
            raise ValueError('invalid particle census')
        if sum(counts) != 5_000_000: raise ValueError('particle conservation failed')
        expected = particle_offset + 32 + int(counts[0])*10*8
        if st.st_size != expected: raise ValueError('truncated or incompatible checkpoint')
    row = dict(path=str(path), bytes=st.st_size, header_bytes=offset,
               time=simtime, cycle=cycle, dt=dt, blocks=nmb,
               particles=int(counts[0]), removed=list(map(int,counts[1:])))
    if hash_file: row['sha256'] = digest(path)
    if path.stat().st_size != st.st_size or path.stat().st_mtime_ns != st.st_mtime_ns:
        raise ValueError('checkpoint changed during verification')
    return row

def last_cycle(log):
    if not Path(log).exists(): return -1
    with Path(log).open('rb') as f:
        f.seek(max(0,Path(log).stat().st_size-1024**2))
        text=f.read().decode(errors='replace')
    cycles=re.findall(r'cycle\s*=\s*(\d+)',text)
    return int(cycles[-1]) if cycles else -1

def retain(root, closed=False, run=None):
    """Only verify completed writes; retain latest three globally, plus active write."""
    root=Path(root)
    with (root/'control/retention.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        known_path=root/'control/checkpoints.json'
        known=json.loads(known_path.read_text()) if known_path.exists() else {}
        for p in (root/'runs').rglob('*.rst'):
            if str(p) in known: continue
            try:
                row=checkpoint(p,False)
                parent=p.parent.parent
                exit_file=parent/'EXIT_CODE'
                finished=exit_file.exists() and exit_file.read_text().strip()=='0'
                if not finished and last_cycle(parent/'run.log') <= row['cycle']: continue
                row=checkpoint(p)
            except (ValueError,OSError,struct.error): continue
            known[str(p)]=row
        existing=[v for k,v in known.items() if Path(k).exists()]
        existing.sort(key=lambda x:(x['cycle'],Path(x['path']).stat().st_mtime_ns))
        archived_path=root/'control/archived_checkpoints.json'
        archived=json.loads(archived_path.read_text()) if archived_path.exists() else {}
        for row in existing[:-3]:
            receipt=archived.get(row['path'])
            if not receipt or receipt.get('sha256') != row['sha256']: continue
            # The Anta Slurm archive receipt proves a complete checksum-verified copy.
            with (root/'evidence/checkpoint_deletions.jsonl').open('a') as f:
                f.write(json.dumps(dict(row,deleted_utc=time.time(),archive=receipt))+'\n')
            Path(row['path']).unlink()
        atomic(known_path,known)
        kept=existing[-3:]
        if kept: atomic(root/'control/latest_checkpoint.json',kept[-1])
        return kept

def stop(root, reason):
    root=Path(root)
    p=root/'control/RESOURCE_STOP.json'
    if not p.exists(): atomic(p,dict(reason=reason,utc=time.time()))
    (root/'control/REQUEST_STOP').touch()

def sample_vram(root,run):
    root,run=Path(root),Path(run)
    host=os.uname().nodename
    while True:
        try:
            out=subprocess.check_output(['rocm-smi','--showmeminfo','vram','--json'],text=True,timeout=20)
            raw=json.loads(out[out.index('{'):])
            rows=[]
            for card, vals in raw.items():
                if not isinstance(vals,dict): continue
                total=int(vals['VRAM Total Memory (B)']); used=int(vals['VRAM Total Used Memory (B)'])
                if total<=0: raise ValueError('VRAM total invalid')
                rows.append(dict(host=host,card=card,total=total,used=used,fraction=used/total,utc=time.time()))
            if len(rows)!=4: raise ValueError('expected four MI210 GPUs')
            with (run/('vram_'+host+'.jsonl')).open('a') as f:
                for row in rows: f.write(json.dumps(row)+'\n')
            if any(r['fraction']>=.85 for r in rows): stop(root,'gpu_memory_85_percent')
        except Exception as e:
            with (run/('vram_'+host+'.errors')).open('a') as f: f.write(str(e)+'\n')
        time.sleep(2)

def monitor(root,run):
    root,run=Path(root),Path(run)
    began=time.time(); last_change=began; prev=-1
    while True:
        try:
            used=int(subprocess.check_output(['du','-sx','--block-size=1',str(root.parent)],text=True,timeout=120).split()[0])
            campaign=int(subprocess.check_output(['du','-sx','--block-size=1',str(root)],text=True,timeout=120).split()[0])
            checkpoint_reserve=256*1024**3
            warning=used>=1.5*1024**4
            atomic(run/'storage_status.json',dict(user_bytes=used,campaign_bytes=campaign,warning=warning,utc=time.time()))
            if used>=1.7*1024**4 or used+checkpoint_reserve>=1.9*1024**4 or campaign+checkpoint_reserve>=1.25*1024**4:
                stop(root,'storage_limit')
            if (root/'control/ARCHIVE_STORAGE_STOP').exists() or (campaign>=256*1024**3 and (root/'control/ARCHIVE_ERROR').exists()):
                stop(root,'archive_blocked_with_storage_growth')
        except Exception as e:
            # A transient du failure never kills the watchdog.
            with (run/'monitor_errors.txt').open('a') as f: f.write(str(e)+'\n')
        cyc=last_cycle(run/'run.log')
        if (root/'control/USER_STOP').exists(): (root/'control/REQUEST_STOP').touch()
        heartbeat=root/'control/ARCHIVE_HEARTBEAT'
        if run.name.startswith('segment_') and (not heartbeat.exists() or time.time()-float(heartbeat.read_text())>900):
            stop(root,'archive_heartbeat_stale_900s')
        if cyc!=prev: prev=cyc; last_change=time.time()
        if time.time()-last_change>3600: stop(root,'progress_stale_3600s')
        try: retain(root)
        except Exception as e:
            stop(root,'checkpoint_retention_error')
            with (run/'monitor_errors.txt').open('a') as f: f.write(str(e)+'\n')
        time.sleep(300)

def seal(run):
    """Science hashes are computed only on an allocated AMD compute node."""
    run=Path(run)
    if (run/'SEALED').exists(): return
    rows=[]
    for p in sorted(run.rglob('*')):
        if p.is_file() and p.name not in ('SCIENCE_MANIFEST.json','SEALED'):
            rows.append(dict(path=str(p.relative_to(run)),bytes=p.stat().st_size,sha256=digest(p)))
    atomic(run/'SCIENCE_MANIFEST.json',rows)
    (run/'SEALED').touch()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['vram','monitor','retain','seal','checkpoint'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--run',type=Path)
    a=p.parse_args()
    if a.action=='vram': sample_vram(a.root,a.run)
    elif a.action=='monitor': monitor(a.root,a.run)
    elif a.action=='retain': print(json.dumps(retain(a.root,True,a.run),indent=2))
    elif a.action=='seal': seal(a.run)
    else: print(json.dumps(checkpoint(a.run),indent=2))
