"""Session8 five-case batch configuration; immutable native physics."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

LOCAL=Path('/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023')
SESSION=LOCAL.parents[1]
AMD=Path('/work1/eliasmost/jiaxiwu/st_s08_scan_20261023')
MACHINE=os.environ.get('S8_MACHINE','amd')
EXT=Path(os.environ.get('ST8X_ROOT',str(LOCAL/'perseus' if MACHINE=='perseus' else AMD)))
TIB=1024**4
GIB=1024**3
THREAD='01a15d4b-acc1-7623-b3e7-209e4daa341c'
configuration=LOCAL/'evidence/cases.json' if MACHINE=='perseus' else EXT/'evidence/cases.json'
ALL_CASES=json.loads(configuration.read_text())

def case_config(key):
    raw=next(c for c in ALL_CASES if c['id']==key)
    return dict(raw,key=key,id=f"yc{round(raw['yc']*1000):04d}",unstable=True,
                num_levels=9 if raw['L']==1024 else 8)

CASE=case_config(os.environ.get('ST8_CASE','yc0500_long700'))
CASES=[CASE]
ROOT=Path(os.environ.get('ROOT',str(EXT/CASE['key'])))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def load(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def job():
    assert os.environ.get('SLURM_JOB_ID'),'Compute allocation required'
    return os.environ['SLURM_JOB_ID']
def cmd(args,**kwargs):return subprocess.run(list(map(str,args)),check=True,**kwargs)
def receipt(path):
    p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def transport_profile():return {k:os.environ.get(k,'') for k in ['UCX_TLS','UCX_MEMTYPE_CACHE','UCX_LOG_LEVEL','OMPI_MCA_btl_vader_single_copy_mechanism']}
def native_launcher():
    if MACHINE=='perseus':return ['mpirun','--bind-to','none','-np',os.environ['SLURM_NTASKS']]
    return ['prun']
def rank_capacity():
    ranks=int(os.environ['SLURM_NTASKS']);default=max(128,512//ranks)
    capacity=int(os.environ.get('ST8_MAX_NMB_PER_RANK',str(default)))
    assert default<=capacity<=(384 if MACHINE=='perseus' else 256)
    return capacity
