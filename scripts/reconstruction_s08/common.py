"""Session 8 immutable configuration and small provenance helpers."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

LOCAL=Path('/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021')
AMD=Path('/work1/eliasmost/jiaxiwu/st_reconstruction_s08_production_20261021')
ROOT=Path(os.environ.get('ROOT',str(LOCAL)))
THREAD='01a15d4b-acc1-7623-b3e7-209e4daa341c'
CASES=[dict(id=f'yc{round(y*1000):04d}',yc=y,unstable=y<.7,L=512 if y<.7 else 1024,
            num_levels=8 if y<.7 else 7) for y in [.283,.5,.605,.819,.9]]
TIB=1024**4
GIB=1024**3

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def load(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def job():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Compute allocation required')
    return os.environ['SLURM_JOB_ID']
def cmd(args,**kwargs):return subprocess.run(list(map(str,args)),check=True,**kwargs)
def receipt(path):
    p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
