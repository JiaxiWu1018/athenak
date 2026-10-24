"""Session 9 explicit case configuration; frozen native physics."""
import datetime, hashlib, json, os, subprocess
from pathlib import Path
LOCAL=Path('/data/jiaxiwu/NRPIC/reconstruction_ST/09_perturbed_production_20261024')
SESSION=LOCAL.parent/'08_production_20261021'
AMD=Path('/work1/eliasmost/jiaxiwu/st_s09_perturbed_production_20261024')
MACHINE='amd'
EXT=Path(os.environ.get('ST8X_ROOT',str(AMD)))
GIB=1024**3
TIB=1024**4
ALL_CASES=json.loads((EXT/'evidence/cases.json').read_text())
REFERENCE=dict(id='yc0500_u0005',yc=.5,U=-.005,L=512,target=200.,initial_dx=.25,finest_dx=.0625,num_levels=8,reused_session8=True)
PLOT_CASES=sorted(ALL_CASES+[REFERENCE],key=lambda c:c['yc'])
def case_config(key):
    raw=next(c for c in PLOT_CASES if c['id']==key)
    return dict(raw,key=key,id=f"yc{round(raw['yc']*1000):04d}",unstable=True)
CASE=case_config(os.environ.get('ST8_CASE','yc0283_u0005'))
CASES=[CASE]
ROOT=Path(os.environ.get('ROOT',str(EXT/CASE['key'])))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def load(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def job():
    assert os.environ.get('SLURM_JOB_ID'),'Slurm compute allocation required'
    return os.environ['SLURM_JOB_ID']
def cmd(args,**kwargs):return subprocess.run(list(map(str,args)),check=True,**kwargs)
def receipt(path):
    p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def transport_profile():return {k:os.environ.get(k,'') for k in ['UCX_TLS','UCX_MEMTYPE_CACHE','UCX_LOG_LEVEL','OMPI_MCA_btl_vader_single_copy_mechanism']}
def native_launcher():return ['prun']
def rank_capacity():
    ranks=int(os.environ['SLURM_NTASKS']);assert ranks in (4,8)
    capacity=int(os.environ.get('ST8_MAX_NMB_PER_RANK','160' if ranks==4 else '128'))
    assert capacity== (160 if ranks==4 else 128)
    return capacity
