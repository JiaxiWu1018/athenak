"""Session 10 explicit cases and allocation-only scientific helpers."""
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

LOCAL = Path('/data/jiaxiwu/NRPIC/ST_migration/session_10_yc0500_outward_20261024')
AMD = Path('/work1/eliasmost/jiaxiwu/st_migration_s10_20261024')
EXT = Path(os.environ.get('ST8X_ROOT', str(AMD)))
SESSION = LOCAL
MACHINE = 'amd'
BASE = Path('/work1/eliasmost/jiaxiwu/st_reconstruction_s08_production_20261021')
DONOR = Path('/work1/eliasmost/jiaxiwu/st_s08_scan_20261023/yc0500_long700')
ANCHOR_SHA = '942230e4315b18e7850ab24e4ba2bf02d296cc324797b1cf50fdf2a6a5ad1624'
OLD_EXE_SHA = '41411a2e024cb910d2313e9b30ec4370fb37cc9a9dea60cc015c3df3682f1e71'
PERTURBED_EXE_SHA = 'fa261c0bd46ea96cc57e8df593946a05f2ea72ffc53863c7d55f902f69c0cafc'
GIB = 1024**3
TIB = 1024**4
ALL_CASES = json.loads((Path(__file__).parent/'cases.json').read_text())

def case_config(key):
    raw = next(c for c in ALL_CASES if c['id'] == key)
    return dict(raw, key=key, id='yc0500', unstable=True)

CASE = case_config(os.environ.get('ST8_CASE', 'yc0500_outward'))
CASES = [CASE]
ROOT = Path(os.environ.get('ROOT', str(EXT/CASE['key'])))

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def atomic(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(p)

def load(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()

def job():
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm compute allocation required'
    return os.environ['SLURM_JOB_ID']

def cmd(args, **kwargs):
    return subprocess.run(list(map(str, args)), check=True, **kwargs)

def receipt(path):
    p = Path(path)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))

def transport_profile():
    return {k: os.environ.get(k, '') for k in ['UCX_TLS', 'UCX_MEMTYPE_CACHE',
        'UCX_LOG_LEVEL', 'OMPI_MCA_btl_vader_single_copy_mechanism']}

def native_launcher():
    return ['prun']

def rank_capacity():
    assert int(os.environ['SLURM_NTASKS']) == 8
    return 128
