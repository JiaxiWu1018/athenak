"""Session 10 explicit cases and allocation-only scientific helpers."""
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

LOCAL = Path('/data/jiaxiwu/NRPIC/ST_migration/session_10_yc0500_outward_20261024/extensions/u0_1500_to2500_20261026')
AMD = Path('/work1/eliasmost/jiaxiwu/st_migration_s10_u0_2500_20261026')
EXT = Path(os.environ.get('ST8X_ROOT', str(AMD)))
PARENT = Path('/work1/eliasmost/jiaxiwu/st_migration_s10_20261024')
PRIOR_ALLOCATED_NODE_SECONDS = 46439
PRIOR_REQUESTED_NODE_SECONDS = 82020
SESSION = LOCAL
MACHINE = 'amd'
BASE = Path('/work1/eliasmost/jiaxiwu/st_reconstruction_s08_production_20261021')
DONOR = Path('/work1/eliasmost/jiaxiwu/st_s08_scan_20261023/yc0500_long700')
ANCHOR_SHA = 'a1eb74e9709fa407d3e0178cd30e561d35556f5d9bb5afe9bf04fafe0db45f3d'
OLD_EXE_SHA = '41411a2e024cb910d2313e9b30ec4370fb37cc9a9dea60cc015c3df3682f1e71'
PERTURBED_EXE_SHA = 'fa261c0bd46ea96cc57e8df593946a05f2ea72ffc53863c7d55f902f69c0cafc'
GIB = 1024**3
TIB = 1024**4
ALL_CASES = json.loads((Path(__file__).parent/'cases.json').read_text())

def case_config(key):
    raw = next(c for c in ALL_CASES if c['id'] == key)
    return dict(raw, key=key, id='yc0500', unstable=True)

CASE = case_config(os.environ.get('ST8_CASE', 'yc0500_control'))
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
        'UCX_LOG_LEVEL', 'UCX_NET_DEVICES', 'OMPI_MCA_btl_vader_single_copy_mechanism']}

def resource_policy():
    policy = load(EXT/'evidence/RESOURCE_POLICY.json')
    assert policy['node_hour_limit'] is None, 'Direct user approval removes the compute cap'
    return policy

def native_launcher():
    return ['prun']

def rank_capacity():
    assert int(os.environ['SLURM_NTASKS']) in [4, 8]
    return 128
