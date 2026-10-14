"""Campaign constants and durable records; scientific work runs under Slurm."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get('ROOT', '/data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014'))
AMD = '/work1/eliasmost/jiaxiwu/st_reconstruction_s05_20261014'
N = 1500000
BASE = 'e011390767040f512286fcb7b461fdc1ca6246a2'
KOKKOS = '6739bc623081648af9e752b616d9671527922cbf'
CASES = [(f'{radial}_{method}_s{seed}', radial, method, seed)
         for radial in ('legacy', 'proper')
         for seed in (1, 2, 3)
         for method in ('mc', 'stratified', 'cell', 'ablation')]

def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.writing')
    tmp.write_text(obj if isinstance(obj, str) else json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def sample(path):
    import numpy as np
    with open(path, 'rb') as f:
        header = f.readline().decode().split()
        offset = f.tell()
    assert header[0] == 'STSAMP05'
    a = np.memmap(path, dtype='<f8', mode='r', offset=offset, shape=(int(header[1]), 8))
    return header, a

def require_job():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Scientific generation/reduction requires a Slurm allocation')
