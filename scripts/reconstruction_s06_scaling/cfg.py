"""Approved session6 follow-up: immutable case definitions and paths."""
import datetime
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'reconstruction_s05'))
from common import atomic, sha, sample, require_job

PERSEUS = Path('/data/jiaxiwu/NRPIC/reconstruction_ST/06_yc0819_long_stability_20261016/followups/paired_scaling_20261017')
AMD = Path('/work1/eliasmost/jiaxiwu/st_reconstruction_s06_paired_scaling_20261017')
ROOT = Path(os.environ.get('ROOT', str(PERSEUS)))
BASE = PERSEUS.parents[1]
S05 = BASE.parent / '05_sampling_shell_response_20261014'
OS_ROOT = Path('/data/jiaxiwu/NRPIC/paper/session_01_os_collapse/convergence')
THREAD = '01a14666-7bb8-74d1-8819-0037c04e9eb6'
PHYSICS = '5a5db71c9d02498bfe3402c73c8a2b4b3180e377'
EXE_SHA = '5fb4fb9f2131984854b3dd2eef455e8acc6808152e800698fcf5349e5bc35482'
PARENT_SHA = '8bcd9df2466c73a61333d6825b4f309d4301925ec5c43cf3eeb6145729a49390'
KOKKOS = '6739bc623081648af9e752b616d9671527922cbf'
P = 197.62595786609717
CASES = [dict(id='paired_3m_s1', n=3000000, paired=True, raw_forecast=100000000000),
         dict(id='ordinary_3m_s1', n=3000000, paired=False, raw_forecast=100000000000),
         dict(id='ordinary_6m_s1', n=6000000, paired=False, raw_forecast=170000000000)]

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def load(path):
    return json.loads(Path(path).read_text())

def source():
    return load(ROOT / 'evidence/SOURCE_PIN.json')['head']

def write_sample(path, data):
    if path.exists():
        raise FileExistsError(path)
    with path.open('wb') as stream:
        stream.write(f'STSAMP05 {len(data)} 1 1 .819 .0001 stratified proper stratified\n'.encode())
        data.astype('<f8').tofile(stream)
