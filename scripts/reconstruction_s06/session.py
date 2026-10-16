"""Session 6 paths and immutable scientific identities."""
import datetime
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'reconstruction_s05'))
from common import atomic, sha, require_job, sample  # noqa: E402

ROOT = Path(os.environ.get('ROOT', '/data/jiaxiwu/NRPIC/reconstruction_ST/06_yc0819_long_stability_20261016'))
CASE = 'proper_stratified_s1'
N = 1500000
END = 1000.0
P = 197.62595786609717
PHYSICS = '5a5db71c9d02498bfe3402c73c8a2b4b3180e377'
EXE_SHA = '5fb4fb9f2131984854b3dd2eef455e8acc6808152e800698fcf5349e5bc35482'
RST_SHA = 'bd785c67bee977b007c43a6dd9acdbaecf4855065f222f36f5d047786a805d4c'
KOKKOS = '6739bc623081648af9e752b616d9671527922cbf'
RESTART = ROOT / 'restart' / (CASE + '.00003.rst')

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def read_csv(path):
    import csv
    with open(path) as stream:
        return list(csv.DictReader(stream))

def load(path):
    return json.loads(Path(path).read_text())
