"""Session7 constants; native data and all computation remain on AMD."""
import datetime
import json
import os
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'reconstruction_s06'))
sys.path.insert(0,str(HERE.parent/'reconstruction_s05'))
from common import atomic,sha,sample,require_job
PERSEUS=Path('/data/jiaxiwu/NRPIC/reconstruction_ST/07_mc_particle_convergence_20261019')
AMD=Path('/work1/eliasmost/jiaxiwu/st_reconstruction_s07_mc_20261019')
ROOT=Path(os.environ.get('ROOT',str(PERSEUS)))
PREVIOUS=PERSEUS.parent/'06_yc0819_long_stability_20261016'
S05=PERSEUS.parent/'05_sampling_shell_response_20261014'
THREAD='01a15302-1ccc-7351-87e7-448a27ee9c12'
PHYSICS='5a5db71c9d02498bfe3402c73c8a2b4b3180e377'
KOKKOS='6739bc623081648af9e752b616d9671527922cbf'
EXE_SHA='5fb4fb9f2131984854b3dd2eef455e8acc6808152e800698fcf5349e5bc35482'
P=197.62595786609717
CASES=[dict(id=f'mc_{n}m_s1',n=n*1000000,paired=False,wall_hours=h,raw_forecast=b*1000000000)
       for n,h,b in [(1,4,60),(2,6,85),(4,10,130),(8,20,220)]]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(path):return json.loads(Path(path).read_text())
def source():return load(ROOT/'evidence/SOURCE_PIN.json')['head']
def write_sample(path,data):
    assert not path.exists()
    with path.open('wb')as f:
        f.write(f'STSAMP05 {len(data)} 1 1 .819 .0001 random proper random\n'.encode())
        data.astype('<f8').tofile(f)
