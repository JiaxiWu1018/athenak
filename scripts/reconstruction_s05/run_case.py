"""One isolated launch with complete native-state, weight and provenance checks."""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
import numpy as np
import s8_checkpoint as C
from common import ROOT, N, CASES, atomic, sha, sample, require_job

def checkpoint(path, expected=None, initial=None, n=N):
    d,b=C.header(path); C.payload(path,d,b)
    if d['particle_count']!=n or any(d['removal_header']):raise ValueError('Particle ledger changed')
    if expected is not None and abs(d['time']-expected)>1e-7:raise ValueError('Checkpoint time differs')
    off=d['ascii_bytes']+252+d['nmb']*20+16+d['data_size_per_mb']*d['nmb']+32
    p=np.memmap(path,dtype='<f8',mode='r',offset=off,shape=(10,n))
    order=np.argsort(p[9]); tags=p[9,order]
    if not np.array_equal(tags,np.arange(n)):raise ValueError('Original IDs changed')
    mass=p[0,order]
    if initial is not None:
        _,a=sample(initial)
        if not np.array_equal(mass,a[:,7]):raise ValueError('Per-ID double-precision weights changed')
    if not np.all(mass>0):raise ValueError('Nonpositive particle mass')
    d.update(particle_data_offset=off,rest_mass=float(mass.sum()),sha256=sha(path))
    return d

def newest(out):
    paths=list((Path(out)/'rst').glob('*.rst'))
    if not paths:raise ValueError('No checkpoint')
    return max(paths,key=lambda p:C.header(p)[0]['time'])

def launch(work, deck, end, executable='athena', restart=None, extra=()):
    work=Path(work); out=work/'out';out.mkdir(parents=True,exist_ok=False)
    exe=ROOT/'protected'/executable
    cmd=['prun',str(exe),'--kokkos-map-device-id-by=mpi_rank','-t','01:40:00',
         '-r' if restart else '-i',str(restart or deck),'-d',str(out),f'time/tlim={end}']+list(extra)
    atomic(work/'launch.json',{'job':os.environ['SLURM_JOB_ID'],'host':__import__('socket').gethostname(),
           'command':cmd,'input':str(deck),'input_sha256':sha(deck),'executable_sha256':sha(exe),
           'source':subprocess.check_output(['git','-C',str(ROOT/'src/athenak'),'rev-parse','HEAD'],text=True).strip(),
           'kokkos':subprocess.check_output(['git','-C',str(ROOT/'src/athenak/kokkos'),'rev-parse','HEAD'],text=True).strip()})
    begin=time.monotonic()
    with open(work/'solver.log','w') as log:
        proc=subprocess.Popen(cmd,cwd=work,stdout=log,stderr=subprocess.STDOUT)
        interim=False
        while proc.poll() is None:
            time.sleep(30)
            if work.parent==ROOT/'runs' and not interim:
                with open(work/'solver.log','rb') as recent:
                    recent.seek(max(0,(work/'solver.log').stat().st_size-32768))
                    times=re.findall(r'cycle=\d+\s+time=([0-9.eE+-]+)',recent.read().decode(errors='replace'))
                if times and float(times[-1])>=104:
                    subprocess.run(['python3',str(ROOT/'scripts/analyze.py'),'--case',work.name,'--until','100'],check=True)
                    atomic(work/'INTERIM_100_OK','Validated initial-to100M comparison data\n')
                    interim=True
        if proc.returncode:raise subprocess.CalledProcessError(proc.returncode,cmd)
    text=(work/'solver.log').read_text()
    if 'Terminating on time limit' not in text or '[conservation OK]' not in text:raise ValueError('Run did not complete cleanly')
    if re.search(r'search_fail=[1-9]|rejected=[1-9]|nonfinite=[1-9]',text):raise ValueError('Invalid motion/deposition')
    return out,time.monotonic()-begin

def main():
    require_job(); ap=argparse.ArgumentParser();ap.add_argument('index',type=int);a=ap.parse_args()
    if not (ROOT/'evidence/PREFLIGHT_OK').exists():raise ValueError('Preflight gate missing')
    if (ROOT/'evidence/STORAGE_STOP').exists():raise ValueError('Storage stop remains active')
    case,radial,method,seed=CASES[a.index]
    work=ROOT/'runs'/case; deck=ROOT/'inputs'/(case+'.athinput');initial=ROOT/'samples'/(case+'.bin')
    expected=json.loads((ROOT/'evidence/samples'/(case+'.json')).read_text())
    if sha(initial)!=expected['sha256']:raise ValueError('Sample identity changed')
    end=json.loads((ROOT/'evidence/PREFLIGHT_OK').read_text())['endpoint_M']
    out,wall=launch(work,deck,end)
    states=[checkpoint(p,initial=initial) for p in sorted((out/'rst').glob('*.rst'))]
    if not states or abs(max(s['time'] for s in states)-end)>1e-7:raise ValueError('Incomplete endpoint')
    files=[]
    for p in sorted(work.rglob('*')):
        if p.is_file():files.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
    atomic(work/'completion.json',{'case':case,'job':os.environ['SLURM_JOB_ID'],'endpoint_M':end,
           'solver_wall_seconds':wall,'checkpoints':states,'raw_bytes':sum(f['bytes'] for f in files),
           'manifest':files,'input_sample_sha256':expected['sha256']})
    # Allocated analysis immediately reduces this case; failed analysis is a job failure.
    subprocess.run(['python3',str(ROOT/'scripts/analyze.py'),'--case',case],check=True)
    atomic(work/'RUN_OK',f'{case} verified and reduced through {end}M\n')

if __name__=='__main__':main()
