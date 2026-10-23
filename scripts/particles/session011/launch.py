#!/usr/bin/env python3
"""Explicit future execution only. Preparation never calls these science actions."""
import argparse
import bisect
import csv
import json
import math
import os
import re
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import time
from prepare import HERE,CODE,SESSION,sha,atomic,source_hash,parse
from telemetry_reader import StableSamplerReader

def job_end():
 value=os.environ.get('SLURM_JOB_END_TIME')
 if value:return int(value)
 text=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'-o'],text=True,timeout=20)
 stamp=re.search(r'EndTime=(\S+)',text)
 if stamp is None:raise RuntimeError('Cannot determine allocation end time')
 value=int(time.mktime(time.strptime(stamp[1],'%Y-%m-%dT%H:%M:%S')))
 os.environ['SLURM_JOB_END_TIME']=str(value)
 return value

def builds():
 h=source_hash()
 for kind in ['gi','os','spin']:
  r=json.loads((SESSION/f'evidence/build/{kind}.json').read_text())
  if r['native_source_sha256']!=h or sha(r['executable'])!=r['executable_sha256']:
   raise RuntimeError('Stale native build: '+kind)
 return h

def rows(path):
 with Path(path).open() as f:return list(csv.DictReader(f))

def require_gpu(partition,nodes):
 if os.environ.get('SLURM_JOB_PARTITION')!=partition or int(os.environ.get('SLURM_NNODES','0'))!=nodes:
  raise RuntimeError(f'Requires allocated {nodes}-node {partition} job')

def run_case(root,label,kind,deck=None,restart=None,extra=(),limit=3300):
 case=root/label;case.mkdir(parents=True,exist_ok=False)
 out=case/'out';out.mkdir()
 remain=job_end()-int(time.time())-180
 seconds=min(limit,remain)
 if seconds<180:raise RuntimeError('Validation allocation reserve reached')
 args=['ibrun','-n','4',str(SESSION/f'build/{kind}/src/athena'),'--kokkos-map-device-id-by=mpi_rank']
 args+=['-r',str(restart)] if restart else ['-i',str(deck)]
 args+=['-d',str(out),'-t',f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}',
        'time/stop_file='+str(out/'REQUEST_STOP'),*extra]
 atomic(case/'COMMAND.json',dict(command=args,job_id=os.environ['SLURM_JOB_ID'],native_source_sha256=source_hash()))
 with (case/'athena.log').open('w') as f:
  p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=seconds+90)
  except subprocess.TimeoutExpired:
   (out/'REQUEST_STOP').touch()
   try:p.wait(timeout=60)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=30)
   raise RuntimeError('Native validation timeout; outputs retained')
 if code:raise RuntimeError(f'Native validation {label} failed: {code}')
 return out

def accepted(out):
 return [r for r in rows(out/'os_ah.ah_0.csv') if r['published']=='1']

def validate():
 require_gpu('gh-dev',4);h=builds()
 root=SESSION/'runtime'/('validation_'+os.environ['SLURM_JOB_ID']);root.mkdir(parents=True,exist_ok=False)
 try:
  smoke=run_case(root,'compact_gi','gi',HERE/'inputs/compact_validation.athinput',limit=600)
  component=rows(smoke/'s11_smoke.components.csv')
  if [int(float(r['count'])) for r in component[:3]]!=[3000000,1000000,1000000]:
   raise RuntimeError('Compact GI initial component census failed')
  for name in ['s11_smoke.radial_profiles.csv','s11_smoke.cohort_updates.csv','s11_smoke.trackers.csv']:
   if not (smoke/name).is_file():raise RuntimeError('Missing output: '+name)
  ref=run_case(root,'os_reference','os',HERE/'inputs/os_validation.athinput',limit=3300)
  pub=accepted(ref)
  if len(pub)<10:raise RuntimeError('OS did not produce accepted horizon history')
  candidates=[]
  for cp in sorted(ref.glob('rst/*.rst')):
   with cp.open('rb') as f:
    data=f.read(1024**2);end=data.index(b'<par_end>')+10
    header=parse(data[:end-10].decode());f.seek(end+232);tm,dt,cycle=struct.unpack('<ddi',f.read(20))
   if tm<35.1 and header.get('fastflow',{}).get('surface_state_valid_0')=='true':candidates.append((tm,cp))
  if not candidates:raise RuntimeError('No mid-run accepted-shape checkpoint')
  tm,cp=max(candidates)
  resumed=run_case(root,'os_bad_fallback_restart','os',restart=cp,
   extra=['fastflow/initial_radius_0=0.01'],limit=1500)
  found=accepted(resumed)
  if len(found)<3 or float(found[0]['state_time'])-tm>0.1:
   raise RuntimeError('OS restart did not reacquire within 0.1 M')
  all_restart=rows(resumed/'os_ah.ah_0.csv')
  if all_restart[0]['published']=='1':raise RuntimeError('Restored guess incorrectly published immediately')
  times=[float(r['state_time']) for r in pub]
  errors=[]
  for r in found:
   t=float(r['state_time']);k=min(range(len(times)),key=lambda i:abs(times[i]-t))
   if abs(times[k]-t)>0.03:continue
   a=pub[k]
   errors.append(max(abs(float(r['mass'])/float(a['mass'])-1),abs(float(r['spin'])-float(a['spin']))))
  if not errors or max(errors)>0.005:raise RuntimeError('OS uninterrupted/restart AH mismatch')
  if abs(float(pub[-1]['mass'])-1)>0.05:raise RuntimeError('OS late mass differs from M=1 by >5%')
  spins=[]
  for label,nt,lm in [('spin_coarse',10,6),('spin_fine',20,8)]:
   out=run_case(root,label,'spin',HERE/'inputs/spin_validation.athinput',
    extra=[f'fastflow/ntheta={nt}',f'fastflow/lmax={lm}'],limit=300)
   a=[r for r in rows(out/'spin_ah.ah_0.csv') if r['published']=='1']
   if not a:raise RuntimeError('No accepted spinning horizon')
   m,s=float(a[-1]['mass']),float(a[-1]['spin'])
   if abs(m-1)>0.02 or abs(s-0.5)>0.02:raise RuntimeError('Spin calibration failed')
   spins.append((m,s))
  if abs(spins[0][0]/spins[1][0]-1)>0.01 or abs(spins[0][1]-spins[1][1])>0.01:
   raise RuntimeError('Angular refinement comparison failed')
  receipt=dict(passed=True,native_source_sha256=h,production_input_sha256=sha(HERE/'inputs/production.athinput'),
   validation_job=os.environ['SLURM_JOB_ID'],outputs=str(root),checkpoint_sha256=sha(cp),
   os_restart_max_error=max(errors),spin_measurements=spins,production_authorized=False,
   full_size_memory_certified=False,poor_fallback_restart_passed=True,
   original_native_executable_replayed=False)
  atomic(root/'PASS.json',receipt);atomic(SESSION/'evidence/VALIDATION_PASS.json',receipt)
 except Exception as e:
  atomic(root/'FAILURE.json',dict(error=repr(e),passed=False,outputs_retained=True));raise

def ready():
 h=builds();v=json.loads((SESSION/'evidence/VALIDATION_PASS.json').read_text())
 if v.get('passed') is not True or v['native_source_sha256']!=h or v['production_input_sha256']!=sha(HERE/'inputs/production.athinput'):
  raise RuntimeError('Missing/stale/failed compact validation')
 p=json.loads((SESSION/'evidence/PUSH_VERIFIED.json').read_text())
 if p.get('passed') is not True or p.get('native_source_sha256')!=h:
  raise RuntimeError('Tested fast-forward push has not been verified')
 return h

def sampler(case):
 case=Path(case);folder=case/'telemetry';folder.mkdir(parents=True,exist_ok=True)
 host=socket.gethostname()
 while not (case/'STOP_SAMPLERS').exists():
  d=dict(host=host,epoch=time.time())
  try:
   mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
   d.update(host_total_bytes=mem['MemTotal'],host_used_bytes=mem['MemTotal']-mem['MemAvailable'])
   raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=15)
   d['gpus']=[dict(uuid=a[0].strip(),used_mib=float(a[1]),total_mib=float(a[2])) for a in (l.split(',') for l in raw.splitlines())]
  except Exception as e:d['error']=repr(e)
  atomic(folder/(host+'.json'),d)
  with (folder/(host+'.jsonl')).open('a') as f:f.write(json.dumps(d)+'\n')
  time.sleep(10)

def production():
 require_gpu('gh',16);ready()
 root=SESSION/'runtime'/('production_'+os.environ['SLURM_JOB_ID']);root.mkdir(parents=True,exist_ok=False)
 out=root/'out';out.mkdir();(root/'telemetry').mkdir()
 seconds=min(36*3600,job_end()-int(time.time()))-2400
 if seconds<600:raise RuntimeError('Insufficient production allocation remaining')
 args=['ibrun','-n','16',str(SESSION/'build/gi/src/athena'),'--kokkos-map-device-id-by=mpi_rank',
  '-i',str(HERE/'inputs/production.athinput'),'-d',str(out),'-t',f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}',
  'time/stop_file='+str(out/'REQUEST_STOP')]
 atomic(root/'COMMAND.json',dict(command=args,job_id=os.environ['SLURM_JOB_ID'],fresh_t0=True,no_planned_restarts=True))
 sampling=['srun','--overlap','--nodes=16','--ntasks=16','--ntasks-per-node=1','--cpus-per-task=1',sys.executable,str(HERE/'launch.py'),'sampler',str(root)]
 reader=StableSamplerReader();cfg={'telemetry_stale_seconds':120};reason=None;p=None
 with (root/'samplers.log').open('w') as sf,(root/'athena.log').open('w') as log:
  sam=subprocess.Popen(sampling,stdout=sf,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   deadline=time.time()+120
   while True:
    try:measures=reader(root/'telemetry',16,cfg);break
    except RuntimeError:
     if time.time()>deadline or sam.poll() is not None:raise RuntimeError('Sampler startup failed')
     time.sleep(1)
   def guard(measures):
    for m in measures:
     if m.get('error'):return 'sampler_error'
     if m['host_used_bytes']/m['host_total_bytes']>=0.85:return 'host_memory_85pct'
     if any(g['used_mib']/g['total_mib']>=0.85 for g in m['gpus']):return 'GPU_memory_85pct'
    return None
   if guard(measures):raise RuntimeError('Startup resource guard failed')
   p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   bad_since=None; stop_sent=None; startup_checked=False
   while p.poll() is None:
    try:
     reason=guard(reader(root/'telemetry',16,cfg));bad_since=None
    except RuntimeError:
     bad_since=bad_since or time.time()
     if time.time()-bad_since>120:reason='persistent_telemetry_failure'
    stats=out/'s11_orbit_gw.components.csv'
    if stats.exists():
     with stats.open('rb') as f:f.seek(max(0,stats.stat().st_size-4096));tail=f.read().decode().splitlines()
     for line in tail[-3:]:
      a=line.split(',')
      if len(a)==19 and a[0].isdigit() and int(a[3])>=5760:reason='leaf_capacity_proxy'
     if not startup_checked:
      initial=rows(stats)
      if len(initial)>=3:
       counts=[int(float(r['count'])) for r in initial[:3]]
       if counts!=[3000000,1000000,1000000] or int(initial[0]['leaf_blocks'])!=4936:
        reason='full_size_initial_census_or_mesh'
       else:
        atomic(root/'FULL_SIZE_STARTUP.json',dict(passed=True,initial_counts=counts,leaf_blocks=4936,
         measured_per_node_telemetry=True,job_id=os.environ['SLURM_JOB_ID'],long_time_memory_certified=False))
       startup_checked=True
    free=os.statvfs(out)
    if free.f_bavail*free.f_frsize<150*1024**3:reason='disk_write_reserve'
    if time.time()>int(os.environ['SLURM_JOB_END_TIME'])-2400:reason=reason or 'wall_reserve'
    if reason:
     (out/'REQUEST_STOP').touch();stop_sent=stop_sent or time.time()
     if time.time()-stop_sent>600:
      os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=30);break
    time.sleep(10)
   code=p.returncode
  finally:
   if p is not None and p.poll() is None:
    (out/'REQUEST_STOP').touch()
    try:p.wait(timeout=90)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=30)
   (root/'STOP_SAMPLERS').touch()
   try:sam.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(sam.pid,signal.SIGTERM);sam.wait(timeout=20)
 from checkpoint_io import checkpoint
 checkpoints=[checkpoint(q,hash_file=False) for q in out.glob('rst/*.rst')]
 if not checkpoints:raise RuntimeError('No final native checkpoint')
 cp=max(checkpoints,key=lambda c:c['time']);cp=checkpoint(cp['path'])
 import numpy as np
 data=np.memmap(cp['path'],mode='r',dtype='<f8',offset=cp['bytes']-cp['particles']*80,shape=(10,cp['particles']))
 if (not np.isfinite(data).all() or len(np.unique(data[9]))!=cp['particles'] or np.any(data[0]<=0)
     or not np.array_equal(data[9],data[9].astype(np.int64)) or np.any(data[9]<0) or np.any(data[9]>=5000000)
     or not np.array_equal(data[8],data[8].astype(np.int64)) or np.any(data[8]<0) or np.any(data[8]>=cp['blocks'])):
  raise RuntimeError('Final native particle audit failed')
 remaining=cp['blocks']*25*40**3
 with Path(cp['path']).open('rb') as f:
  f.seek(cp['bytes']-cp['particles']*80-32-remaining*8)
  while remaining:
   n=min(remaining,4000000);a=np.fromfile(f,dtype='<f8',count=n)
   if len(a)!=n or not np.isfinite(a).all():raise RuntimeError('Final native grid audit failed')
   remaining-=n
 complete=code==0 and reason is None and abs(cp['time']-150)<1e-10
 atomic(root/'FINAL.json',dict(passed=complete,checkpoint=cp,exit_code=code,stop_reason=reason,full_native_finite=True))
 if not complete:raise RuntimeError('Stopped before certified t=150; no automatic restart')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['validate','production','readiness','sampler']);p.add_argument('case',nargs='?');a=p.parse_args()
 if a.action=='validate':validate()
 elif a.action=='production':production()
 elif a.action=='sampler':sampler(a.case)
 else:ready();print('READY: tests and push verified; no job submitted')
