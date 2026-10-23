#!/usr/bin/env python3
"""Explicit future execution only. Preparation never calls these science actions."""
import argparse
import csv
import json
import os
import re
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from prepare import HERE,SESSION,sha,atomic
from telemetry_reader import StableSamplerReader
from workflow import (CASES, manifest, ready as workflow_ready, endpoint,
                      checkpoint_header, horizon_quality, fresh_persistence, startup_census)
from output_audit import smoke_diagnostics, audit_production
from checkpoint_io import checkpoint, audit_particles

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
 return manifest()['native_source_sha256']

def rows(path):
 with Path(path).open() as f:return list(csv.DictReader(f))

def initial_rows(path):
 with Path(path).open() as stream:
  lines=[stream.readline() for _ in range(4)]
 if any(not line.endswith('\n') for line in lines):return []
 return list(csv.DictReader(lines))

def startup_native(out,census):
 states=[checkpoint_header(p) for p in out.glob('rst/*.rst')]
 first=[state for state in states if state['time']==0 and state['cycle']==0]
 if len(first)!=1:raise RuntimeError('Missing unique full-size t0 native checkpoint')
 cp=checkpoint(first[0]['path'],hash_file=False)
 audit=audit_particles(cp)
 if audit['component_counts']!=census['initial_counts'] or cp['blocks']!=4936:
  raise RuntimeError('Native t0 particle census/mesh mismatch')
 if any(abs(a-b)>1.e-10*max(abs(b),1.) for a,b in zip(
        audit['component_rest_masses'],census['component_rest_masses'])):
  raise RuntimeError('Native t0 rest-mass diagnostics mismatch')
 profile=out/'gi_profile_M076_two_clump_s10.txt'
 if not profile.is_file() or profile.stat().st_size==0:
  raise RuntimeError('Missing native t0 GI model profile')
 return dict(checkpoint=cp,particle_audit=audit,profile_sha256=sha(profile))

def require_gpu(partition,nodes):
 if (os.environ.get('SLURM_JOB_PARTITION')!=partition
     or int(os.environ.get('SLURM_NNODES','0'))!=nodes
     or int(os.environ.get('SLURM_NTASKS','0'))!=nodes
     or os.environ.get('SLURM_JOB_ACCOUNT','').upper()!='AST21006'):
  raise RuntimeError(f'Requires allocated {nodes}-node {partition} job')

def run_case(root,label,kind,deck=None,restart=None,extra=(),limit=3300,
             target_time=None,target_cycle=None):
 case=root/label;case.mkdir(parents=True,exist_ok=False)
 out=case/'out';out.mkdir()
 remain=job_end()-int(time.time())-180
 seconds=min(limit,remain)
 if seconds<180:raise RuntimeError('Validation allocation reserve reached')
 args=['ibrun','-n','4',str(SESSION/f'build/{kind}/src/athena'),'--kokkos-map-device-id-by=mpi_rank']
 args+=['-r',str(restart)] if restart else ['-i',str(deck)]
 args+=['-d',str(out),'-t',f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}',
        'time/stop_file='+str(out/'REQUEST_STOP'),*extra]
 atomic(case/'COMMAND.json',dict(command=args,cwd=str(out),job_id=os.environ['SLURM_JOB_ID'],manifest=manifest()))
 with (case/'athena.log').open('w') as f:
  p=subprocess.Popen(args,cwd=out,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=seconds+90)
  except subprocess.TimeoutExpired:
   (out/'REQUEST_STOP').touch()
   try:p.wait(timeout=60)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=30)
   raise RuntimeError('Native validation timeout; outputs retained')
 if code:raise RuntimeError(f'Native validation {label} failed: {code}')
 final=endpoint(out,target_time=target_time,target_cycle=target_cycle)
 atomic(case/'RESULT.json',dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],
  exit_code=code,endpoint=final,endpoint_checked=True))
 return out

def validation_case(root,label,restart=None):
 kind,deck,extra,limit=CASES[label]
 return run_case(root,label,kind,HERE/'inputs'/deck,restart=restart,extra=extra,limit=limit,
  target_cycle=2 if label=='compact_gi' else None,
  target_time=None if label=='compact_gi' else 40. if kind=='os' else .25)

def validate():
 require_gpu('gh-dev',4)
 root=SESSION/'runtime'/('validation_'+os.environ['SLURM_JOB_ID']);root.mkdir(parents=True,exist_ok=False)
 try:
  frozen=manifest()
  smoke=validation_case(root,'compact_gi')
  smoke_result=smoke_diagnostics(smoke)
  ref=validation_case(root,'os_reference')
  pub=horizon_quality(rows(ref/'os_ah.ah_0.csv'))
  if len(pub)<10:raise RuntimeError('OS did not produce accepted horizon history')
  candidates=[]
  for cp in sorted(ref.glob('rst/*.rst')):
   state=checkpoint_header(cp)
   if state['time']<35.1 and state['deck'].get('fastflow',{}).get('surface_state_valid_0')=='true':
    candidates.append((state['time'],cp))
  if not candidates:raise RuntimeError('No mid-run accepted-shape checkpoint')
  tm,cp=max(candidates)
  resumed=validation_case(root,'os_bad_fallback_restart',restart=cp)
  all_restart=rows(resumed/'os_ah.ah_0.csv')
  found=horizon_quality(all_restart)
  if len(found)<3 or not 0<float(found[0]['state_time'])-tm<=0.1:
   raise RuntimeError('OS restart did not reacquire within 0.1 M')
  if all_restart[0]['published']=='1' or all_restart[0]['guess_valid']!='1':
   raise RuntimeError('Restored guess missing or incorrectly published immediately')
  fresh_persistence(resumed/'os_ah.horizon_consumer_0.csv')
  reference={int(r['state_cycle']):r for r in pub}
  errors=[]
  for r in found:
   a=reference.get(int(r['state_cycle']))
   if a is None or abs(float(a['state_time'])-float(r['state_time']))>1.e-10:
    raise RuntimeError('OS restart lacks matching uninterrupted accepted state')
   errors.append(max(abs(float(r['mass'])/float(a['mass'])-1),abs(float(r['spin'])-float(a['spin']))))
  if not errors or max(errors)>0.005:raise RuntimeError('OS uninterrupted/restart AH mismatch')
  if abs(float(pub[-1]['mass'])-1)>0.05:raise RuntimeError('OS late mass differs from M=1 by >5%')
  spins=[]
  for label in ('spin_coarse','spin_fine'):
   out=validation_case(root,label)
   a=horizon_quality(rows(out/'spin_ah.ah_0.csv'))
   m,s=float(a[-1]['mass']),float(a[-1]['spin'])
   if abs(m-1)>0.02 or abs(s-0.5)>0.02:raise RuntimeError('Spin calibration failed')
   spins.append((m,s))
  if abs(spins[0][0]/spins[1][0]-1)>0.01 or abs(spins[0][1]-spins[1][1])>0.01:
   raise RuntimeError('Angular refinement comparison failed')
  if manifest()!=frozen:raise RuntimeError('Source, inputs, tools or executables changed during validation')
  artifacts={}
  for label in CASES:
   case=root/label
   # Retain checksums of the actual native diagnostics, commands, and logs.
   paths=[case/'COMMAND.json',case/'RESULT.json',case/'athena.log']
   paths+=list((case/'out').glob('*.csv'))+list((case/'out').glob('*.hst'))
   for p in paths:artifacts[str(p.relative_to(root))]=sha(p)
  receipt=dict(passed=True,manifest=frozen,native_source_sha256=frozen['native_source_sha256'],
   production_input_sha256=sha(HERE/'inputs/production.athinput'),
   validation_job=os.environ['SLURM_JOB_ID'],outputs=str(root),checkpoint=str(cp),checkpoint_sha256=sha(cp),
   artifacts=artifacts,smoke_diagnostics=smoke_result,
   os_restart_max_error=max(errors),spin_measurements=spins,production_authorized=False,
   full_size_memory_certified=False,poor_fallback_restart_passed=True,
   three_fresh_hits_checked=True,original_native_executable_replayed=False)
  atomic(root/'PASS.json',receipt);atomic(SESSION/'evidence/VALIDATION_PASS.json',receipt)
 except Exception as e:
  atomic(root/'FAILURE.json',dict(error=repr(e),passed=False,outputs_retained=True));raise

def ready():
 return workflow_ready()['native_source_sha256']

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
 require_gpu('gh',16)
 root=SESSION/'runtime'/('production_'+os.environ['SLURM_JOB_ID']);root.mkdir(parents=True,exist_ok=False)
 frozen=workflow_ready()
 out=root/'out';out.mkdir();(root/'telemetry').mkdir()
 seconds=min(36*3600,job_end()-int(time.time()))-2400
 if seconds<600:raise RuntimeError('Insufficient production allocation remaining')
 args=['ibrun','-n','16',str(SESSION/'build/gi/src/athena'),'--kokkos-map-device-id-by=mpi_rank',
  '-i',str(HERE/'inputs/production.athinput'),'-d',str(out),'-t',f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}',
  'time/stop_file='+str(out/'REQUEST_STOP')]
 atomic(root/'COMMAND.json',dict(command=args,cwd=str(out),job_id=os.environ['SLURM_JOB_ID'],manifest=frozen,fresh_t0=True,no_planned_restarts=True))
 sampling=['srun','--overlap','--nodes=16','--ntasks=16','--ntasks-per-node=1','--cpus-per-task=1',sys.executable,str(HERE/'launch.py'),'sampler',str(root)]
 reader=StableSamplerReader();cfg={'telemetry_stale_seconds':120};reason=None;p=None
 telemetry_samples=0;startup_checked=False;startup_passed=False
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
   p=subprocess.Popen(args,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   bad_since=None; stop_sent=None; startup_deadline=time.time()+1800
   last_output_epoch=time.time();last_output_bytes=0
   while p.poll() is None:
    try:
     measures=reader(root/'telemetry',16,cfg)
     reason=reason or guard(measures);bad_since=None;telemetry_samples+=1
    except RuntimeError:
     bad_since=bad_since or time.time()
     if time.time()-bad_since>120:reason=reason or 'persistent_telemetry_failure'
    stats=out/'s11_orbit_gw.components.csv'
    if stats.exists():
     with stats.open('rb') as f:f.seek(max(0,stats.stat().st_size-4096));tail=f.read().decode().splitlines(keepends=True)
     for line in tail[-3:]:
      if not line.endswith('\n'):continue
      a=line.split(',')
      if len(a)==19 and a[0].isdigit() and int(a[3])>=5760:reason=reason or 'leaf_capacity_proxy'
     if not startup_checked:
      initial=initial_rows(stats)
      if len(initial)>=3:
       try:
        census=startup_census(initial[:3])
        native=startup_native(out,census)
        atomic(root/'FULL_SIZE_STARTUP.json',dict(passed=True,**census,
         native_t0=native,
         manifest=frozen,measured_per_node_telemetry=True,job_id=os.environ['SLURM_JOB_ID'],
         long_time_memory_certified=False))
        startup_passed=True
       except (ValueError,KeyError,RuntimeError):
        reason=reason or 'full_size_initial_census_or_mesh'
       startup_checked=True
    if not startup_checked and time.time()>startup_deadline:
     reason=reason or 'missing_initial_census'
    free=os.statvfs(out)
    if free.f_bavail*free.f_frsize<150*1024**3:reason=reason or 'disk_write_reserve'
    if time.time()>int(os.environ['SLURM_JOB_END_TIME'])-2400:reason=reason or 'wall_reserve'
    now=time.time()
    if now-last_output_epoch>=300:
     size=sum(q.stat().st_size for q in out.rglob('*') if q.is_file())
     with (root/'output_progress.jsonl').open('a') as stream:
      stream.write(json.dumps(dict(epoch=now,output_bytes=size,free_bytes=free.f_bavail*free.f_frsize,
       measured_bytes_per_second=(size-last_output_bytes)/(now-last_output_epoch)))+'\n')
     last_output_epoch=now;last_output_bytes=size
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
 checkpoints=[checkpoint(q,hash_file=False) for q in out.glob('rst/*.rst')]
 if not checkpoints:raise RuntimeError('No final native checkpoint')
 cp=max(checkpoints,key=lambda c:c['time']);cp=checkpoint(cp['path'])
 import numpy as np
 particle_audit=audit_particles(cp)
 remaining=cp['blocks']*25*40**3
 with Path(cp['path']).open('rb') as f:
  f.seek(cp['bytes']-cp['particles']*80-32-remaining*8)
  while remaining:
   n=min(remaining,4000000);a=np.fromfile(f,dtype='<f8',count=n)
   if len(a)!=n or not np.isfinite(a).all():raise RuntimeError('Final native grid audit failed')
   remaining-=n
 coverage=audit_production(out,checkpoints)
 atomic(root/'OUTPUT_COVERAGE.json',coverage)
 complete=production_complete(code,reason,cp,startup_passed,telemetry_samples,coverage,manifest()==frozen)
 total_bytes=sum(q.stat().st_size for q in out.rglob('*') if q.is_file())
 atomic(root/'FINAL.json',dict(passed=complete,checkpoint=cp,exit_code=code,stop_reason=reason,
  manifest=frozen,full_native_finite=True,particle_audit=particle_audit,startup_passed=startup_passed,
  runtime_telemetry_samples=telemetry_samples,output_coverage_passed=coverage['passed'],
  output_bytes=total_bytes,scientific_analysis_pending=True))
 if not complete:raise RuntimeError('Production completion checks failed; inspect retained evidence; no automatic restart')

def production_complete(code,reason,cp,startup_passed,telemetry_samples,coverage,unchanged):
 return (code==0 and reason is None and abs(cp['time']-150)<1e-10
         and startup_passed and telemetry_samples>0 and coverage.get('passed') is True and unchanged)


def record_accounting(root):
 job=os.environ.get('SLURM_JOB_ID','')
 try:
  result=subprocess.run(['sacct','-n','-P','-j',job,'--format=JobIDRaw,State,ElapsedRaw,AllocNodes,TotalCPU'],
   text=True,capture_output=True,timeout=15)
  atomic(root/'ACCOUNTING.json',dict(job_id=job,epoch=time.time(),returncode=result.returncode,
   stdout=result.stdout,stderr=result.stderr,final_billing_verified=False))
 except (OSError,subprocess.TimeoutExpired) as exc:
  atomic(root/'ACCOUNTING.json',dict(job_id=job,error=repr(exc),final_billing_verified=False))


def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('action',choices=['validate','production','readiness','sampler'])
 parser.add_argument('case',nargs='?');args=parser.parse_args()
 if args.action=='sampler':sampler(args.case);return
 if args.action=='readiness':ready();print('READY: tests and push verified; no job submitted');return
 stage='validation' if args.action=='validate' else 'production'
 root=SESSION/'runtime'/(stage+'_'+os.environ.get('SLURM_JOB_ID','unknown'))
 try:
  validate() if args.action=='validate' else production()
 except Exception as exc:
  if root.is_dir() and not (root/'FAILURE.json').exists():
   atomic(root/'FAILURE.json',dict(passed=False,error=repr(exc),outputs_retained=True))
  raise
 finally:
  if root.is_dir():
   record_accounting(root)
   with (SESSION/'REPORT.md').open('a') as stream:
    stream.write(f"\nSession 11 {stage} job {os.environ.get('SLURM_JOB_ID','unknown')}: "
                 f"evidence and retained outputs at `{root}`. "
                 "Inspect PASS/FINAL/FAILURE and OUTPUT_COVERAGE; ACCOUNTING may still be provisional.\n")

if __name__=='__main__':main()
