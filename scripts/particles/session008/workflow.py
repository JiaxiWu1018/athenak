#!/usr/bin/env python3
"""Finite, locked AMD workflow. All production successors are ordinary Slurm jobs."""
import argparse,contextlib,fcntl,json,os
from pathlib import Path
import subprocess,time
from runtime import ROOT,atomic,checkpoint,digest,retain,seal

CONTROL=ROOT/'control'
TERMINAL={'complete_t12','segment_cap','user_stop','emergency_cancel','resource_limit','configuration_failure','numerical_failure','scheduler_failure','no_progress'}

@contextlib.contextmanager
def locked():
 CONTROL.mkdir(parents=True,exist_ok=True)
 with (CONTROL/'workflow.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);yield

def readstate():return json.loads((CONTROL/'state.json').read_text())
def save(s):s['updated_utc']=time.time();atomic(CONTROL/'state.json',s)
def config():return json.loads((CONTROL/'config.json').read_text())

def bindings():
 c=config()
 if digest(ROOT/'inputs/gi_cluster_s8.athinput')!=c['input_sha256']:raise RuntimeError('input changed')
 if subprocess.check_output(['git','-C',str(ROOT/'athenak'),'rev-parse','HEAD'],text=True).strip()!=c['compiled_source_commit']:raise RuntimeError('compiled source revision changed')
 for name,h in c['script_hashes'].items():
  if digest(ROOT/'scripts'/name)!=h:raise RuntimeError('workflow script changed: '+name)
 exe=ROOT/'build/src/athena'
 if digest(exe)!=(ROOT/'evidence/executable.sha256').read_text().split()[0]:raise RuntimeError('executable changed')

def init():
 if (CONTROL/'config.json').exists():raise RuntimeError('configuration already frozen')
 hashes={p.name:digest(p) for p in sorted((ROOT/'scripts').iterdir()) if p.is_file()}
 c=dict(campaign_id='jeans8_20261002',compiled_source_commit='0c0a5a9bd47fa10ed8732ea12f1a53cd591b3f69',
 input_sha256=digest(ROOT/'inputs/gi_cluster_s8.athinput'),script_hashes=hashes,target_time=12,
 max_segments=3,nodes=3,node_hour_cap=48,maximum_reserved_node_hours=44.5,
 checkpoint_keep=3,campaign_storage_cap_bytes=int(1.25*1024**4),anta_storage_cap_bytes=1024**4,
 archive_root='/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002',automatic_retries=0)
 atomic(CONTROL/'config.json',c)
 build=int((CONTROL/'build_submission.txt').read_text().strip())
 save(dict(campaign_id=c['campaign_id'],status='build_submitted',stop_requested=False,segments_completed=0,time=0,
 jobs=[dict(name='build',id=build,nodes=1,max_wall_hours=.5)],pending_submission=None))

def recover(name):
 # Close sbatch-accepted/state-save crash window using campaign-unique job names.
 matches=[]
 out=subprocess.check_output(['sacct','-X','-n','-P','-S','now-2days','-u','jiaxiwu','--format=JobIDRaw,JobName%80'],text=True)
 for line in out.splitlines():
  row=line.split('|')
  if len(row)>=2 and row[1].strip()==name:matches.append(int(row[0]))
 out=subprocess.check_output(['squeue','-h','-u','jiaxiwu','-o','%i|%j'],text=True)
 for line in out.splitlines():
  row=line.split('|')
  if len(row)==2 and row[1]==name:matches.append(int(row[0]))
 unique=set(matches)
 if len(unique)>1:raise RuntimeError('ambiguous duplicate submission')
 return next(iter(unique)) if unique else None

def submit_one(s,name,script,args,nodes,wall,dependency):
 old=next((j for j in s['jobs'] if j['name']==name),None)
 if old:return old['id']
 jobname='jn8_20261002_'+name
 pending=s.get('pending_submission')
 if pending and pending['name']!=name:raise RuntimeError('unresolved submission window')
 jid=recover(jobname) if pending else None
 if jid is None:
  s['pending_submission']=dict(name=name,job_name=jobname);save(s)
  cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes','--job-name='+jobname,
       '--output='+str(ROOT/'logs'/('%j_'+name+'.log')),'--dependency='+dependency,
       str(ROOT/'scripts'/script)]+list(map(str,args))
  out=subprocess.check_output(cmd,text=True).strip()
  jid=int(out.split(';')[0])
 s['jobs'].append(dict(name=name,id=jid,nodes=nodes,max_wall_hours=wall,dependency=dependency))
 s['pending_submission']=None;save(s)
 return jid

def submit():
 s=readstate()
 if s['status'] in TERMINAL or s['stop_requested']:raise RuntimeError('sticky terminal/stop intent')
 if s.get('chain_submitted'):print(json.dumps(s,indent=2));return
 build=s['jobs'][0]['id']
 gate=submit_one(s,'gate','amd_gate.sbatch',[],3,2,'afterok:'+str(build))
 dep=submit_one(s,'inspect0','amd_inspect.sbatch',[0,gate],1,.5,'afterany:'+str(gate))
 for i in range(1,4):
  seg=submit_one(s,'segment'+str(i),'amd_segment.sbatch',[i],3,4,'afterok:'+str(dep))
  dep=submit_one(s,'inspect'+str(i),'amd_inspect.sbatch',[i,seg],1,.5,'afterany:'+str(seg))
 exposure=sum(j['nodes']*j['max_wall_hours'] for j in s['jobs'])
 if exposure>48:raise RuntimeError('resource exposure exceeds approval')
 s['chain_submitted']=True;s['maximum_reserved_node_hours']=exposure;s['status']='gates_queued';save(s)
 print(json.dumps(s,indent=2))

def gate_receipt():
 bindings();checks=json.loads((ROOT/'evidence/initial_validation.json').read_text())['checks']
 if not all(checks.values()):raise RuntimeError('initial gate failed')
 if not (ROOT/'evidence/restart_validation.json').exists():raise RuntimeError('restart gate missing')
 peaks={}
 for name in ('gate_reference','gate_split','gate_restart','gate_clean_stop','gate_output'):
  run=ROOT/'runs'/name;seen={}
  for f in run.glob('vram_*.jsonl'):
   for line in f.read_text().splitlines():
    r=json.loads(line);key=r['host']+'/'+r['card'];seen[key]=max(seen.get(key,0),r['fraction'])
  if len(seen)!=12 or max(seen.values())>=.85:raise RuntimeError('missing/allocation memory gate failed '+name)
  peaks[name]=seen
  log=(run/'run.log').read_text(errors='replace')
  if '### FATAL ERROR' in log or '[conservation OK]' not in log:raise RuntimeError('numerical/particle ledger failure')
 run=ROOT/'runs/gate_output'
 required=['*.part.vtk','*.hst','*.cbin']
 for plane in ('xy','xz','yz'):
  required += ['*.'+v+'_'+plane+'.*.bin' for v in ('z4c','con','tmunu','weyl')]
 required += ['*.'+v+'.*.bin' for v in ('z4c3d','con3d','tmunu3d','weyl3d','E3d')]
 for pattern in required:
  if not list(run.rglob(pattern)):raise RuntimeError('missing output '+pattern)
 for part in ('real','imag'):
  for r in (40,50,60,70):
   p=run/'waveforms'/('rpsi4_'+part+'_'+str(r).zfill(4)+'.txt')
   lines=[x for x in p.read_text().splitlines() if x.strip() and not x.startswith('#')]
   if not lines:raise RuntimeError('empty waveform')
   import math
   if any(len(x.split())!=78 or any(not math.isfinite(float(v)) for v in x.split()) for x in lines):raise RuntimeError('invalid raw multipoles')
 latest=json.loads((CONTROL/'latest_checkpoint.json').read_text())
 atomic(ROOT/'evidence/gate_receipt.json',dict(passed=True,input_sha256=config()['input_sha256'],
 executable_sha256=digest(ROOT/'build/src/athena'),script_hashes=config()['script_hashes'],
 memory_peaks=peaks,checkpoint=latest,accepted_horizon_fixture='reused Session 007 6531/6533 evidence; not a new AMD acceptance fixture',utc=time.time()))

def cancel_future(s):
 current=int(os.environ.get('SLURM_JOB_ID','-1'))
 ids=[str(j['id']) for j in s['jobs'] if j['id']!=current]
 if ids:subprocess.run(['scancel']+ids,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def terminal(s,status,detail):
 s['status']=status;s['detail']=detail;save(s);cancel_future(s)
 raise SystemExit(42)

def scheduler_status(jid):
 for _ in range(12):
  out=subprocess.check_output(['sacct','-X','-n','-P','-j',str(jid),'--format=JobIDRaw,State%30,ExitCode,ElapsedRaw,AllocNodes'],text=True)
  rows=[x.split('|') for x in out.splitlines() if x.split('|')[0]==str(jid)]
  if rows and rows[0][1].strip() not in ('RUNNING','COMPLETING','PENDING'):return rows[0]
  time.sleep(5)
 raise RuntimeError('scheduler accounting unavailable')

def inspect(segment,jid):
 s=readstate()
 if s['status'] in TERMINAL:cancel_future(s);raise SystemExit(42)
 job=scheduler_status(jid);s.setdefault('accounting',{})[str(jid)]=job;save(s)
 if s['stop_requested'] or (CONTROL/'USER_STOP').exists():
  latest=CONTROL/'latest_checkpoint.json'
  if latest.exists():s['checkpoint']=json.loads(latest.read_text());s['time']=s['checkpoint']['time'];save(s)
  for run in (ROOT/'runs').glob('*'):
   if (run/'EXIT_CODE').exists() and not (run/'SEALED').exists():seal(run)
  terminal(s,'user_stop','sticky user request; last verified checkpoint retained')
 if job[1].strip()!='COMPLETED' or job[2]!='0:0':
  terminal(s,'configuration_failure' if segment==0 else 'scheduler_failure','predecessor '+str(job))
 bindings()
 if segment==0:
  receipt=json.loads((ROOT/'evidence/gate_receipt.json').read_text())
  if not receipt['passed'] or receipt['script_hashes']!=config()['script_hashes']:terminal(s,'configuration_failure','gate receipt binding')
  row=checkpoint(receipt['checkpoint']['path'])
  s['gates_passed']=True
 else:
  run=ROOT/'runs'/('segment_'+str(segment).zfill(2))
  if (run/'EXIT_CODE').read_text().strip()!='0':terminal(s,'numerical_failure','application exit')
  rows=[x for x in retain(ROOT,True,run) if str(run)+'/' in x['path']]
  if not rows:terminal(s,'no_progress','no verified segment checkpoint')
  row=max(rows,key=lambda x:x['cycle'])
  if row['time']<=s['time']:terminal(s,'no_progress','checkpoint time did not increase')
  text=(run/'run.log').read_text(errors='replace')
  if '### FATAL ERROR' in text or '[conservation OK]' not in text:terminal(s,'numerical_failure','final particle ledger or fatal')
  seal(run)
  s['segments_completed']=segment
 s['time']=row['time'];s['cycle']=row['cycle'];s['checkpoint']=row;save(s)
 if s['stop_requested'] or (CONTROL/'USER_STOP').exists():terminal(s,'user_stop','sticky user request')
 if (CONTROL/'RESOURCE_STOP.json').exists():terminal(s,'resource_limit',(CONTROL/'RESOURCE_STOP.json').read_text())
 if row['time']>=12-1e-10:terminal(s,'complete_t12','assessment endpoint reached')
 if segment==3:terminal(s,'segment_cap','three evolution jobs exhausted; review cost toward t50')
 s['status']='ready_segment_'+str(segment+1);save(s)

def permit(segment):
 bindings();s=readstate()
 if not s.get('gates_passed') or s['status']!='ready_segment_'+str(segment) or s['stop_requested'] or (CONTROL/'REQUEST_STOP').exists():raise RuntimeError('production gate/stop/segment failed')
 if int(os.environ['SLURM_NNODES'])!=3 or int(os.environ['SLURM_NTASKS'])!=12:raise RuntimeError('allocation changed')
 heartbeat=CONTROL/'ARCHIVE_HEARTBEAT'
 if not heartbeat.exists() or time.time()-float(heartbeat.read_text())>900:raise RuntimeError('Anta archival heartbeat absent/stale; preserve data and halt')
 row=checkpoint(s['checkpoint']['path'])
 if row['sha256']!=s['checkpoint']['sha256']:raise RuntimeError('restart checkpoint changed')
 s['status']='running_segment_'+str(segment);save(s)

def request_stop(emergency=False):
 s=readstate();s['stop_requested']=True;(CONTROL/'USER_STOP').touch();(CONTROL/'REQUEST_STOP').touch()
 s['status']='emergency_cancel' if emergency else 'stop_requested';save(s)
 if emergency:cancel_future(s)

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['init','submit','bindings','gate_receipt','permit','inspect','status','stop','cancel'])
 p.add_argument('--segment',type=int);p.add_argument('--predecessor',type=int);a=p.parse_args()
 if a.action=='bindings':bindings();return
 if a.action=='gate_receipt':gate_receipt();return
 with locked():
  if a.action=='init':init()
  elif a.action=='submit':submit()
  elif a.action=='permit':permit(a.segment)
  elif a.action=='inspect':inspect(a.segment,a.predecessor)
  elif a.action=='status':print(json.dumps(readstate(),indent=2))
  elif a.action=='stop':request_stop()
  else:request_stop(True)

if __name__=='__main__':
 try:main()
 except Exception as e:
  atomic(CONTROL/('ERROR_'+str(os.environ.get('SLURM_JOB_ID','interactive'))+'.json'),dict(error=str(e),utc=time.time()))
  if os.environ.get('SLURM_JOB_ID') and (CONTROL/'state.json').exists():
   with locked():
    s=readstate()
    if s['status'] not in TERMINAL:
     s['status']='configuration_failure';s['detail']=str(e);save(s);cancel_future(s)
  raise
