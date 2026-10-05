#!/usr/bin/env python3
"""Campaign-scoped durable finite AMD workflow; no automatic numerical retries."""
import argparse,contextlib,fcntl,json,math,os,subprocess,time
from pathlib import Path
from runtime import ROOT,atomic,checkpoint,digest,retain,seal
CONTROL=ROOT/'control'
TERMINAL={'complete_t400','complete_ringdown','calendar_cap','segment_cap','user_stop','emergency_cancel','resource_limit','configuration_failure','numerical_failure','scheduler_failure','no_progress'}
CLOSED={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
@contextlib.contextmanager
def locked():
 CONTROL.mkdir(parents=True,exist_ok=True)
 with (CONTROL/'workflow.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);yield

def readstate():return json.loads((CONTROL/'state.json').read_text())
def save(s):s['updated_utc']=time.time();atomic(CONTROL/'state.json',s)
def config():return json.loads((CONTROL/'config.json').read_text())
def budget(jobs):
 ids=','.join(str(int(j['id'])) for j in jobs)
 out=subprocess.check_output(['sacct','-X','-n','-P','-j',ids,'--format=JobIDRaw,State%30,ExitCode,ElapsedRaw,AllocNodes'],text=True)
 rows={r[0]:r for r in (s.split('|') for s in out.splitlines()) if r[0] in {str(j['id']) for j in jobs}}
 active=dict(s.split('|',1) for s in subprocess.check_output(['squeue','-h','-u','jiaxiwu','-o','%i|%T'],text=True).splitlines())
 actual=exposure=0.;unmeasured=[]
 for j in jobs:
  r=rows.get(str(j['id']))
  if r is None:
   if str(j['id']) not in active:raise RuntimeError('missing accounting '+str(j['id']))
   exposure+=j['nodes']*j['max_wall_hours'];unmeasured.append(j['id']);continue
  cost=float(r[3])*int(r[4])/3600;actual+=cost
  exposure+=cost if r[1].split()[0].rstrip('+') in CLOSED else j['nodes']*j['max_wall_hours']
 return dict(actual_node_hours=actual,maximum_possible_node_hours=exposure,accounting=rows,unmeasured_active=unmeasured)

def bindings(executables=True):
 c=config()
 if digest(ROOT/'inputs/gi_cluster_s9.athinput')!=c['input_sha256']:raise RuntimeError('canonical input changed')
 if subprocess.check_output(['git','-C',str(ROOT/'athenak'),'rev-parse','HEAD'],text=True).strip()!=c['compiled_source_commit']:raise RuntimeError('source changed')
 for n,h in c['script_hashes'].items():
  if digest(ROOT/'scripts'/n)!=h:raise RuntimeError('script changed '+n)
 if executables:
  for name in ('executable','wave_executable'):
   h=(ROOT/'evidence'/(name+'.sha256')).read_text().split()[0]
   exe=ROOT/('build' if name=='executable' else 'build_wave')/'src/athena'
   if digest(exe)!=h:raise RuntimeError('executable changed')

def init():
 if (CONTROL/'config.json').exists():raise RuntimeError('configuration already frozen')
 jid=int((CONTROL/'build_submission.txt').read_text())
 now=time.time();head=subprocess.check_output(['git','-C',str(ROOT/'athenak'),'rev-parse','HEAD'],text=True).strip()
 c=dict(campaign_id='jeans9_20261005',compiled_source_commit=head,input_sha256=digest(ROOT/'inputs/gi_cluster_s9.athinput'),
 script_hashes={p.name:digest(p) for p in (ROOT/'scripts').iterdir() if p.is_file()},target_time=400,M_ref=1,
 max_segments=90,nodes=12,ranks=48,node_hour_cap=10000,created_utc=now,deadline_utc=now+45*86400,
 checkpoint_keep=3,campaign_storage_cap_bytes=int(1.25*1024**4),anta_storage_cap_bytes=16*1024**4,
 archive_root='/data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005',automatic_retries=0)
 atomic(CONTROL/'config.json',c)
 save(dict(campaign_id=c['campaign_id'],status='build_submitted',stop_requested=False,segments_completed=0,time=0,jobs=[dict(name='build',id=jid,nodes=1,max_wall_hours=.5)],pending_submission=None))

def recover(name):
 matches=set()
 for cmd in (['sacct','-X','-n','-P','-S','now-45days','-u','jiaxiwu','--format=JobIDRaw,JobName%100'],['squeue','-h','-u','jiaxiwu','-o','%i|%j']):
  for line in subprocess.check_output(cmd,text=True).splitlines():
   r=line.split('|')
   if len(r)>=2 and r[1].strip()==name:matches.add(int(r[0]))
 if len(matches)>1:raise RuntimeError('duplicate campaign job name')
 return next(iter(matches)) if matches else None

def submit_one(s,name,script,args,nodes,wall,dependency):
 old=next((j for j in s['jobs'] if j['name']==name),None)
 if old:return old['id']
 pending=s.get('pending_submission');jobname='jn9_20261005_'+name
 if pending and pending['name']!=name:raise RuntimeError('unresolved submission window')
 jid=recover(jobname) if pending else None
 if jid is None:
  if budget(s['jobs'])['maximum_possible_node_hours']+nodes*wall>10000:raise RuntimeError('allocation would exceed hard node-hour cap')
  s['pending_submission']=dict(name=name,job_name=jobname);save(s)
  cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes','--job-name='+jobname,'--output='+str(ROOT/'logs'/('%j_'+name+'.log'))]
  if dependency:cmd+=['--dependency='+dependency]
  cmd+=[str(ROOT/'scripts'/script)]+list(map(str,args))
  jid=int(subprocess.check_output(cmd,text=True).strip().split(';')[0])
 s['jobs'].append(dict(name=name,id=jid,nodes=nodes,max_wall_hours=wall,dependency=dependency));s['pending_submission']=None;save(s);return jid

def submit():
 s=readstate()
 if s['status'] in TERMINAL or s['stop_requested']:raise RuntimeError('sticky terminal/stop')
 if s.get('gates_submitted'):return
 wave=submit_one(s,'wave_gate','amd_wave_gate.sbatch',[],3,2,'afterok:'+str(s['jobs'][0]['id']))
 gate=submit_one(s,'gate','amd_gate.sbatch',[],12,4,'afterok:'+str(wave))
 submit_one(s,'inspect0','amd_inspect.sbatch',[0,gate],1,.5,'afterany:'+str(gate))
 s['gates_submitted']=True;s['status']='gates_queued';save(s)

def gate_receipt():
 bindings()
 checks=json.loads((ROOT/'evidence/initial_validation.json').read_text())['checks']
 if not all(checks.values()):raise RuntimeError('initialization failed')
 if not json.loads((ROOT/'evidence/wave_gate.json').read_text())['passed']:raise RuntimeError('wave propagation gate failed')
 if not (ROOT/'evidence/restart_validation.json').exists():raise RuntimeError('restart receipt absent')
 peaks={}
 for name in ('gate_reference','gate_split','gate_restart','gate_clean_stop','gate_output'):
  run=ROOT/'runs'/name;seen={}
  for p in run.glob('vram_*.jsonl'):
   for line in p.read_text().splitlines():
    r=json.loads(line);key=r['host']+'/'+r['card'];seen[key]=max(seen.get(key,0),r['fraction'])
  if len(seen)!=48 or max(seen.values())>=.85:raise RuntimeError('48-GPU memory gate failed '+name)
  peaks[name]=seen
  text=(run/'run.log').read_text(errors='replace')
  if '### FATAL ERROR' in text or '[conservation OK]' not in text:raise RuntimeError('numerical/particle conservation failed')
 run=ROOT/'runs/gate_output'
 required=['*.part.vtk','*.hst','*.cbin']
 for plane in ('xy','xz','yz'):required+=['*.'+v+'_'+plane+'.*.bin' for v in ('z4c','con','tmunu','weyl')]
 required+=['*.'+v+'.*.bin' for v in ('z4c3d','con3d','tmunu3d','weyl3d','E3d')]
 for pattern in required:
  if not list(run.rglob(pattern)):raise RuntimeError('missing output '+pattern)
 for part in ('real','imag'):
  p=run/'waveforms'/('rpsi4_'+part+'_0050.txt')
  rows=[x.split() for x in p.read_text().splitlines() if x.strip() and not x.startswith('#')]
  if not rows or any(len(r)!=78 or any(not math.isfinite(float(x)) for x in r) for r in rows):raise RuntimeError('invalid raw complex multipoles')
 latest=json.loads((CONTROL/'latest_checkpoint.json').read_text())
 if latest['header_bytes']<=16384:raise RuntimeError('restart-header repair not exercised by a relevant large header')
 atomic(ROOT/'evidence/gate_receipt.json',dict(passed=True,input_sha256=config()['input_sha256'],script_hashes=config()['script_hashes'],memory_peaks=peaks,checkpoint=json.loads((CONTROL/'latest_checkpoint.json').read_text()),utc=time.time()))

def cancel_future(s):
 current=int(os.environ.get('SLURM_JOB_ID','-1'))
 ids=[str(j['id']) for j in s['jobs'] if j['id']!=current]
 if ids:subprocess.run(['scancel']+ids,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def terminal(s,status,detail):
 s['status']=status;s['detail']=detail;save(s);cancel_future(s)
 return

def scheduler_status(jid):
 for _ in range(12):
  out=subprocess.check_output(['sacct','-X','-n','-P','-j',str(jid),'--format=JobIDRaw,State%30,ExitCode,ElapsedRaw,AllocNodes'],text=True)
  rows=[x.split('|') for x in out.splitlines() if x.split('|')[0]==str(jid)]
  if rows and rows[0][1].split()[0].rstrip('+') in CLOSED:return rows[0]
  time.sleep(5)
 raise RuntimeError('scheduler accounting unavailable')

def inspect(segment,jid):
 s=readstate()
 if s['status'] in TERMINAL:cancel_future(s);return
 job=scheduler_status(jid);s.setdefault('accounting',{})[str(jid)]=job;save(s)
 # Seal stopped/failure runs as evidence; checkpoints are layout/hash verified separately.
 for run in (ROOT/'runs').glob('*'):
  if (run/'EXIT_CODE').exists() and not (run/'SEALED').exists():seal(run)
 if s['stop_requested'] or (CONTROL/'USER_STOP').exists():terminal(s,'user_stop','sticky user stop; preserved evidence');return
 if job[1].strip()!='COMPLETED' or job[2]!='0:0':terminal(s,'configuration_failure' if segment==0 else 'scheduler_failure','predecessor '+str(job)+'; no automatic retry');return
 bindings()
 if segment==0:
  receipt=json.loads((ROOT/'evidence/gate_receipt.json').read_text())
  if not receipt['passed'] or receipt['script_hashes']!=config()['script_hashes']:raise RuntimeError('gate binding failed')
  row=checkpoint(receipt['checkpoint']['path']);s['gates_passed']=True
 else:
  run=ROOT/'runs'/('segment_'+str(segment).zfill(3))
  rows=[r for r in retain(ROOT,True,run) if r['path'].startswith(str(run)+'/')]
  if not rows:terminal(s,'no_progress','no verified segment checkpoint');return
  row=max(rows,key=lambda r:r['cycle'])
  if row['time']<=s['time']:terminal(s,'no_progress','checkpoint time unchanged');return
  text=(run/'run.log').read_text(errors='replace')
  if '### FATAL ERROR' in text or '[conservation OK]' not in text:terminal(s,'numerical_failure','fatal/particle conservation failure');return
  s['segments_completed']=segment
 s['time']=row['time'];s['cycle']=row['cycle'];s['checkpoint']=row;s['resources']=budget(s['jobs']);save(s)
 if (CONTROL/'RESOURCE_STOP.json').exists() or (CONTROL/'REQUEST_STOP').exists():terminal(s,'resource_limit','clean resource/archive stop; review evidence');return
 if row['time']>=400-1e-9:terminal(s,'complete_t400','hard physical endpoint; scientific outcome requires analysis');return
 if time.time()+13*3600>config()['deadline_utc']:terminal(s,'calendar_cap','insufficient time for a bounded segment and inspection before45-day cap');return
 # A scientific receipt may only be written by the validated Anta ringdown analysis.
 science=CONTROL/'ringdown_stop_receipt.json'
 if science.exists():
  r=json.loads(science.read_text())
  if all(r.get(k) is True for k in ('common_encloses_both','strict_common_accepted','ringdown_usable','gaps_checked','frequency_band_valid')) and row['time']>=r['wave_peak_time']+100:
   terminal(s,'complete_ringdown','approved common-horizon/ringdown-plus100M rule; inspect receipt');return
 if segment>=config()['max_segments']:terminal(s,'segment_cap','90 finite segments exhausted');return
 # Reserve both successor and its inspector BEFORE making either submission.
 b=budget(s['jobs'])
 if b['maximum_possible_node_hours']+145>10000:terminal(s,'resource_limit','next 12-hour allocation and inspection would cross10000 raw nodeh');return
 nextseg=segment+1
 target=12. if row['time']<12-1e-9 else min(400.,50.*(math.floor((row['time']+1e-8)/50.)+1))
 s['segment_target']=target;s['status']='ready_segment_'+str(nextseg);save(s)
 seg=submit_one(s,'segment'+str(nextseg),'amd_segment.sbatch',[nextseg],12,12,'afterok:'+os.environ['SLURM_JOB_ID'])
 submit_one(s,'inspect'+str(nextseg),'amd_inspect.sbatch',[nextseg,seg],1,.5,'afterany:'+str(seg))
 s['status']='ready_segment_'+str(nextseg);save(s)

def permit(segment):
 bindings();s=readstate()
 if not s.get('gates_passed') or s['status']!='ready_segment_'+str(segment) or s['stop_requested'] or (CONTROL/'REQUEST_STOP').exists():raise RuntimeError('gate/stop/segment failed')
 if int(os.environ['SLURM_NNODES'])!=12 or int(os.environ['SLURM_NTASKS'])!=48:raise RuntimeError('allocation changed')
 if time.time()+12*3600>config()['deadline_utc'] or budget(s['jobs'])['maximum_possible_node_hours']>10000:raise RuntimeError('resource/calendar cap')
 p=CONTROL/'ARCHIVE_HEARTBEAT'
 if not p.exists() or time.time()-float(p.read_text())>900:raise RuntimeError('archive heartbeat absent/stale')
 row=checkpoint(s['checkpoint']['path'])
 if row['sha256']!=s['checkpoint']['sha256']:raise RuntimeError('checkpoint hash changed')
 s['status']='running_segment_'+str(segment);save(s)

def request_stop(emergency=False):
 s=readstate();s['stop_requested']=True;(CONTROL/'USER_STOP').touch();(CONTROL/'REQUEST_STOP').touch()
 if s['status'] not in TERMINAL:s['status']='emergency_cancel' if emergency else 'stop_requested'
 save(s)
 if emergency:cancel_future(s)

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['init','submit','bindings','gate_receipt','permit','inspect','status','stop','cancel']);p.add_argument('--segment',type=int);p.add_argument('--predecessor',type=int);a=p.parse_args()
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
    if s['status'] not in TERMINAL:terminal(s,'configuration_failure',str(e))
  raise
