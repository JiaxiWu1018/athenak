#!/usr/bin/env python3
"""Explicitly authorized, finite continuation after the reviewed MPI startup failure."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import workflow as w
from runtime import ROOT,atomic,checkpoint,digest,retain,seal

CONTROL=ROOT/'control'
CONFIG=CONTROL/'recovery_20261005.json'
HERE=ROOT/'scripts/recovery_20261005'
ALLOWED={'k005-002','k005-003','k005-004','k005-005','k005-007','k003-007'}
ORIGINAL_TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
NEW_JOBS=[('preflight','amd_preflight.sbatch',[],3,1/6),
          ('continuation1','amd_continue.sbatch',[1],3,4),
          ('check1','amd_check.sbatch',[1],1,.5),
          ('continuation2','amd_continue.sbatch',[2],3,4),
          ('check2','amd_check.sbatch',[2],1,.5)]

def budget(jobs):
 ids=','.join(str(int(j['id'])) for j in jobs)
 out=subprocess.check_output(['sacct','-X','-n','-P','-j',ids,'--format=JobIDRaw,State%30,ExitCode,ElapsedRaw,AllocNodes'],text=True)
 rows={r[0]:r for r in (line.split('|') for line in out.splitlines()) if r[0] in {str(j['id']) for j in jobs}}
 missing=[j for j in jobs if str(j['id']) not in rows]
 active={}
 if missing:
  queue=subprocess.check_output(['squeue','-h','-u','jiaxiwu','-o','%i|%T'],text=True)
  active=dict(line.split('|',1) for line in queue.splitlines())
 actual=0.;exposure=0.;unmeasured=[]
 for j in jobs:
  row=rows.get(str(j['id']))
  if row is None:
   if str(j['id']) not in active:raise RuntimeError('missing campaign accounting '+str(j['id']))
   # Accounting can lag submission. Reserve the full allocation for a listed
   # active job; its elapsed usage is explicitly marked as unmeasured.
   exposure+=j['nodes']*j['max_wall_hours'];unmeasured.append(j['id']);continue
  cost=float(row[3])*int(row[4])/3600;actual+=cost
  closed=row[1].split()[0].rstrip('+') in ORIGINAL_TERMINAL
  exposure+=cost if closed else j['nodes']*j['max_wall_hours']
 return dict(actual_node_hours=actual,maximum_possible_node_hours=exposure,accounting=rows,active_jobs_with_unmeasured_elapsed=unmeasured)

def guard():
 s=w.readstate()
 if s['stop_requested'] or (CONTROL/'USER_STOP').exists():raise RuntimeError('sticky user stop')
 if (CONTROL/'RESOURCE_STOP.json').exists():raise RuntimeError('resource stop needs review')
 w.bindings()
 c=json.loads(CONFIG.read_text())
 for name,h in c['script_hashes'].items():
  if digest(HERE/name)!=h:raise RuntimeError('recovery script changed '+name)
 if w.config()['target_time']!=12:raise RuntimeError('physical endpoint changed')
 b=budget(s['jobs'])
 if b['maximum_possible_node_hours']>48+1e-9:raise RuntimeError('48 node-hour cap exceeded')
 return s,c,b

def allocation():
 if int(os.environ['SLURM_NNODES'])!=3 or int(os.environ['SLURM_NTASKS'])!=12:raise RuntimeError('allocation changed')
 hosts=set(subprocess.check_output(['scontrol','show','hostnames',os.environ['SLURM_JOB_NODELIST']],text=True).split())
 if len(hosts)!=3 or not hosts<=ALLOWED:raise RuntimeError('allocation outside known-working nodes')
 return sorted(hosts)

def heartbeat():
 p=CONTROL/'ARCHIVE_HEARTBEAT'
 if not p.exists() or time.time()-float(p.read_text())>900:raise RuntimeError('archive heartbeat absent/stale')
 if (CONTROL/'REQUEST_STOP').exists():raise RuntimeError('clean stop requested')

def setup():
 s=w.readstate()
 if CONFIG.exists():raise RuntimeError('recovery already configured; use recorded submission/state')
 if s['status']!='scheduler_failure' or not s.get('gates_passed') or s['segments_completed']!=2:raise RuntimeError('unexpected prior state')
 if s['stop_requested'] or (CONTROL/'USER_STOP').exists() or (CONTROL/'RESOURCE_STOP.json').exists():raise RuntimeError('user/resource stop requires review')
 log=(ROOT/'runs/segment_03/run.log').read_text()
 if 'selected pml ob1' not in log or 'selected pml ucx' not in log or 'MPI_INIT has failed' not in log:raise RuntimeError('not the reviewed communication failure')
 w.bindings();b=budget(s['jobs'])
 if b['maximum_possible_node_hours']!=b['actual_node_hours']:raise RuntimeError('previous jobs still active')
 extra=sum(j[3]*j[4] for j in NEW_JOBS)
 if b['actual_node_hours']+extra>48:raise RuntimeError('continuation exposure exceeds cap')
 (ROOT/'evidence/recovery_20261005').mkdir(exist_ok=True)
 atomic(ROOT/'evidence/recovery_20261005/prior_state.json',s)
 atomic(CONFIG,dict(authorization='User requested continuing the remaining work on 2026-10-05; reviewed MPI startup failure.',
  target_time=12,node_hour_cap=48,max_continuations=2,additional_max_node_hours=extra,
  baseline=b,allowed_nodes=sorted(ALLOWED),script_hashes={p.name:digest(p) for p in HERE.iterdir() if p.is_file()},
  old_checkpoint=s['checkpoint'],created_utc=time.time()))
 # Only the expired automatic archive stop is cleared; human/resource stops stay sticky.
 for name in ('ARCHIVE_ERROR','REQUEST_STOP'):
  p=CONTROL/name
  if p.exists():
   (ROOT/'evidence/recovery_20261005'/('prior_'+name)).write_bytes(p.read_bytes());p.unlink()
 s['status']='recovery_prepared';s['recovery_authorized']=True;s['recovery_continuations_completed']=0
 s['maximum_possible_total_node_hours']=b['actual_node_hours']+extra;w.save(s)

def submit():
 s,c,b=guard();heartbeat()
 if s.get('recovery_chain_submitted'):return
 if s['status']!='recovery_prepared':raise RuntimeError('unexpected recovery submission state')
 dep=None
 for name,script,args,nodes,wall in NEW_JOBS:
  unique='recovery_20261005_'+name
  old=next((j for j in s['jobs'] if j['name']==unique),None)
  if old:dep=old['id'];continue
  pending=s.get('pending_submission')
  jobname='jn8_20261002_'+unique
  jid=w.recover(jobname) if pending else None
  if pending and pending['name']!=unique:raise RuntimeError('unresolved submission')
  if jid is None:
   s['pending_submission']=dict(name=unique,job_name=jobname);w.save(s)
   cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes','--job-name='+jobname,
        '--output='+str(ROOT/'logs'/('%j_'+unique+'.log'))]
   if dep:cmd+=['--dependency='+('afterany:' if name.startswith('check') else 'afterok:')+str(dep)]
   cmd+=[str(HERE/script)]+list(map(str,args))
   jid=int(subprocess.check_output(cmd,text=True).strip().split(';')[0])
  s['jobs'].append(dict(name=unique,id=jid,nodes=nodes,max_wall_hours=wall));s['pending_submission']=None;w.save(s);dep=jid
 s['recovery_chain_submitted']=True;s['status']='recovery_preflight_queued';w.save(s)

def verify():
 s,c,b=guard();heartbeat();hosts=allocation()
 row=checkpoint(s['checkpoint']['path'])
 if row['sha256']!=s['checkpoint']['sha256']:raise RuntimeError('saved checkpoint changed')
 atomic(ROOT/'evidence/recovery_20261005/checkpoint_verified.json',dict(checkpoint=row,hosts=hosts,budget=b,utc=time.time()))

def preflight(rc):
 s,c,b=guard()
 if rc:w.terminal(s,'scheduler_failure','Reviewed continuation: MPI mesh preflight failed; no retry.')
 allocation();heartbeat()
 s['status']='ready_recovery_1';w.save(s)
 atomic(ROOT/'evidence/recovery_20261005/preflight_passed.json',dict(passed=True,utc=time.time(),script_hashes=c['script_hashes']))

def permit(number):
 s,c,b=guard();heartbeat();allocation()
 if s['status']!='ready_recovery_'+str(number) or not 1<=number<=2:raise RuntimeError('recovery state/number mismatch')
 if not (ROOT/'evidence/recovery_20261005/preflight_passed.json').exists():raise RuntimeError('preflight missing')
 row=checkpoint(s['checkpoint']['path'])
 if row['sha256']!=s['checkpoint']['sha256']:raise RuntimeError('restart checkpoint changed')
 s['status']='running_recovery_'+str(number);w.save(s)

def inspect(number):
 s=w.readstate()
 if s['status'] in w.TERMINAL:w.cancel_future(s);return
 name='recovery_20261005_continuation'+str(number)
 jid=next(j['id'] for j in s['jobs'] if j['name']==name)
 acct=w.scheduler_status(jid);s.setdefault('accounting',{})[str(jid)]=acct;w.save(s)
 run=ROOT/'runs'/('segment_'+str(number+2).zfill(2)+'_recovery_20261005')
 rows=[];row=None
 if (run/'EXIT_CODE').exists():
  rows=retain(ROOT,True,run);rows=[r for r in rows if str(run)+'/' in r['path']]
  if rows:
   row=max(rows,key=lambda r:r['cycle']);s['checkpoint']=row;s['time']=row['time'];s['cycle']=row['cycle'];w.save(s)
  seal(run)
 if s['stop_requested'] or (CONTROL/'USER_STOP').exists():w.terminal(s,'user_stop','Sticky user stop; verified checkpoint retained.')
 if acct[1].strip()!='COMPLETED' or acct[2]!='0:0':
  log=(run/'run.log').read_text(errors='replace') if (run/'run.log').exists() else ''
  reason='numerical_failure' if '### FATAL ERROR' in log else 'scheduler_failure'
  w.terminal(s,reason,'Continuation failed; review preserved logs. No automatic retry.')
 if (CONTROL/'RESOURCE_STOP.json').exists() or (CONTROL/'REQUEST_STOP').exists():w.terminal(s,'resource_limit','Resource/archive clean stop; verified checkpoint retained.')
 s,c,b=guard()
 if not rows: w.terminal(s,'no_progress','No verified continuation checkpoint')
 log=(run/'run.log').read_text(errors='replace')
 if '### FATAL ERROR' in log or '[conservation OK]' not in log:w.terminal(s,'numerical_failure','Fatal/particle conservation failure; no retry.')
 previous=c['old_checkpoint']['time'] if number==1 else s['recovery_previous_time']
 if row['time']<=previous:w.terminal(s,'no_progress','Checkpoint time did not increase')
 s['recovery_continuations_completed']=number;s['segments_completed']=number+2;s['recovery_previous_time']=row['time'];w.save(s)
 if (CONTROL/'RESOURCE_STOP.json').exists() or (CONTROL/'REQUEST_STOP').exists():w.terminal(s,'resource_limit','Resource/archive clean stop; inspect evidence.')
 if row['time']>=12-1e-10:w.terminal(s,'complete_t12','Approved assessment endpoint reached.')
 if number==2:w.terminal(s,'segment_cap','Two authorized continuations exhausted; no extension.')
 s['status']='ready_recovery_2';w.save(s)

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['setup','submit','verify','preflight','permit','inspect']);p.add_argument('--number',type=int);p.add_argument('--rc',type=int,default=0);a=p.parse_args()
 with w.locked():
  if a.action=='setup':setup()
  elif a.action=='submit':submit()
  elif a.action=='verify':verify()
  elif a.action=='preflight':preflight(a.rc)
  elif a.action=='permit':permit(a.number)
  else:inspect(a.number)

if __name__=='__main__':
 try:main()
 except Exception as e:
  atomic(ROOT/'evidence/recovery_20261005'/('ERROR_'+os.environ.get('SLURM_JOB_ID','interactive')+'.json'),dict(error=str(e),utc=time.time()))
  if os.environ.get('SLURM_JOB_ID'):
   with w.locked():
    s=w.readstate()
    if s['status'] not in w.TERMINAL:
     s['status']='configuration_failure';s['detail']=str(e);w.save(s);w.cancel_future(s)
  raise
