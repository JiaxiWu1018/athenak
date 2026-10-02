#!/usr/bin/env python3
"""Finite Anta cron metadata trigger; all transfers/hashes/rendering use Slurm."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from runtime import atomic
from anta_archive import AMD,DEST,remote
MARKER='JEANS8_20261002_METADATA'

def cron_lines():
 p=subprocess.run(['crontab','-l'],text=True,capture_output=True)
 if p.returncode==0:return p.stdout.splitlines()
 if 'no crontab' in p.stderr:return []
 raise RuntimeError(p.stderr)

def remove_entry():
 lines=[s for s in cron_lines() if MARKER not in s]
 subprocess.run(['crontab','-'],input='\n'.join(lines)+'\n',text=True,check=True)

def install():
 cfg=DEST/'evidence/trigger_config.json'
 if not cfg.exists():
  atomic(cfg,dict(deadline=time.time()+48*3600,max_jobs=4,max_gpu_hours=24))
 lines=cron_lines()
 (DEST/'evidence/crontab_before.txt').write_text('\n'.join(lines)+'\n')
 if not any(MARKER in s for s in lines):
  lines.append('*/5 * * * * /usr/bin/python3 '+str(DEST/'scripts/archive_trigger.py')+' tick >> '+str(DEST/'evidence/metadata_trigger.log')+' 2>&1 # '+MARKER)
  subprocess.run(['crontab','-'],input='\n'.join(lines)+'\n',text=True,check=True)

def tick():
 with (DEST/'evidence/trigger.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  cfg=json.loads((DEST/'evidence/trigger_config.json').read_text())
  if (DEST/'evidence/ARCHIVE_ANALYSIS_COMPLETE.json').exists():remove_entry();return
  if time.time()>cfg['deadline']:
   remote('touch '+AMD+'/control/ARCHIVE_ERROR; touch '+AMD+'/control/REQUEST_STOP')
   atomic(DEST/'evidence/TRIGGER_EXPIRED.json',dict(utc=time.time()));remove_entry();return
  free=os.statvfs(DEST);available=free.f_bavail*free.f_frsize
  if available<576*1024**3:
   remote('touch '+AMD+'/control/ARCHIVE_STORAGE_STOP; touch '+AMD+'/control/REQUEST_STOP');return
  # Successful metadata/capacity contact keeps production availability current.
  remote('date +%s > '+AMD+'/control/ARCHIVE_HEARTBEAT')
  state=json.loads(remote('cat '+AMD+'/control/state.json'))
  sealed=remote("find "+AMD+"/runs -mindepth 2 -maxdepth 2 -name SEALED -printf '%h\\n'").splitlines()
  ready=[r for r in sealed if not (DEST/'runs'/Path(r).name/'ARCHIVE_VERIFIED.json').exists()]
  terminal=state['status'] in ('complete_t12','segment_cap','user_stop','resource_limit','configuration_failure','numerical_failure','scheduler_failure','no_progress','emergency_cancel')
  if not ready and not terminal:return
  ledger=DEST/'evidence/active_archive_jobs.json'
  jobs=json.loads(ledger.read_text()) if ledger.exists() else dict(jobs=[],pending=None)
  if jobs['jobs']:
   jid=jobs['jobs'][-1]['id']
   pending=subprocess.check_output(['squeue','-h','-j',str(jid),'-o','%T'],text=True).strip()
   if pending:return
   acct=subprocess.check_output(['sacct','-X','-n','-P','-j',str(jid),'--format=JobIDRaw,State,ExitCode'],text=True)
   row=next((s.split('|') for s in acct.splitlines() if s.split('|')[0]==str(jid)),None)
   if not row or row[1].strip()!='COMPLETED' or row[2]!='0:0':
    remote('touch '+AMD+'/control/ARCHIVE_ERROR; touch '+AMD+'/control/REQUEST_STOP')
    atomic(DEST/'evidence/ARCHIVE_HALTED.json',dict(job=jid,accounting=row,utc=time.time()));remove_entry();return
  if len(jobs['jobs'])>=cfg['max_jobs']:
   remote('touch '+AMD+'/control/ARCHIVE_ERROR; touch '+AMD+'/control/REQUEST_STOP')
   atomic(DEST/'evidence/ARCHIVE_CAP.json',dict(utc=time.time()));remove_entry();return
  name='jn8_20261002_activearchive'+str(len(jobs['jobs']))
  found=set()
  if jobs['pending']==name:
   for line in subprocess.check_output(['sacct','-X','-n','-P','-S','now-2days','--format=JobIDRaw,JobName%80'],text=True).splitlines():
    row=line.split('|')
    if len(row)>1 and row[1].strip()==name:found.add(int(row[0]))
   for line in subprocess.check_output(['squeue','-h','-u','jiaxiwu','-o','%i|%j'],text=True).splitlines():
    row=line.split('|')
    if len(row)==2 and row[1]==name:found.add(int(row[0]))
  if len(found)>1:raise RuntimeError('ambiguous archive submission')
  if found:jid=found.pop()
  else:
   jobs['pending']=name;atomic(ledger,jobs)
   out=subprocess.check_output(['sbatch','--parsable','--job-name='+name,'--output='+str(DEST/'evidence/archive.%j.log'),str(DEST/'scripts/anta_archive.sbatch')],text=True)
   jid=int(out.strip().split(';')[0])
  jobs['jobs'].append(dict(id=jid,name=name,max_gpu_hours=6));jobs['pending']=None;atomic(ledger,jobs)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['install','tick','remove']);a=p.parse_args()
 if a.action=='install':install()
 elif a.action=='remove':remove_entry()
 else:tick()
