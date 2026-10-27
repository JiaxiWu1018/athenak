"""Completion/failure-only notification. No hourly wakeups or auto campaign restarts."""
from pathlib import Path
import subprocess,json,shlex,time,datetime
LOCAL=Path('/data/jiaxiwu/NRPIC/ST_migration/session_10_yc0500_outward_20261024/extensions/u0_1500_to2500_20261026')
REMOTE='/work1/eliasmost/jiaxiwu/st_migration_s10_u0_2500_20261026'
THREAD='01a1700d-9f0a-7801-98d4-0d700ff3c413'
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)
program=r'''
from pathlib import Path
import json,subprocess
r=Path('/work1/eliasmost/jiaxiwu/st_migration_s10_u0_2500_20261026');rows=json.loads((r/'evidence/job_registry.json').read_text());result=[]
for x in rows:
 a=r/'evidence/allocations'/x['id'];d=dict(x)
 for name in ['STARTED','WORKER_OK','FAILURE','ALLOCATION_END']:
  p=a/(name+'.json');d[name]=json.loads(p.read_text()) if p.exists() else None
 p=a/'scheduler_terminal.json'
 if p.exists():d['scheduler']=json.loads(p.read_text())['scheduler']
 else:
  q=subprocess.run(['scontrol','show','job',x['id'],'-o'],capture_output=True,text=True);d['scheduler']=(q.stdout+q.stderr).strip()
  if any('JobState='+s in d['scheduler'] for s in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL']):p.write_text(json.dumps(dict(utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),scheduler=d['scheduler']),indent=2)+'\n')
 result.append(d)
out=dict(jobs=result)
for name in ['FINAL_PRODUCTS_READY','PIPELINE_DONE','DISPATCH_FAILURE','CONTINUATION_PREPARATION_OK','PERFORMANCE_REPAIR_PLAN']:
 p=r/'evidence'/(name+'.json');out[name]=json.loads(p.read_text()) if p.exists() else None
p=r/'yc0500_control/evidence/RESOURCE_STOP.json';out['RESOURCE_STOP']=json.loads(p.read_text()) if p.exists() else None
p=r/'yc0500_control/runs/yc0500/state.json';out['latest']=json.loads(p.read_text())['latest'] if p.exists() else None
print(json.dumps(out))
'''
def check():
 q=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','hpcfund.amd.com','python3 -c '+shlex.quote(program)],capture_output=True,text=True,timeout=55);assert q.returncode==0,q.stderr;d=json.loads(q.stdout);d['observed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(LOCAL/'evidence/remote_status.json',d);save(LOCAL/'evidence/job_registry.json',[{k:v for k,v in x.items() if k not in ['scheduler','STARTED','WORKER_OK','FAILURE','ALLOCATION_END']} for x in d['jobs']]);return d

def reviewed_failure(x,d):
 p=d.get('PERFORMANCE_REPAIR_PLAN')
 return bool(p and x['id']==p['stopped_job'] and x.get('FAILURE')==p['acknowledged_failure'] and x.get('ALLOCATION_END')==p['acknowledged_allocation_end'] and 'JobState=FAILED' in x['scheduler'] and 'ExitCode=1:0' in x['scheduler'])

def needs_repair(d):
 failed=any(not reviewed_failure(x,d) and (x.get('FAILURE') or (x.get('ALLOCATION_END') and x['ALLOCATION_END']['exit_status']!=0) or any('JobState='+s in x['scheduler'] for s in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'])) for x in d['jobs'])
 resource=d['RESOURCE_STOP'];p=d.get('PERFORMANCE_REPAIR_PLAN')
 reviewed_resource=bool(resource and p and resource.get('job')==p['stopped_job'] and resource.get('reason')==p['stop_reason'])
 return bool(failed or d['DISPATCH_FAILURE'] or (resource and not reviewed_resource))

def main():
 while not (LOCAL/'evidence/EXTENSION_CLOSED.json').exists():
  try:
   d=check();error=needs_repair(d)
   latest=d['jobs'][-1];terminal='JobState=COMPLETED' in latest['scheduler'] and 'ExitCode=0:0' in latest['scheduler'] and latest['WORKER_OK'] and latest['ALLOCATION_END'] and latest['ALLOCATION_END']['exit_status']==0
   ready=d['PIPELINE_DONE'] and d['FINAL_PRODUCTS_READY'] and terminal
   if ready or error:
    ident=latest['id'];event=('plots_ready_' if ready else 'repair_needed_')+ident;p=LOCAL/'evidence'/('notification_'+event+'.json')
    if not p.exists() or json.loads(p.read_text()).get('returncode')!=0:
     message=('AMD unperturbed1500->2500M extension has finished its numerical analysis and generated the requested shell and central-lapse plots. ' if ready else 'AMD unperturbed1500->2500M extension requires safe review of an actual recorded failure/resource stop. ')+f'Read {LOCAL}/HANDOFF.md, APPROVED_PLAN.md and evidence/remote_status.json. '+('Verify complete endpoint/actual rendered plots, checksum thin-copy, costs/storage and deliver them; do not submit duplicate production. ' if ready else 'Inspect actual registered state and repair/resume only this authorized extension within shared50nodeh/350GiB; preserve1500M and other campaigns. ')+ 'Original hourly wakes stay disabled; this is a completion/failure-only event.'
     q=subprocess.run(['/home/jiaxiwu/.local/bin/codex','queue','--thread',THREAD,'--message',message],capture_output=True,text=True,timeout=50);save(p,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event,returncode=q.returncode,stdout=q.stdout,stderr=q.stderr))
     if q.returncode!=0:raise RuntimeError(q.stderr)
    return
  except Exception as error:save(LOCAL/'evidence/completion_monitor_error.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(error)))
  time.sleep(30)
if __name__=='__main__':main()
