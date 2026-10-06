#!/usr/bin/env python3
"""Two-hour metadata check and wake of the existing Codex thread, bounded by campaign."""
import argparse,fcntl,json,os,shlex,sqlite3,subprocess,time
from pathlib import Path
LOCAL=Path('/data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005')
AMD='/work1/eliasmost/jiaxiwu/gi_s009_amd_20261005'
CODEX='/home/jiaxiwu/.local/bin/codex'
THREAD='01a0f9d8-f1e1-7250-b3b3-0bb0c6dfa014'
MARKER='JEANS9_R40_TWO_HOUR_WAKE'
PROMPT='Automatic two-hour Session009 check requested by Jiaxi. Read AGENTS/logistics and evidence/CURRENT_STATUS.md. Check live registered AMD jobs, latest verified checkpoint/time, numerical health, node-hours and Anta archival/report/plot progress. R40/R46, boost0.133215 and all approved limits apply. Continue approved routine work if needed; heavy work only in Slurm. Respect sticky user-stop and hard limits. Do not change physics or automatically retry numerical failures. Preserve evidence and give a brief truthful status update. Record the check; do not spawn agents.'
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)
def cron():
 p=subprocess.run(['crontab','-l'],text=True,capture_output=True)
 if p.returncode and 'no crontab' not in p.stderr:raise RuntimeError(p.stderr)
 return p.stdout.splitlines() if p.returncode==0 else []
def remove():
 subprocess.run(['crontab','-'],input='\n'.join(s for s in cron() if MARKER not in s)+'\n',text=True,check=True)
def activity():
 # Bounded read-only metadata from this thread's persisted rollout. Never edit
 # Codex databases. Unknown/interrupted histories conservatively skip a wake.
 db=sqlite3.connect('file:/home/jiaxiwu/.codex/state_5.sqlite?mode=ro',uri=True)
 row=db.execute('select rollout_path from threads where id=?',(THREAD,)).fetchone();db.close()
 if not row:raise RuntimeError('existing Codex thread not found')
 path=Path(row[0]);state='unknown'
 with path.open('rb') as f:f.seek(max(0,path.stat().st_size-512*1024));lines=f.read().splitlines()
 for line in lines:
  try:item=json.loads(line)
  except (ValueError,UnicodeError):continue
  payload=item.get('payload',{})
  if item.get('type')=='response_item':
   state='idle' if payload.get('type')=='message' and payload.get('role')=='assistant' and payload.get('channel')=='final' else 'active'
  elif item.get('type')=='event_msg' and payload.get('type') in ('task_complete','turn_completed','turn_aborted'):state='idle'
 return state
def deliver():
 stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());log=LOCAL/'evidence'/('wake_'+stamp+'.jsonl');answer=LOCAL/'evidence'/('wake_'+stamp+'.md')
 cmd=[CODEX,'exec','--approve-for-me','--skip-git-repo-check','--cd','/data/jiaxiwu/NRPIC','--json','--output-last-message',str(answer),'resume',THREAD,PROMPT]
 with log.open('w') as f:
  result=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
 if result.returncode:raise RuntimeError('Codex resume wake failed; see '+str(log))
 return dict(status='wake_completed',wake_log=str(log),answer=str(answer))
def metadata():
 code="import json,subprocess;from pathlib import Path;r=Path("+repr(AMD)+");s=json.loads((r/'control/state.json').read_text());ids=','.join(str(j['id']) for j in s['jobs']);print(json.dumps(dict(state=s,queue=subprocess.check_output(['squeue','-h','-u','jiaxiwu','-o','%i|%j|%T|%R'],text=True),accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',ids,'--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocNodes'],text=True),user_stop=(r/'control/USER_STOP').exists(),heartbeat=(r/'control/ARCHIVE_HEARTBEAT').read_text() if (r/'control/ARCHIVE_HEARTBEAT').exists() else None)))"
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','hpcfund.amd.com','python3 -c '+shlex.quote(code)],text=True,capture_output=True,timeout=120)
 if p.returncode:raise RuntimeError('AMD metadata contact failed: '+p.stderr[-500:])
 return json.loads(p.stdout)
def check(probe=False):
 config=json.loads((LOCAL/'control/wake_config.json').read_text());record=dict(utc=time.time(),probe=probe)
 if time.time()>config['deadline_utc'] or (LOCAL/'control/WAKE_STOP').exists():remove();record['status']='expired_or_user_disabled';return record
 snapshot=metadata();record['AMD']=snapshot;write(LOCAL/'evidence/two_hour_live_status.json',record)
 if snapshot['user_stop']:remove();record['status']='sticky_user_stop';return record
 status=activity();record['thread_status']=status
 if probe:record['status']='probe_passed';return record
 if status!='idle':record['status']='checked_agent_already_active' if status=='active' else 'checked_agent_status_unknown';return record
 record.update(deliver());return record
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['install','tick','probe','remove']);a=p.parse_args()
 if a.action=='remove':remove();return
 if a.action=='install':
  lines=cron();(LOCAL/'evidence/crontab_before_two_hour_wake.txt').write_text('\n'.join(lines)+'\n')
  if not any(MARKER in s for s in lines):lines.append('0 */2 * * * /usr/bin/python3 '+str(LOCAL/'code/scripts/particles/session009/wake_check.py')+' tick >> '+str(LOCAL/'evidence/two_hour_wake.log')+' 2>&1 # '+MARKER)
  subprocess.run(['crontab','-'],input='\n'.join(lines)+'\n',text=True,check=True);return
 with (LOCAL/'control/wake.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  try:r=check(a.action=='probe')
  except Exception as e:r=dict(utc=time.time(),status='check_failed',error=str(e))
  write(LOCAL/'evidence/two_hour_check_latest.json',r)
  with (LOCAL/'evidence/two_hour_checks.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
  print(json.dumps({k:v for k,v in r.items() if k!='AMD'}))
  if r['status']=='check_failed':raise SystemExit(1)
if __name__=='__main__':main()
