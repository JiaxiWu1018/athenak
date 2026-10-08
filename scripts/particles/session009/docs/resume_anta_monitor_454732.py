from pathlib import Path
import fcntl,json,time,shutil,hashlib,subprocess,sys
root=Path('/data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005')
sys.path.insert(0,str(root/'scripts'))
from runtime import atomic
from anta_archive import remote,AMD
from archive_trigger import install,tick
history=root/'history/validation_failure_454732_20261008/anta_metadata'
with (root/'evidence/trigger.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX)
 state=json.loads(remote('cat '+AMD+'/control/state.json'))
 assert state['status']=='reference_reviewed' and not state['stop_requested']
 remote('python3 '+AMD+'/scripts/workflow.py bindings')
 cfg=json.loads((root/'evidence/trigger_config.json').read_text())
 assert cfg['deadline']==1795125727.0788982 and time.time()<cfg['deadline']-4*3600
 jobs=json.loads((root/'evidence/active_archive_jobs.json').read_text())
 assert len(jobs['jobs'])==6 and not jobs['pending'] and cfg['max_jobs']==96 and cfg['max_gpu_hours']==384
 live={x.split('|')[0] for x in subprocess.check_output(['squeue','-h','-u','jiaxiwu','-o','%i|%T'],text=True).splitlines()}
 assert not any(str(j['id']) in live for j in jobs['jobs'])
 row=json.loads((root/'evidence/ARCHIVE_ANALYSIS_COMPLETE.json').read_text())
 assert row['state']['status']=='configuration_failure' and row['state']['jobs'][-2]['id']==454732
 assert not (root/'evidence/ARCHIVE_HALTED.json').exists() and not (root/'evidence/ARCHIVE_CAP.json').exists()
 assert not history.exists()
 history.mkdir()
 names=['evidence/ARCHIVE_ANALYSIS_COMPLETE.json','evidence/updates.json','analysis/latest.json','REPORT_Jeans9.md','REPORT_AGENT.md','manifest.json']
 for n in names:
  p=root/n
  if p.exists():
   dest=history/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 atomic(history/'INVENTORY.json',dict(utc=time.time(),reason='Reviewed saved-reference validator filename repair; retain terminal closeout and products before monitoring resumes.',files=[dict(path=str(p.relative_to(history)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(history.rglob('*')) if p.is_file()]))
 for n in ['evidence/ARCHIVE_ANALYSIS_COMPLETE.json','analysis/latest.json']:
  p=root/n
  if p.exists():p.rename(history/(p.name+'.inactive'))
 updates=json.loads((root/'evidence/updates.json').read_text())
 atomic(root/'evidence/updates.json',[m for m in updates if m!='final'])
 name='jn9_20261008_savedref_validation'
 jobs['pending']=name;atomic(root/'evidence/active_archive_jobs.json',jobs)
 out=subprocess.check_output(['sbatch','--parsable','--job-name='+name,'--output='+str(root/'evidence/reference_validation.%j.log'),str(root/'scripts/anta_validate_reference.sbatch')],text=True,stderr=subprocess.STDOUT)
 jid=int(out.strip().split(';')[0])
 jobs['jobs'].append(dict(id=jid,name=name,max_gpu_hours=4,role='Reviewed saved startup validation, minimal reference-file restoration and guarded AMD submission'))
 jobs['pending']=None;atomic(root/'evidence/active_archive_jobs.json',jobs)
 atomic(root/'evidence/REFERENCE_VALIDATION_SUBMISSION.json',dict(utc=time.time(),job=jid,revision='a60f312e7a5145325f543bc48b108ca8f5f47cb6',max_hours=4,history=str(history)))
install();tick()
print(json.dumps(dict(job=jid,ledger_jobs=len(jobs['jobs']),cron=subprocess.check_output(['crontab','-l'],text=True),queue=subprocess.check_output(['squeue','-h','-j',str(jid),'-o','%i|%T|%M|%R'],text=True)),indent=2))
