#!/usr/bin/env python3
"""Reviewed filename repair, reusing successful saved evolution; not a numerical retry."""
import argparse,fcntl,hashlib,json,shutil,subprocess,time
from pathlib import Path

ROOT=Path('/work1/eliasmost/jiaxiwu/gi_s009_amd_20261005')
HISTORY=ROOT/'history/validation_failure_454732_20261008'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)

def preconditions():
 s=json.loads((ROOT/'control/state.json').read_text())
 assert s['status']=='configuration_failure' and s['time']==0 and not s['stop_requested']
 assert not s.get('pending_submission')
 for n in ('USER_STOP','REQUEST_STOP','RESOURCE_STOP.json','ARCHIVE_ERROR','ARCHIVE_STORAGE_STOP'):
  assert not (ROOT/'control'/n).exists(),n
 live={x.split('|')[0] for x in subprocess.check_output(['squeue','-h','-u','jiaxiwu','-o','%i|%T'],text=True).splitlines()}
 assert not any(str(j['id']) in live for j in s['jobs'])
 acct=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(str(j['id']) for j in s['jobs']),
  '--format=JobIDRaw,State%30,ExitCode,ElapsedRaw,AllocNodes'],text=True)
 rows={x.split('|')[0]:x.split('|') for x in acct.splitlines()}
 assert rows['454732'][1:3]==['FAILED','1:0'] and rows['454733'][1:3]==['COMPLETED','0:0']
 assert all(rows[str(j['id'])][1].split()[0].rstrip('+') in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'} for j in s['jobs'])
 text=(ROOT/'logs/454732_gate.log').read_text()
 assert 'FileNotFoundError: gi_profile_M076_two_clump_s7.txt' in text
 run=ROOT/'runs/gate_reference'
 assert (run/'SEALED').exists() and (run/'EXIT_CODE').read_text().strip()=='0'
 assert '[conservation OK]' in (run/'run.log').read_text() and '### FATAL ERROR' not in (run/'run.log').read_text()
 row=json.loads((ROOT/'control/latest_checkpoint.json').read_text())
 assert row['time']==.0125 and row['cycle']==2 and row['particles']==5000000 and row['blocks']==5384
 assert row['sha256']=='899c920a37f6bcd1a20ebe081256ce0ac36baf5e6bbb87d4e445348c32bf67b7'
 assert Path(row['path']).exists() and Path(row['path']).stat().st_size==row['bytes']
 archive=json.loads((ROOT/'control/archived_checkpoints.json').read_text())
 assert archive[row['path']]['sha256']==row['sha256']
 return s,acct,row

def preserve(s,acct,row):
 assert not HISTORY.exists(),'Already preserved; inspect instead of repeating'
 HISTORY.mkdir(parents=True)
 for n in ('control','scripts','inputs'):shutil.copytree(ROOT/n,HISTORY/n)
 for n in ('evidence','logs'):(HISTORY/n).mkdir()
 for n in ('REPORT_Jeans9.md','REPORT_AGENT.md','README.md','manifest.json'):
  if (ROOT/n).exists():shutil.copy2(ROOT/n,HISTORY/n)
 for n in ('gate_provenance.txt','mpi_48rank_gate.txt','CURRENT_STATUS.md','frozen_config.json','gate_receipt.json','initial_validation.json','mesh_audit.json'):
  if (ROOT/'evidence'/n).exists():shutil.copy2(ROOT/'evidence'/n,HISTORY/'evidence'/n)
 for p in (ROOT/'logs').glob('45473[123]*'):shutil.copy2(p,HISTORY/'logs'/p.name)
 (HISTORY/'accounting.psv').write_text(acct)
 write(HISTORY/'PRESERVATION_INVENTORY.json',dict(utc=time.time(),reason='Successful finite two-cycle evolution; inherited validator requested Session007 profile filename. Raw sealed run remains in place and is archived by checksum on Anta.',checkpoint=row,state=s,
  files=[dict(path=str(p.relative_to(HISTORY)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(HISTORY.rglob('*')) if p.is_file()]))

def reset(s,row,revision):
 assert len(revision)==40 and all(x in '0123456789abcdef' for x in revision)
 assert json.loads((HISTORY/'control/state.json').read_text())==s
 c=json.loads((ROOT/'control/config.json').read_text())
 assert c['compiled_source_commit']=='6892be3e3f04ec573f91cb2034bc9d3009a3bdff'
 assert c['input_sha256']==sha(ROOT/'inputs/gi_cluster_s9.athinput')
 assert c['created_utc']==1791237703.7323442 and c['deadline_utc']==1795125703.7323442
 assert c['extraction_radius']==40 and c['wave_floor_radius']==46
 c.update(resume_reference=True,resume_reference_checkpoint=row,reference_repair_revision=revision,
  reference_repair_utc=time.time(),job_prefix='jn9_20261008_resume_',reuse_wave_gate=True)
 c['script_hashes']={p.name:sha(p) for p in (ROOT/'scripts').iterdir() if p.is_file()}
 s.setdefault('attempt_history',[]).append(dict(status=s['status'],detail=s['detail'],time=s['time'],history=str(HISTORY),utc=time.time(),
  reviewed_action='Repair profile filename and revalidate saved reference; skip duplicate reference evolution. Continue remaining gates only on successful allocated validation.'))
 for j in s['jobs']:
  if j['id'] in (454731,454732,454733):j['name']='profile454732_failure_'+j['name']
 s.update(status='reference_reviewed',gates_submitted=False,gates_passed=False,detail='Saved reference t0.0125 checkpoint will be revalidated, then remaining restart/clean-stop/output gates run once.',pending_submission=None,updated_utc=time.time())
 write(ROOT/'control/config.json',c);write(ROOT/'evidence/frozen_config.json',c);write(ROOT/'control/state.json',s)
 write(ROOT/'evidence/REFERENCE_REPAIR_REVIEW_20261008.json',dict(utc=time.time(),revision=revision,history=str(HISTORY),checkpoint=row,config=c,remaining_startup_checks_pending=True))
 print(json.dumps(dict(status=s['status'],script_count=len(c['script_hashes']),checkpoint=row)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['preserve','reset']);p.add_argument('--revision');a=p.parse_args()
 with (ROOT/'control/workflow.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);s,acct,row=preconditions()
  if a.action=='preserve':preserve(s,acct,row)
  else:reset(s,row,a.revision)
