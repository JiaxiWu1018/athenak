#!/usr/bin/env python3
"""Approved, one-off metadata reset of failed R50 attempt; never deletes data."""
import argparse, hashlib, json, pathlib, shutil, subprocess, time
AMD=pathlib.Path('/work1/eliasmost/jiaxiwu/gi_s009_amd_20261005')
ANTA=pathlib.Path('/data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005')
HISTORY='r50_before_reset_20261006'
OPS='6b9806ebc6150153b4976af6ebd63ceba1f6fdbd'
SCIENCE='ae89f066'
def atomic(p,value):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)
def prepare(site):
 root=AMD if site=='amd' else ANTA
 history=root/'history'/HISTORY
 if history.exists():raise RuntimeError('History exists; inspect before retrying')
 if site=='amd':
  state=json.loads((root/'control/state.json').read_text())
  if state['status']!='configuration_failure' or state['time']!=0:raise RuntimeError('Unexpected scientific state')
  if any((root/'control'/n).exists() for n in ('USER_STOP','REQUEST_STOP','RESOURCE_STOP.json')):raise RuntimeError('Stop marker must be reviewed, never erased')
  registered={str(j['id']) for j in state['jobs']}
 else:
  ledger=json.loads((root/'evidence/active_archive_jobs.json').read_text())
  if ledger['pending'] is not None:raise RuntimeError('Pending archival submission')
  registered={str(j['id']) for j in ledger['jobs']}
 queued=set(subprocess.check_output(['squeue','-h','-u','jiaxiwu','-o','%i'],text=True).split())
 if queued & registered:raise RuntimeError('Registered jobs still active')
 inventory=[]
 for folder in ('control','inputs','scripts','evidence'):
  for p in (root/folder).rglob('*'):
   if p.is_file():inventory.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size))
 for name in ('gate_mesh','gate_reference'):
  run=root/'runs'/name
  if site=='anta' and not (run/'ARCHIVE_VERIFIED.json').exists():raise RuntimeError('Old failed run lacks verified archive')
  for p in run.rglob('*'):
   if p.is_file():inventory.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size))
 history.mkdir(parents=True)
 for folder in ('control','inputs','scripts','evidence'):
  if (root/folder).exists():shutil.copytree(root/folder,history/folder)
 for name in ('README.md','REPORT_AGENT.md','REPORT_Jeans9.md','manifest.json'):
  if (root/name).exists():shutil.copy2(root/name,history/name)
 (history/'runs').mkdir()
 for name in ('gate_mesh','gate_reference'):
  (root/'runs'/name).rename(history/'runs'/name)
 if site=='anta':
  for name in ('ARCHIVE_ANALYSIS_COMPLETE.json','updates.json','INITIAL_SCIENCE_REPORT.json','latest_checkpoint.json','initial_validation.json','restart_validation.json','gate_receipt.json','mesh_audit.json'):
   p=root/'evidence'/name
   if p.exists():p.rename(history/'evidence'/('active_'+name))
  link=root/'analysis/latest'
  if link.is_symlink():link.rename(history/'analysis_latest')
 else:
  p=root/'control/ERROR_interactive.json'
  if p.exists():p.rename(history/'control/active_ERROR_interactive.json')
 atomic(history/'RESET_INVENTORY.json',dict(site=site,utc=time.time(),authorization='2026-10-06 user: change extraction to40, reset and run; automatic two-hour wake',files=inventory,notes='Metadata copies and failed-run directory renames only; no checkpoint/source/binary changes or deletions. Wave receipts and cumulative budgets retained.'))
 print(json.dumps(dict(prepared=site,history=str(history),files=len(inventory))))
def freeze():
 root=AMD
 if not (root/'history'/HISTORY/'control/config.json').exists():raise RuntimeError('History absent')
 c=json.loads((root/'control/config.json').read_text());s=json.loads((root/'control/state.json').read_text())
 if s['status']!='configuration_failure':raise RuntimeError('Unexpected state')
 if subprocess.check_output(['git','-C',str(root/'athenak'),'rev-parse','HEAD'],text=True).strip()!=c['compiled_source_commit']:raise RuntimeError('Compiled source changed')
 h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 c.update(input_sha256=h(root/'inputs/gi_cluster_s9.athinput'),script_hashes={p.name:h(p) for p in (root/'scripts').iterdir() if p.is_file()},extraction_radius=40,wave_floor_radius=46,initial_blocks_expected=5384,job_prefix='jn9_20261006_r40_',reuse_wave_gate=True,r40_operations_revision=OPS,r40_science_revision=SCIENCE,r40_reset_utc=time.time(),r40_history=str(root/'history'/HISTORY))
 s.setdefault('attempt_history',[]).append(dict(status=s['status'],detail=s.get('detail'),radius=50,time=0,history=str(root/'history'/HISTORY),utc=time.time()))
 for j in s['jobs']:
  if j['name']=='gate':j['name']='r50_gate_failed'
  elif j['name']=='inspect0':j['name']='r50_inspect0'
 s.update(status='r40_preparing',gates_submitted=False,gates_passed=False,pending_submission=None,detail='Approved R40/R46 fresh start; failed R50 preserved; historical costs and deadlines retained.',updated_utc=time.time())
 atomic(root/'control/config.json',c);atomic(root/'control/state.json',s)
 atomic(root/'evidence/r40_reset_binding.json',dict(input_sha256=c['input_sha256'],compiled_source_commit=c['compiled_source_commit'],operations_revision=OPS,scientific_revision=SCIENCE,created_utc=c['created_utc'],deadline_utc=c['deadline_utc'],historical_jobs=s['jobs'],history=c['r40_history'],utc=time.time()))
 print(json.dumps(dict(frozen=True,input_sha256=c['input_sha256'],scripts=len(c['script_hashes']),historical_jobs=[j['id'] for j in s['jobs']],deadline=c['deadline_utc'])))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare-amd','prepare-anta','freeze']);a=p.parse_args()
 if a.action=='freeze':freeze()
 else:prepare(a.action.split('-')[1])
