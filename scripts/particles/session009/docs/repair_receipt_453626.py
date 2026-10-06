#!/usr/bin/env python3
"""Reviewed operations repair after actual input accepted, receipt import failed."""
import hashlib,json,pathlib,shutil,subprocess,time
r=pathlib.Path('/work1/eliasmost/jiaxiwu/gi_s009_amd_20261005')
s=json.loads((r/'control/state.json').read_text());c=json.loads((r/'control/config.json').read_text())
assert s['status']=='configuration_failure' and s['time']==0
assert not any((r/'control'/n).exists() for n in ('USER_STOP','REQUEST_STOP','RESOURCE_STOP.json'))
ids={str(j['id']) for j in s['jobs']}
assert not ids.intersection(subprocess.check_output(['squeue','-h','-u','jiaxiwu','-o','%i'],text=True).split())
h=r/'history/r40_receipt_import_failure_453626';h.mkdir()
for name in ('config.json','state.json'):shutil.copy2(r/'control'/name,h/name)
shutil.copytree(r/'evidence/r40_preflight_453626',h/'r40_preflight_453626')
for jid,name in [(453626,'input_preflight'),(453628,'inspect0')]:
 p=r/'logs'/f'{jid}_{name}.log'
 if p.exists():shutil.copy2(p,h/p.name)
(h/'REVIEW.md').write_text('All actual compiled -n commands accepted full/slim/segment input. Failure only in inline Python receipt import from batch working directory. Explicit scripts path added, pushed ddf31ebd; no particles/evolution or numerical retry. Failed/cancelled jobs retain cumulative accounting.\n')
s.setdefault('attempt_history',[]).append(dict(status=s['status'],detail=s['detail'],radius=40,time=0,history=str(h),utc=time.time()))
for j in s['jobs']:
 if j['id'] in (453626,453627,453628):j['name']='receipt_failure_'+j['name']
s.update(status='r40_preparing',gates_submitted=False,gates_passed=False,pending_submission=None,detail='Reviewed receipt import path repaired; input commands already passed. No evolution yet.',updated_utc=time.time())
c.update(job_prefix='jn9_20261006_r40b_',r40_operations_revision='ddf31ebd47f515756637b643656d47cb465c3fa4',r40_receipt_repair_utc=time.time(),script_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'scripts').iterdir() if p.is_file()})
for p,v in [(r/'control/config.json',c),(r/'control/state.json',s)]:
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
print(json.dumps(dict(reviewed=True,historical_jobs=len(s['jobs']),deadline=c['deadline_utc'],input_sha256=c['input_sha256'])))
