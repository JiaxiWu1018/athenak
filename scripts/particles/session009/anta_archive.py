#!/usr/bin/env python3
"""One finite allocated Anta pull/verify/cleanup/update pass. Never busy-waits for data."""
import argparse,json,os,subprocess,time
from pathlib import Path
from runtime import atomic,digest
from workflow import TERMINAL
AMD='/work1/eliasmost/jiaxiwu/gi_s009_amd_20261005'
DEST=Path('/data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005')
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','hpcfund.amd.com']
def remote(cmd):return subprocess.check_output(SSH+[cmd],text=True,timeout=120)

def archive():
 sealed=remote("find "+AMD+"/runs -mindepth 2 -maxdepth 2 -name SEALED -printf '%h\\n'").splitlines()
 for run in sealed:
  name=Path(run).name
  if not name.startswith(('gate_','wave_','segment_')):raise RuntimeError('unexpected run')
  target=DEST/'runs'/name
  if (target/'ARCHIVE_VERIFIED.json').exists():continue
  manifest=json.loads(remote('cat '+run+'/SCIENCE_MANIFEST.json'));expected=sum(r['bytes'] for r in manifest)
  used=int(subprocess.check_output(['du','-sx','--block-size=1',str(DEST)],text=True).split()[0])
  stat=os.statvfs(DEST);free=stat.f_bavail*stat.f_frsize
  if used+expected>16*1024**4 or free-expected<512*1024**3:
   remote('touch '+AMD+'/control/REQUEST_STOP '+AMD+'/control/ARCHIVE_STORAGE_STOP')
   raise RuntimeError('Anta capacity limit')
  target.mkdir(parents=True,exist_ok=True)
  subprocess.run(['rsync','-a','--partial','hpcfund.amd.com:'+run+'/',str(target)+'/'],check=True,timeout=6000)
  for r in manifest:
   p=target/r['path']
   if not p.resolve().is_relative_to(target.resolve()):raise RuntimeError('unsafe path')
   if p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:raise RuntimeError('archive checksum mismatch '+str(p))
  receipt=dict(source=run,destination=str(target),files=manifest,verified_utc=time.time(),job=os.environ.get('SLURM_JOB_ID'))
  atomic(target/'ARCHIVE_VERIFIED.json',receipt)
  payload=dict(run=run,files=[r for r in manifest if Path(r['path']).suffix in ('.bin','.vtk','.cbin')],checkpoints=[r for r in manifest if Path(r['path']).suffix=='.rst'],destination=str(target),verified_utc=receipt['verified_utc'],job=receipt['job'])
  subprocess.run(SSH+['python3 '+AMD+'/scripts/delete_archived.py'],input=json.dumps(payload),text=True,check=True,timeout=120)

def snapshot():
 state=json.loads(remote('cat '+AMD+'/control/state.json'));atomic(DEST/'evidence/amd_state.json',state)
 ids=','.join(str(int(j['id'])) for j in state['jobs'])
 (DEST/'evidence/amd_accounting.psv').write_text(remote('sacct -X -n -P -j '+ids+' --format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocNodes'))
 for name in ('config.json','latest_checkpoint.json'):
  atomic(DEST/'evidence'/name,json.loads(remote('if test -f '+AMD+'/control/'+name+'; then cat '+AMD+'/control/'+name+'; else echo "{}"; fi')))
 for name in ('initial_validation.json','restart_validation.json','gate_receipt.json','wave_gate.json','weyl_convention.json','mesh_audit.json'):
  atomic(DEST/'evidence'/name,json.loads(remote('if test -f '+AMD+'/evidence/'+name+'; then cat '+AMD+'/evidence/'+name+'; else echo "{}"; fi')))
 for name in ('executable.sha256','wave_executable.sha256','build_provenance.txt'):
  (DEST/'evidence'/name).write_text(remote('if test -f '+AMD+'/evidence/'+name+'; then cat '+AMD+'/evidence/'+name+'; fi'))
 return state

def main():
 p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');p.parse_args()
 try:
  archive();state=snapshot()
  # Analyze only verified physics history; repeated validation branches are excluded.
  evolution=[r for r in (DEST/'runs').glob('*') if r.name=='gate_output' or r.name.startswith('segment_')]
  times=[]
  for r in evolution:
   receipt=r/'ARCHIVE_VERIFIED.json'
   if receipt.exists():
    records=[q for q in json.loads(receipt.read_text())['files'] if q['path'].endswith('.rst')]
    if records:
     from runtime import checkpoint
     times.append(max(checkpoint(r/q['path'],False)['time'] for q in records))
  covered=max(times,default=0.)
  completed=json.loads((DEST/'evidence/updates.json').read_text()) if (DEST/'evidence/updates.json').exists() else []
  milestones=[0,12]+list(range(50,401,50))
  due=[m for m in milestones if m<=covered+1e-8 and m not in completed]
  terminal=state['status'] in TERMINAL
  if due or (terminal and 'final' not in completed):
   for milestone in list(map(str,due))+(['final'] if terminal else []):
    subprocess.run(['/home/jiaxiwu/miniconda3/bin/python',str(DEST/'scripts/analyze_s9.py'),'--root',str(DEST),'--milestone',milestone],check=True,timeout=5400)
   completed+=due
   if terminal:completed.append('final')
   atomic(DEST/'evidence/updates.json',completed)
  if terminal:atomic(DEST/'evidence/ARCHIVE_ANALYSIS_COMPLETE.json',dict(state=state,covered_time=covered,utc=time.time()))
 except Exception as e:
  with (DEST/'evidence/archive_errors.jsonl').open('a') as f:f.write(json.dumps(dict(error=str(e),utc=time.time()))+'\n')
  try:remote('touch '+AMD+'/control/ARCHIVE_ERROR '+AMD+'/control/REQUEST_STOP')
  except Exception:pass
  raise
if __name__=='__main__':main()
