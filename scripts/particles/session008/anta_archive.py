#!/usr/bin/env python3
"""Bounded Anta compute-node pull, checksum verification, cleanup and analysis."""
import argparse,json,os,subprocess,time,threading
from pathlib import Path
from runtime import atomic,digest

AMD='/work1/eliasmost/jiaxiwu/gi_s008_amd_20261002'
DEST=Path('/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002')
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','hpcfund.amd.com']
def remote(cmd):return subprocess.check_output(SSH+[cmd],text=True,timeout=120)

def archive():
 listing=remote("find "+AMD+"/runs -mindepth 2 -maxdepth 2 -name SEALED -printf '%h\\n'")
 changed=False
 for run in listing.splitlines():
  name=Path(run).name
  if not (name.startswith('gate_') or name.startswith('segment_')):raise RuntimeError('unexpected run')
  target=DEST/'runs'/name
  if (target/'ARCHIVE_VERIFIED.json').exists():continue
  manifest=json.loads(remote('cat '+run+'/SCIENCE_MANIFEST.json'))
  expected=sum(x['bytes'] for x in manifest)
  used=int(subprocess.check_output(['du','-sx','--block-size=1',str(DEST)],text=True).split()[0])
  stat=os.statvfs(DEST);free=stat.f_bavail*stat.f_frsize
  if used+expected>1024**4 or free-expected<512*1024**3:
   remote('touch '+AMD+'/control/REQUEST_STOP; touch '+AMD+'/control/ARCHIVE_STORAGE_STOP')
   raise RuntimeError('Anta capacity gate failed')
  target.mkdir(parents=True,exist_ok=True)
  subprocess.run(['rsync','-a','--partial','--exclude=*.rst','hpcfund.amd.com:'+run+'/',str(target)+'/'],check=True,timeout=9000)
  for row in manifest:
   p=target/row['path']
   if not p.resolve().is_relative_to(target.resolve()):raise RuntimeError('unsafe manifest path')
   if p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:raise RuntimeError('archive checksum mismatch '+str(p))
  receipt=dict(source=run,destination=str(target),files=manifest,verified_utc=time.time(),job=os.environ.get('SLURM_JOB_ID'))
  atomic(target/'ARCHIVE_VERIFIED.json',receipt)
  # Only completed binary/particle/coarse-volume files are removed. Keep AMD logs,
  # inputs, waveform streams, manifests and the separately rotated checkpoints.
  cleanup=[r for r in manifest if Path(r['path']).suffix in ('.bin','.vtk','.cbin')]
  payload=json.dumps(dict(run=run,files=cleanup))
  subprocess.run(SSH+['python3 '+AMD+'/scripts/delete_archived.py'],input=payload,text=True,check=True,timeout=120)
  changed=True
 return changed

def main():
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,default=19800);p.add_argument('--once',action='store_true');a=p.parse_args()
 deadline=time.time()+a.seconds
 def heartbeat():
  while time.time()<deadline:
   try:remote("date +%s > "+AMD+"/control/ARCHIVE_HEARTBEAT")
   except Exception:pass
   time.sleep(300)
 heartbeat_thread=threading.Thread(target=heartbeat,daemon=True);heartbeat_thread.start()
 while time.time()<deadline:
  try:
   archive()
   state=json.loads(remote('cat '+AMD+'/control/state.json'))
   remote('rm -f '+AMD+'/control/ARCHIVE_ERROR')
   atomic(DEST/'evidence/amd_state.json',state)
   if state['status'] in ('complete_t12','segment_cap','user_stop','resource_limit','configuration_failure','numerical_failure','scheduler_failure','no_progress','emergency_cancel'):
    archive()
    for name in ('config.json','latest_checkpoint.json'):
     atomic(DEST/'evidence'/name,json.loads(remote('if test -f '+AMD+'/control/'+name+'; then cat '+AMD+'/control/'+name+'; else echo "{}"; fi')))
    (DEST/'evidence/executable.sha256').write_text(remote('cat '+AMD+'/evidence/executable.sha256'))
    subprocess.run(['/home/jiaxiwu/miniconda3/bin/python',str(DEST/'scripts/analyze_s8.py'),'--root',str(DEST)],check=True,timeout=5400)
    atomic(DEST/'evidence/ARCHIVE_ANALYSIS_COMPLETE.json',dict(state=state,utc=time.time()))
    return
  except Exception as e:
   with (DEST/'evidence/archive_errors.jsonl').open('a') as f:f.write(json.dumps(dict(error=str(e),utc=time.time()))+'\n')
   # Archival failure does not authorize deleting source evidence.
   try:remote('touch '+AMD+'/control/ARCHIVE_ERROR')
   except Exception:pass
   if a.once:raise
  if a.once:return
  time.sleep(300)
 atomic(DEST/'evidence/archive_window_closed_'+os.environ.get('SLURM_JOB_ID','manual')+'.json',dict(utc=time.time(),reason='finite polling window exhausted'))

if __name__=='__main__':main()
