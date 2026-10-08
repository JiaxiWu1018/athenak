#!/usr/bin/env python3
"""Verify the saved successful two-cycle run before resuming unfinished gates."""
import argparse,json,subprocess,time
from pathlib import Path
from runtime import ROOT,atomic,checkpoint,digest
from workflow import bindings,config

def check_receipt(receipt,c,run):
 if receipt.get('passed') is not True or receipt['input_sha256']!=c['input_sha256'] or receipt['script_hashes']!=c['script_hashes']:
  raise RuntimeError('saved reference acceptance is absent or bound to other scripts/input')
 if receipt['checkpoint']['sha256']!=c['resume_reference_checkpoint']['sha256']:
  raise RuntimeError('saved reference checkpoint differs from reviewed checkpoint')
 if not (run/'SEALED').exists() or (run/'EXIT_CODE').read_text().strip()!='0':
  raise RuntimeError('saved reference is not sealed successful output')
 if receipt['log_sha256']!=digest(run/'run.log') or receipt['manifest_sha256']!=digest(run/'SCIENCE_MANIFEST.json'):
  raise RuntimeError('sealed reference metadata changed')

def main():
 p=argparse.ArgumentParser();p.add_argument('--receipt-only',action='store_true');a=p.parse_args()
 bindings();c=config();run=ROOT/'runs/gate_reference';path=ROOT/'evidence/reference_resume_validation.json'
 if a.receipt_only:
  check_receipt(json.loads(path.read_text()),c,run);return
 if not c.get('resume_reference'):raise RuntimeError('saved-reference reuse was not reviewed')
 archived=json.loads((ROOT/'evidence/archived_reference_validation.json').read_text())
 if archived.get('passed') is not True or archived['script_hashes']!=c['script_hashes'] or archived['input_sha256']!=c['input_sha256']:
  raise RuntimeError('allocated Anta reference validation is absent or incorrectly bound')
 manifest={r['path']:r for r in json.loads((run/'SCIENCE_MANIFEST.json').read_text())}
 restored=archived['restored_for_restart_comparison']
 if len(restored)!=4:raise RuntimeError('saved restart-comparison inputs incomplete')
 for r in restored:
  p=run/r['path']
  if manifest.get(r['path'])!=r or p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:
   raise RuntimeError('restored reference comparison data changed')
 initial=json.loads((ROOT/'evidence/initial_validation.json').read_text())
 if not all(initial['checks'].values()):raise RuntimeError('full-particle initialization checks did not pass')
 row=checkpoint(c['resume_reference_checkpoint']['path'])
 if row!=c['resume_reference_checkpoint']:raise RuntimeError('reviewed checkpoint contents changed')
 peaks={}
 for f in run.glob('vram_*.jsonl'):
  for line in f.read_text().splitlines():
   r=json.loads(line);key=r['host']+'/'+r['card'];peaks[key]=max(peaks.get(key,0),r['fraction'])
 if len(peaks)!=48 or max(peaks.values())>=.85:raise RuntimeError('48-GPU reference memory gate failed')
 text=(run/'run.log').read_text()
 if '### FATAL ERROR' in text or '[conservation OK]' not in text:raise RuntimeError('saved reference numerical/conservation failure')
 receipt=dict(passed=True,utc=time.time(),job_id=__import__('os').environ.get('SLURM_JOB_ID'),
  checkpoint=row,input_sha256=c['input_sha256'],script_hashes=c['script_hashes'],memory_peaks=peaks,
  log_sha256=digest(run/'run.log'),manifest_sha256=digest(run/'SCIENCE_MANIFEST.json'))
 check_receipt(receipt,c,run)
 atomic(path,receipt)
 user=int(subprocess.check_output(['du','-sx','--block-size=1',str(ROOT.parent)],text=True).split()[0])
 campaign=int(subprocess.check_output(['du','-sx','--block-size=1',str(ROOT)],text=True).split()[0])
 atomic(ROOT/'evidence/reference_resume_storage.json',dict(utc=time.time(),user_bytes=user,campaign_bytes=campaign))
 if user+256*1024**3>=1.9*1024**4 or user>=1.7*1024**4 or campaign+256*1024**3>=1.25*1024**4:
  raise RuntimeError('storage reserve insufficient for remaining startup checks')
 print(json.dumps(dict(passed=True,checkpoint_time=row['time'],memory_peak=max(peaks.values()),user_bytes=user,campaign_bytes=campaign)))

if __name__=='__main__':main()
