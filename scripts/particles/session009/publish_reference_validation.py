#!/usr/bin/env python3
"""Allocated Anta saved-data validation: publish receipts and resume finite AMD gates."""
import json,os,subprocess,time
from pathlib import Path
from runtime import atomic,digest
from anta_archive import AMD,DEST,SSH,remote
from validate_initial_s7 import latest

def main():
 state=json.loads(remote('cat '+AMD+'/control/state.json'))
 c=json.loads(remote('cat '+AMD+'/control/config.json'))
 if state['status']!='reference_reviewed' or state['stop_requested'] or c.get('resume_reference') is not True:
  raise RuntimeError('saved-reference continuation is no longer permitted')
 remote('python3 '+AMD+'/scripts/workflow.py bindings')
 initial=json.loads((DEST/'evidence/initial_validation.json').read_text())
 if not all(initial['checks'].values()):raise RuntimeError('saved full-particle ledger validation failed')
 run=DEST/'runs/gate_reference'
 archive=run/'ARCHIVE_VERIFIED.json'
 records={r['path']:r for r in json.loads(archive.read_text())['files']}
 if not (run/'SEALED').exists():raise RuntimeError('archived reference is not sealed')
 restored=[]
 for pattern in ('*.part.vtk','*.z4c_xy.*.bin','*.con_xy.*.bin','*.tmunu_xy.*.bin'):
  p=latest(run,pattern);relative=str(p.relative_to(run));r=records[relative]
  if p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:raise RuntimeError('archived comparison file checksum changed')
  parent=AMD+'/runs/gate_reference/'+str(p.relative_to(run).parent)
  remote('mkdir -p '+parent)
  subprocess.run(['rsync','-a',str(p),'hpcfund.amd.com:'+parent+'/'],check=True,timeout=300)
  restored.append(r)
 receipt=dict(passed=True,utc=time.time(),job=os.environ['SLURM_JOB_ID'],source=str(run),
  archive_receipt_sha256=digest(archive),input_sha256=c['input_sha256'],script_hashes=c['script_hashes'],
  operations_revision=c['reference_repair_revision'],restored_for_restart_comparison=restored,
  method='Full saved-particle ledger, constraints, complete mesh and finite field checks on checksum-verified Anta outputs. Four reference comparison files restored without repeating evolution; allocated AMD job rehashes restored files and checkpoint.')
 atomic(DEST/'evidence/archived_reference_validation.json',receipt)
 for name in ('initial_validation.json','mesh_audit.json','archived_reference_validation.json'):
  subprocess.run(SSH+['tee '+AMD+'/evidence/'+name],input=(DEST/'evidence'/name).read_bytes(),stdout=subprocess.DEVNULL,check=True,timeout=120)
 subprocess.run(SSH+['python3 '+AMD+'/scripts/workflow.py submit'],check=True,timeout=120)
 print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
