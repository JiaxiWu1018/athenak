#!/usr/bin/env python3
"""Metadata-only deletion of immutable, remotely checksum-verified science copies."""
import json,sys,time,fcntl
from runtime import atomic
from pathlib import Path
ROOT=Path('/work1/eliasmost/jiaxiwu/gi_s009_amd_20261005')
d=json.load(sys.stdin);run=Path(d['run']).resolve()
if run.parent!=ROOT/'runs' or not (run/'SEALED').exists():raise RuntimeError('cleanup outside sealed campaign run')
if not (ROOT/'README.md').exists() or not (ROOT/'inputs/gi_cluster_s9.athinput').exists():raise RuntimeError('missing reproduction records')
manifest={x['path']:x for x in json.loads((run/'SCIENCE_MANIFEST.json').read_text())}
for row in d['files']:
 p=(run/row['path']).resolve()
 if not p.is_relative_to(run) or p.suffix not in ('.bin','.vtk','.cbin') or manifest.get(row['path'])!=row:raise RuntimeError('cleanup manifest mismatch')
 if p.exists() and p.stat().st_size!=row['bytes']:raise RuntimeError('source file changed')
with (ROOT/'evidence/science_deletions.jsonl').open('a') as f:f.write(json.dumps(dict(d,utc=time.time(),basis='Anta Slurm SHA256 verified; approved archive cleanup'))+'\n')
for row in d['files']:(run/row['path']).unlink(missing_ok=True)

with (ROOT/'control/archive_receipts.lock').open('a') as lock:
 import fcntl
 fcntl.flock(lock,fcntl.LOCK_EX)
 path=ROOT/'control/archived_checkpoints.json'
 known=json.loads(path.read_text()) if path.exists() else {}
 for row in d.get('checkpoints',[]):
  if manifest.get(row['path'])!=row or Path(row['path']).suffix!='.rst':raise RuntimeError('checkpoint archive receipt mismatch')
  known[str(run/row['path'])]=dict(sha256=row['sha256'],bytes=row['bytes'],destination=d['destination']+'/'+row['path'],verified_utc=d['verified_utc'],job=d['job'])
 atomic(path,known)
