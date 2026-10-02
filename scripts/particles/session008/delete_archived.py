#!/usr/bin/env python3
"""Metadata-only deletion of immutable, remotely checksum-verified science copies."""
import json,sys,time
from pathlib import Path
ROOT=Path('/work1/eliasmost/jiaxiwu/gi_s008_amd_20261002')
d=json.load(sys.stdin);run=Path(d['run']).resolve()
if run.parent!=ROOT/'runs' or not (run/'SEALED').exists():raise RuntimeError('cleanup outside sealed campaign run')
if not (ROOT/'README.md').exists() or not (ROOT/'inputs/gi_cluster_s8.athinput').exists():raise RuntimeError('missing reproduction records')
manifest={x['path']:x for x in json.loads((run/'SCIENCE_MANIFEST.json').read_text())}
for row in d['files']:
 p=(run/row['path']).resolve()
 if not p.is_relative_to(run) or p.suffix not in ('.bin','.vtk','.cbin') or manifest.get(row['path'])!=row:raise RuntimeError('cleanup manifest mismatch')
 if p.exists() and p.stat().st_size!=row['bytes']:raise RuntimeError('source file changed')
with (ROOT/'evidence/science_deletions.jsonl').open('a') as f:f.write(json.dumps(dict(d,utc=time.time(),basis='Anta Slurm SHA256 verified; approved archive cleanup'))+'\n')
for row in d['files']:(run/row['path']).unlink(missing_ok=True)
