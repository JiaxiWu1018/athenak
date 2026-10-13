#!/usr/bin/env python3
"""Independently verify a closed run copied to Anta; no transfer or deletion."""
import argparse,json,os,time
from pathlib import Path
from checkpoint_io import atomic,checkpoint,digest

def safe_path(root,relative):
 rel=Path(relative)
 if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe manifest path')
 path=root/rel
 if not path.resolve().is_relative_to(root.resolve()):raise ValueError('Manifest path escapes archive')
 return path

def verify(root,manifest):
 for row in manifest['files']:
  f=safe_path(root,row['path']);st=f.stat()
  if not f.is_file() or st.st_size!=row['bytes']:raise ValueError(('Archive size mismatch',row['path']))
  if digest(f)!=row['sha256']:raise ValueError(('Archive SHA256 mismatch',row['path']))
  if (f.stat().st_size,f.stat().st_mtime_ns)!=(st.st_size,st.st_mtime_ns):raise ValueError('Archive still changing')
 native=checkpoint(safe_path(root,manifest['checkpoint_relative_path']))
 for key in ['sha256','bytes','time','cycle','blocks','particles','removed']:
  if native[key]!=manifest['native_checkpoint'][key]:raise ValueError(('Native checkpoint differs',key))
 return native

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args()
 if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Anta compute allocation required')
 manifest=a.run/'SCIENCE_VERIFIED.json';m=json.loads(manifest.read_text());native=verify(a.run,m)
 atomic(a.run/'ARCHIVE_VERIFIED.json',dict(verified_epoch=time.time(),host=os.uname().nodename,job=os.environ['SLURM_JOB_ID'],source_manifest_sha256=digest(manifest),native_checkpoint=native,files_verified=len(m['files']),bytes_verified=sum(r['bytes'] for r in m['files'])))
 print('CHECKSUM-VERIFIED ARCHIVE',a.run)
if __name__=='__main__':main()
