#!/usr/bin/env python3
"""Allocated verification of a closed run; no transfer, cleanup or submission."""
import argparse,json,os,time
from pathlib import Path
from checkpoint_io import atomic,checkpoint,digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--result',required=True,type=Path);a=p.parse_args()
 if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm allocation required')
 r=json.loads(a.result.read_text())
 if r.get('exit_code')!=0 or not r.get('checkpoint_complete'):raise RuntimeError('Refuse unverified or failed run')
 native=checkpoint(r['checkpoint'])
 if native['sha256']!=r['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
 rows=[]
 for f in sorted(a.run.rglob('*')):
  if not f.is_file() or f.name in ('SCIENCE_VERIFIED.json','ARCHIVE_VERIFIED.json'):continue
  stat=f.stat();h=digest(f)
  if (f.stat().st_size,f.stat().st_mtime_ns)!=(stat.st_size,stat.st_mtime_ns):raise RuntimeError('Run still changing')
  rows.append(dict(path=str(f.relative_to(a.run)),bytes=stat.st_size,sha256=h))
 atomic(a.run/'SCIENCE_VERIFIED.json',dict(verified_utc=time.time(),job=os.environ['SLURM_JOB_ID'],native_checkpoint=native,checkpoint_relative_path=str(Path(r['checkpoint']).resolve().relative_to(a.run.resolve())),files=rows))
if __name__=='__main__':main()
