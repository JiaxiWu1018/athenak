"""Inventory closed production state for an authorized serial machine migration."""
from common import *
job()
queue=subprocess.check_output(['squeue','-h','-u','jiaxiwu','-o','%i|%j|%T'],text=True)
assert not [l for l in queue.splitlines() if '|st_s08_run|' in l and '|PENDING' not in l],queue
paths=[]
for case in ['yc0500','yc0605','yc0819','yc0900']:
    paths += sorted(p for p in (ROOT/'runs'/case).rglob('*') if p.is_file())
    paths += [ROOT/'samples'/f'{case}.bin',ROOT/'inputs'/f'{case}.athinput']
entries=[]
for p in paths:
    rec=receipt(p);rec['relative']=str(p.relative_to(ROOT));entries.append(rec)
atomic(ROOT/'evidence/PERSEUS_EXPORT.json',dict(utc=now(),job=job(),source_root=str(ROOT),entries=entries,total_bytes=sum(x['bytes'] for x in entries)))
(ROOT/'evidence/PERSEUS_EXPORT.files').write_text(''.join(x['relative']+'\n' for x in entries))
