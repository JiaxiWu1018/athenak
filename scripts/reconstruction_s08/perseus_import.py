"""Verify every copied source byte before accepting a machine migration."""
from common import *
job(); manifest=load(ROOT/'evidence/PERSEUS_EXPORT.json')
verified=[]
for x in manifest['entries']:
    p=ROOT/x['relative'];r=receipt(p)
    assert r['bytes']==x['bytes'] and r['sha256']==x['sha256'],x['relative']
    verified.append(dict(relative=x['relative'],bytes=r['bytes'],sha256=r['sha256']))
atomic(ROOT/'evidence/IMPORT_VERIFIED.json',dict(utc=now(),job=job(),source_manifest=receipt(ROOT/'evidence/PERSEUS_EXPORT.json'),files=len(verified),bytes=sum(x['bytes'] for x in verified),entries=verified,source_retained=True))
