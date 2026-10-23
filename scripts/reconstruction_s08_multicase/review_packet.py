"""Source-bound compact cluster review bundle; no full-population raw copy."""
import shutil
from common import *
from extension_resources import storage

def checked_copy(source,dest):
    before=receipt(source);dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():shutil.copy2(source,dest)
    after=receipt(dest);assert before['sha256']==after['sha256'] and before['bytes']==after['bytes']
    return dict(source=before,relative=str(dest.relative_to(ROOT/'review_packet')),bytes=after['bytes'],sha256=after['sha256'])

def export(c):
    job();accept=load(ROOT/'evidence/CASE_NUMERICAL_ACCEPTANCE.json');assert accept['numerical_checks_passed']
    work=ROOT/'runs'/c['id'];red=ROOT/'reduced'/c['id'];packet=ROOT/'review_packet'
    sources=[(p,packet/'reduced'/p.name) for p in red.iterdir() if p.is_file()]
    sources += [(p,packet/'cohort'/p.name) for p in (work/'cohort').glob('*.s8c')]
    sources += [(p,packet/'native'/p.name) for p in work.iterdir() if p.is_file() and (p.suffix in ['.hst','.txt'] or p.name=='FINAL_FIELDS_OK.json')]
    sources += [(ROOT/'inputs'/f"{c['id']}.athinput",packet/'exact.athinput'),(ROOT/'evidence/samples'/f"{c['id']}.json",packet/'sample.json'),(ROOT/'evidence/CASE_NUMERICAL_ACCEPTANCE.json',packet/'numerical_acceptance.json'),(work/'state.json',packet/'state.json')]
    for name in ['INITIAL_CONSTRAINT_ADMISSION.json','PRODUCTION_ADMISSION.json']:
        p=ROOT/'evidence'/name
        if p.exists():sources.append((p,packet/'evidence'/name))
    missing=sum(p.stat().st_size for p,d in sources if not d.exists());disk=storage();assert disk['own']+missing+GIB<disk['limit'],('Review packet storage reservation',disk,missing)
    entries=[checked_copy(p,d) for p,d in sources]
    record=dict(utc=now(),job=job(),machine=MACHINE,case=c,source_root=str(ROOT),checkpoint=accept['checkpoint'],files=entries,logical_bytes=sum(r['bytes'] for r in entries),raw_full_population_retained_at=str(work),actual_visual_review=False)
    atomic(packet/'MANIFEST.json',record)
    return record

if __name__=='__main__':print(json.dumps(export(CASE),indent=2))
