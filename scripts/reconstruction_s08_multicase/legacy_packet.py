"""Read-only reuse of the independently accepted original 0..500M lineage."""
import csv
import fcntl
import shutil
from common import *
from checkpoint import validate
from extension_resources import storage

def main():
    job();assert MACHINE=='amd'
    out=EXT/'legacy500_packet';out.mkdir(exist_ok=True)
    with (EXT/'evidence/legacy_packet.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (out/'MANIFEST.json').exists():return
        old=Path('/work1/eliasmost/jiaxiwu/st_s08_yc0500_extensions_20261023')
        source=old/'review/reduced/yc0500_long'
        summary=load(source/'summary.json');assert summary['complete'] and summary['endpoint']==500
        anchor=validate(old/'long/runs/yc0500/rst/yc0500.00013.rst');anchor.pop('parameters')
        assert anchor['sha256']=='7309c1b16b29d5c3de300dadce3ab37be2ba73796ae92b78ed45681134ff2e64'
        pairs=[(p,out/'reduced'/p.name) for p in source.iterdir() if p.is_file()]
        with (source/'cohort_index.csv').open() as f:
            pairs += [(Path(r['file']),out/'cohort'/Path(r['file']).name) for r in csv.DictReader(f)]
        for name in ['FINAL_ACCEPTANCE.json','INDEPENDENT_NUMERIC_QA.json','SOURCE_MAP.json','BOUNDARY_DIAGNOSTICS.json']:
            pairs.append((old/'review/evidence'/name,out/'prior_evidence'/name))
        pairs.append((old/'review/reduced/boundary_lapse.npz',out/'boundary_lapse.npz'))
        missing=sum(p.stat().st_size for p,d in pairs if not d.exists());disk=storage();assert disk['own']+missing+GIB<disk['limit']
        files=[]
        for p,d in pairs:
            before=receipt(p);d.parent.mkdir(parents=True,exist_ok=True)
            if not d.exists():shutil.copy2(p,d)
            after=receipt(d);assert before['sha256']==after['sha256'] and before['bytes']==after['bytes']
            files.append(dict(source=before,relative=str(d.relative_to(out)),bytes=after['bytes'],sha256=after['sha256']))
        atomic(out/'MANIFEST.json',dict(utc=now(),job=job(),endpoint_M=500,anchor=anchor,files=files,logical_bytes=sum(r['bytes'] for r in files),previous_roots_modified=False,scope='Previously accepted 0..500M lineage; no new simulation or revalidation claim'))

if __name__=='__main__':main()
