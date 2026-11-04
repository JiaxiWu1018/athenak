"""Inventory immutable scientific outputs and admit a bounded, additive Anta archive."""
import json
import os
from pathlib import Path
from campaign import ROOT, resource_snapshot
from state_io import require_job, sha, write_json


def main():
    require_job();files={};segments=[]
    def add(path):
        rel=str(path.relative_to(ROOT));assert path.is_file() and not path.is_symlink()
        files[rel]=dict(path=rel,bytes=path.stat().st_size,sha256=sha(path))
    cases=[f'fixed_fast_v1_{n}m_h010' for n in (1,2,4,8)]+['fixed_fast_v1_1m_h005','fixed_fast_v1_1m_h0025']
    for case in cases:
        endpoints=[]
        for manifest in sorted((ROOT/'runs'/case).glob('*/OUTPUT_MANIFEST.json')):
            complete=manifest.parent/'COMPLETE.json'
            if not complete.exists():continue
            record=json.loads(complete.read_text());endpoints.append(record['target'])
            assert record['checkpoint']['finite'] and record['checkpoint']['original_ids_masses']
            assert record['solver']['failures']==0
            for r in json.loads(manifest.read_text()):
                rel=Path(r['path']);assert not rel.is_absolute() and '..' not in rel.parts
                path=manifest.parent/rel
                assert path.is_file() and not path.is_symlink() and path.stat().st_size==r['bytes']
                name=str(path.relative_to(ROOT));assert name not in files
                files[name]=dict(path=name,bytes=r['bytes'],sha256=r['sha256'])
            add(manifest);add(complete);segments.append(str(manifest.parent.relative_to(ROOT)))
        assert max(endpoints)==(500 if case.endswith('h010') else 100)
    assert len(segments)==29
    # Small, immutable reductions, review packet, native source and reproduction helpers.
    selected=[ROOT/'analysis/review_499543',ROOT/'analysis/job_495334/reduced',
              ROOT/'src/athenak/src',ROOT/'src/athenak/kokkos',ROOT/'src/athenak/scripts/reconstruction_s10']
    selected += [ROOT/f'reference/session7/reduced/mc_{n}m_s1' for n in (1,2,4,8)]
    for directory in selected:
        assert directory.is_dir(),directory
        for path in sorted(directory.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and '.git' not in path.parts:add(path)
    for name in ('CMakeLists.txt','config.hpp.in','.gitmodules','LICENSE'):
        add(ROOT/'src/athenak'/name)
    for name in ('athena.rk3','athena.fixed_v1','athena.fixed_fast_v1'):add(ROOT/'protected'/name)
    for name in ('FIXED_V1_OK.json','FIXED_FAST_V1_OK.json','FIXED_FAST_V1_SOURCE.json',
                 'SCHEME_VARIANTS.json','ALL_PRODUCTION_OK.json','IMPLEMENTATION_SPEC_REVIEW.json',
                 'OPTIMIZATION_TOLERANCE_498261.json'):
        add(ROOT/'evidence'/name)
    assert json.loads((ROOT/'analysis/review_499543/REVIEW.json').read_text())['science_and_visual_review_complete']
    resources=resource_snapshot();total=sum(r['bytes'] for r in files.values())
    perseus_bytes=1124110336  # Recorded local du at final scientific review; reserve below covers metadata growth.
    anta_review_reserve=64*2**20;growth_reserve=2*2**30
    projected=resources['campaign_bytes']+perseus_bytes+anta_review_reserve+total+growth_reserve
    forecast=resources['node_hours']+3/3600+1+.25  # Anta PDF job, archive1h cap, closeout reserve.
    assert projected<300*2**30 and forecast<40,(projected,forecast)
    assert not resources['stop']
    write_json(ROOT/'evidence/ARCHIVE_PLAN.json',dict(status='READY_FOR_ARCHIVE',source_root=str(ROOT),
        inventory_job=os.environ['SLURM_JOB_ID'],inventory_script_sha256=sha(__file__),
        deletion_authorized=False,transfer_started=False,source_originals_retained=True,
        segments=segments,total_bytes=total,files=[files[k] for k in sorted(files)],resources=resources,
        perseus_snapshot_bytes=perseus_bytes,anta_review_reserve_bytes=anta_review_reserve,
        additional_growth_reserve_bytes=growth_reserve,projected_all_copies_bytes=projected,
        projected_total_node_hours=forecast,archive_job_max_node_hours=1,
        scope='Additive archive of all29 primary/fine-control segments, reviewed final packet, cached legacy reductions, archived comparison tables, native source, scripts, protected executables and key qualification evidence. All other historical/failed raw data remain retained on AMD.',
        checksum_method='Closed native manifests supply raw-file hashes; this allocation verifies source sizes and hashes added metadata/source/executable files. Anta recomputes every destination SHA256. No deletion or bidirectional sync.'))


if __name__=='__main__':main()
