"""Allocated integrity/link audit of the final reviewed compact delivery."""
import re
import subprocess
from session import *

def main():
    require_job()
    load(ROOT / 'evidence/QA_FINAL_OK.json')
    visual = load(ROOT / 'evidence/VISUAL_REVIEW.json')
    if visual['status'] != 'accepted':
        raise ValueError('Actual visual acceptance missing')
    for row in load(ROOT / 'evidence/DESCRIPTIVE_METRICS.json')['figures']:
        if sha(ROOT / row['path']) != row['sha256']:
            raise ValueError('Figure changed since reviewed generation')
    links = []
    for report in (ROOT / 'reports').glob('*.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', report.read_text()):
            if target.startswith(('https://', 'http://', '#')):
                continue
            path = (report.parent / target.split('#')[0]).resolve()
            if not path.exists():
                raise ValueError(f'Broken report link: {report}: {target}')
            links.append({'report': str(report.relative_to(ROOT)), 'target': target})
    repo = ROOT / 'src/athenak'
    subprocess.run(['git', '-C', str(repo), 'diff', '--quiet', 'HEAD'], check=True)
    compiled_diff = subprocess.check_output(['git', '-C', str(repo), 'diff', PHYSICS, 'HEAD', '--', 'src'], text=True)
    if compiled_diff:
        raise ValueError('Physics source differs from protected compiled source')
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    remote = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'origin/experiment/ST-reconstruction-s06-long-stability'], text=True).strip()
    if head != remote:
        raise ValueError('Final source commit not pushed')
    root_size = int(subprocess.check_output(['du', '-s', '-B1', str(ROOT)], text=True).split()[0])
    atomic(ROOT / 'evidence/CLOSEOUT_STORAGE.json', {'utc': now(), 'root': str(ROOT), 'session_allocated_bytes': root_size,
           'production_raw_logical_bytes': load(ROOT / 'evidence/OUTPUT_MANIFEST.json')['local_production_raw_logical_bytes'],
           'retention': 'All AMD originals and Perseus native analysis copies/final checkpoint retained; no deletion performed'})
    artifacts = [p for name in ('reports', 'plots', 'reduced') for p in sorted((ROOT / name).rglob('*')) if p.is_file()]
    artifacts += [ROOT / 'evidence' / name for name in
                  ('QA_FINAL_OK.json', 'VISUAL_REVIEW.json', 'SCIENCE_REVIEW.json', 'DESCRIPTIVE_METRICS.json',
                   'PERSEUS_ANALYSIS_STAGE.json', 'PERSEUS_ANALYSIS_COMPLETE.json', 'ANALYSIS_HOST_CHANGE.json',
                   'SETUP.json', 'SOURCE_PROVENANCE.json', 'AMD_ACCOUNTING.json', 'CLOSEOUT_STORAGE.json', 'STORAGE_FINAL.json')]
    manifest = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in artifacts]
    atomic(ROOT / 'evidence/FINAL_DELIVERY_MANIFEST.json', {'utc': now(), 'job': os.environ['SLURM_JOB_ID'],
           'source': head, 'source_pushed': True, 'protected_physics_source': PHYSICS, 'physics_source_diff': 'empty',
           'links_checked': links, 'files': manifest, 'raw_hash_evidence': 'PERSEUS_ANALYSIS_STAGE.json and OUTPUT_MANIFEST.json; not recomputed by compact-delivery audit'})
    atomic(ROOT / 'evidence/FINAL_AUDIT_OK.json', {'utc': now(), 'job': os.environ['SLURM_JOB_ID'], 'source': head,
           'compact_files_checked': len(manifest), 'report_links_checked': len(links), 'status': 'passed'})
    print('FINAL_AUDIT_OK', len(manifest), 'files;', len(links), 'links; source', head)

if __name__ == '__main__':
    main()
