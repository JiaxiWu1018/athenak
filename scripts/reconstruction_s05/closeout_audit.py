"""Audit accepted deliverables and retain a final compact artifact manifest."""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess

from common import ROOT, atomic, require_job, sha


def main():
    require_job()
    review = json.loads((ROOT/'evidence/VISUAL_REVIEW_FINAL.json').read_text())
    assert review['status'].startswith('ACCEPTED')
    assert len(review['original_artifacts']) == 68
    for row in review['original_artifacts']+review['supplemental']['artifacts']:
        path = ROOT/row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    cases = json.loads((ROOT/'evidence/ALL24_CLOSEOUT_METADATA.json').read_text())['cases']
    assert len(cases) == len({r['case'] for r in cases}) == 24
    for row in cases:
        path = ROOT/'runs'/row['case']/'completion.json'
        assert sha(path) == row['completion_sha256']
        completion = json.loads(path.read_text())
        sample = json.loads((ROOT/'evidence/amd_build_463468/samples'/f"{row['case']}.json").read_text())
        assert completion['input_sample_sha256'] == sample['sha256'] == row['initial_sample_sha256']
        assert sample['n'] == 1500000
        assert (ROOT/'runs'/row['case']/'RUN_OK').exists()
        assert (ROOT/'runs'/row['case']/'INTERIM_100_OK').exists()
    documents = ['README.md', 'reports/REPORT_session5.md', 'reports/FIGURES.md',
                 'reports/METHODS.md', 'reports/REPRODUCE.md', 'runs/README.md']
    checked_links = []
    for name in documents:
        path = ROOT/name
        text = path.read_text()
        assert '{{' not in text
        for link in re.findall(r'\]\(([^)]+)\)', text):
            if link.startswith(('https:', 'http:', '#')):
                continue
            destination = path.parent/link.split('#')[0]
            assert destination.exists(), (name, link)
            checked_links.append({'document':name, 'target':link})
    paths = {ROOT/name for name in documents}
    for directory in ('reports', 'inputs', 'reduced', 'reduced_interim', 'runs',
                      'plots/final', 'plots/interim100', 'plots/seed_review_13673',
                      'qa_final/463854', 'science_review/463894', 'report_tables/13670'):
        paths.update(path for path in (ROOT/directory).rglob('*') if path.is_file())
    for name in ('VISUAL_REVIEW_FINAL.json', 'ALL24_CLOSEOUT_METADATA.json',
                 'retention_inventory_final.tsv', 'MIRROR_CHECK_FINAL.json',
                 'STORAGE_FINAL.json', 'AMD_ACCOUNTING_FINAL.json',
                 'ACTUAL_MODEL_REFERENCE_MANIFEST.json'):
        paths.add(ROOT/'evidence'/name)
    manifest = [{'path':str(path.relative_to(ROOT)), 'bytes':path.stat().st_size,
                 'sha256':sha(path)} for path in sorted(paths)]
    destination = ROOT/'evidence/FINAL_ARTIFACT_MANIFEST.json'
    atomic(destination, {'job':os.environ['SLURM_JOB_ID'], 'files':manifest,
                         'note':'Compact deliverables and source-backed reductions; native raw payload hashes remain in each completion manifest.'})
    repo = Path(__file__).resolve().parents[2]
    atomic(ROOT/'evidence/FINAL_AUDIT_OK.json', {
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'job':os.environ['SLURM_JOB_ID'], 'host':__import__('socket').gethostname(),
        'source_commit':subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'script_sha256':sha(Path(__file__)), 'validated_cases':24, 'original_plot_pairs':34,
        'accepted_supplemental_pairs':1, 'verified_local_links':checked_links,
        'manifest_sha256':sha(destination), 'files':len(manifest),
        'scope':'Accepted artifact identities, all24 completion/sample receipts, final-document links and compact-file manifest. Native payload validation and arithmetic were performed by the recorded AMD gates; visual review is separate.'})
    print('FINAL_AUDIT_OK', len(manifest), 'files', len(checked_links), 'links')


if __name__ == '__main__':
    main()
