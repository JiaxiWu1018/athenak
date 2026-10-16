"""Audit the final Perseus-only archive and refresh the delivery manifest."""
import json
import os
from pathlib import Path
import re
import subprocess

from archive_closeout import SESSION, ARCHIVE, RESTART_DEST, now, sha, write, require_job


def main():
    require_job()
    removed = json.loads((SESSION/'evidence/AMD_CLEANUP_RECEIPT.json').read_text())
    assert removed['status']=='REMOVED' and removed['campaign_allocated_bytes_after']==0
    assert json.loads((SESSION/'evidence/AMD_ARCHIVE_REPRODUCED.json').read_text())['status']=='REPRODUCED'
    old = json.loads((SESSION/'evidence/before_amd_cleanup_20261016/evidence/DELIVERY_MANIFEST.json').read_text())
    changed = {'README.md','HANDOFF.md','runs/README.md','reports/REPORT_session5.md',
               'reports/REPRODUCE.md','reports/FIGURES.md','evidence/state.json',
               'evidence/AMD_ACCOUNTING.json','evidence/CAMPAIGN_COMPLETE'}
    original_checked = 0
    for row in old['files']:
        if row['path'] in changed:
            continue
        p = SESSION/row['path']
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
        original_checked += 1
    retained = json.loads((SESSION/ARCHIVE/'closeout_records/KEEP_MANIFEST.json').read_text())
    assert len([r for r in retained['files'] if r['path'].endswith('.rst')])==1
    for row in retained['files']:
        p = SESSION/row['destination']
        if row['kind']=='file':
            assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
        else:
            assert p.is_symlink() and os.readlink(p)==row['target']
    assert (SESSION/RESTART_DEST).stat().st_size==2372815454
    documents = ['README.md','HANDOFF.md','runs/README.md','reports/REPORT_session5.md',
                 'reports/METHODS.md','reports/FIGURES.md','reports/REPRODUCE.md',
                 'reports/ARCHIVE_CLOSEOUT.md','restarts/proper_stratified_s1/README.md',
                 str(ARCHIVE/'README_ARCHIVE.md')]
    links = []
    for name in documents:
        p = SESSION/name
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if target.startswith(('https:','http:','#')):
                continue
            destination = p.parent/target.strip('<>').split('#')[0]
            assert destination.exists(),(name,target)
            links.append({'document':name,'target':target})
    excluded = {'evidence/DELIVERY_MANIFEST.json','evidence/CAMPAIGN_COMPLETE',
                'evidence/ARCHIVE_DELIVERY_AUDIT_OK.json','evidence/STORAGE_AFTER_ARCHIVE_AUDIT.json'}
    files = []
    for directory,dirs,names in os.walk(SESSION,followlinks=False):
        dirs[:] = sorted(n for n in dirs if n not in ('.git','__pycache__'))
        for name in sorted(names):
            p = Path(directory)/name
            rel = str(p.relative_to(SESSION))
            if rel in excluded or p.suffix=='.pyc' or rel.startswith('logs/archive_perseus.'):
                continue
            if p.is_symlink():
                files.append({'path':rel,'kind':'symlink','target':os.readlink(p)})
            else:
                files.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha(p)})
    write(SESSION/'evidence/DELIVERY_MANIFEST.json',{
        'job':os.environ['SLURM_JOB_ID'],'created_utc':now(),'files':files,
        'scope':'Perseus Session5 final archive; Git administration, bytecode, current audit outputs and mutable Slurm archive logs excluded; retained native source hashes independently checked',
        'previous_manifest':'evidence/before_amd_cleanup_20261016/evidence/DELIVERY_MANIFEST.json'})
    marker_path = SESSION/'evidence/CAMPAIGN_COMPLETE'
    marker = json.loads(marker_path.read_text())
    marker.update({'report_sha256':sha(SESSION/'reports/REPORT_session5.md'),
                   'delivery_manifest_sha256':sha(SESSION/'evidence/DELIVERY_MANIFEST.json'),
                   'amd_archive_closed':True,'amd_session_allocated_bytes':0,
                   'archive_report':'reports/ARCHIVE_CLOSEOUT.md','monitor_stop_requested':True})
    write(marker_path,marker)
    write(SESSION/'evidence/ARCHIVE_DELIVERY_AUDIT_OK.json',{
        'status':'PASSED','created_utc':now(),'job':os.environ['SLURM_JOB_ID'],
        'files':len(files),'unchanged_original_deliverables_checked':original_checked,
        'retained_amd_files_checked':len(retained['files']),'links':links,
        'delivery_manifest_sha256':sha(SESSION/'evidence/DELIVERY_MANIFEST.json'),
        'completion_marker_sha256':sha(marker_path),
        'source_commit':subprocess.check_output(['git','-C',str(SESSION/'src/athenak'),'rev-parse','HEAD'],text=True).strip()})
    write(SESSION/'evidence/STORAGE_AFTER_ARCHIVE_AUDIT.json',{
        'measured_utc':now(),'amd_session_allocated_bytes':0,
        'perseus_session_allocated_bytes':int(subprocess.check_output(['du','-s','-B1',str(SESSION)],text=True).split()[0]),
        'perseus_df':subprocess.check_output(['df','-B1',str(SESSION)],text=True)})
    print('ARCHIVE_DELIVERY_AUDIT_OK',len(files),'files',len(links),'links',flush=True)


if __name__=='__main__':
    main()
