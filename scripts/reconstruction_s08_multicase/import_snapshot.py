"""Qualify a copied snapshot in a new AMD case directory.

Only new destination copies and derived operational metadata are written. The
Perseus source and the received staging snapshot remain unchanged. This module
does not launch a solver or permanently close either continuation location.
"""
import shutil
from common import *
from checkpoint import validate
from extension_resources import storage

def materialize(stage):
    job();assert MACHINE=='amd' and CASE['key']=='yc0500_long700'
    stage=Path(stage);assert stage.parent==EXT/'staging' and ROOT!=stage
    manifest=load(stage/'SNAPSHOT_MANIFEST.json');fingerprint=sha(stage/'SNAPSHOT_MANIFEST.json')
    assert manifest['source_case_lock_acquired'] and manifest['ownership_unchanged'] and manifest['case']['key']==CASE['key']
    marker=ROOT/'evidence/SNAPSHOT_IMPORTED.json'
    if marker.exists():
        assert load(marker)['manifest_sha256']==fingerprint;return
    assert not (ROOT/'runs').exists(),'Destination must be unused; preserve an interrupted preparation for explicit recovery'
    for r in manifest['files']:
        p=stage/r['relative'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
    needed=sum(r['bytes'] for r in manifest['files']);disk=storage()
    assert disk['own']+needed+disk['checkpoint_reserve']<disk['limit']
    archive=ROOT/'lineage';archive.mkdir(parents=True,exist_ok=True)
    shutil.copy2(stage/'SNAPSHOT_MANIFEST.json',archive/'PERSEUS_SNAPSHOT_MANIFEST.json')
    # Both the staging payload and original Perseus files stay retained. The
    # new work copy is the only location that future native output may append to.
    for name in ['runs','inputs','evidence']:
        source=stage/name
        if not source.exists():continue
        destination=ROOT/'runs' if name=='runs' else archive/('source_'+name)
        shutil.copytree(source,destination)
    for r in manifest['files']:
        relative=Path(r['relative']);parts=relative.parts
        destination=ROOT/relative if parts[0]=='runs' else archive/('source_'+parts[0])/Path(*parts[1:])
        assert destination.stat().st_size==r['bytes'] and sha(destination)==r['sha256'],str(destination)
    work=ROOT/'runs'/CASE['id'];source_state=load(work/'state.json')
    shutil.copy2(work/'state.json',archive/'PERSEUS_STATE.json')
    old=Path(manifest['source_root'])
    state=source_state
    for segment in state['segments']:
        if segment.get('log'):
            segment['original_log_path']=segment['log']
            segment['log']=str(ROOT/Path(segment['log']).relative_to(old))
        segment['provenance_machine']='perseus'
    latest=validate(work/'rst'/Path(manifest['checkpoint']['path']).name);latest.pop('parameters')
    assert latest['sha256']==manifest['checkpoint']['sha256']
    latest['source_Perseus_path']=manifest['checkpoint']['path'];state['latest']=latest
    atomic(work/'state.json',state)
    if (work/'checkpoints.json').exists():shutil.copy2(work/'checkpoints.json',archive/'PERSEUS_CHECKPOINT_INDEX.json')
    atomic(work/'checkpoints.json',[latest])
    atomic(marker,dict(utc=now(),job=job(),manifest_sha256=fingerprint,source_manifest=manifest,
      checkpoint=latest,all_received_files_SHA256_verified=True,
      original_Perseus_and_staging_bytes_preserved=True,
      original_historical_commands_retained=True,only_copied_operational_log_paths_remapped=True,
      actual_destination_zero_step_gate_still_required=True))

if __name__=='__main__':
    import sys
    materialize(sys.argv[1])
