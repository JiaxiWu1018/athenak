"""Hash closed case outputs and fixed provenance on an allocated compute node."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

AMD=Path('/work1/eliasmost/jiaxiwu/st_s09_perturbed_production_20261024')
PERSEUS=Path('/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023/perseus')

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(path):return json.loads(path.read_text())
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()

def selected(root):
    """Exclude live telemetry, this inventory and unrelated batch directories."""
    out=set()
    for name in ['runs','inputs','reduced','samples','protected','tests','interrupted','lineage']:
        out.update(p for p in (root/name).rglob('*') if p.is_file())
    for name in ['EXACT_SETUP.json','INITIAL_CONSTRAINT_ADMISSION.json','PRODUCTION_ADMISSION.json','CASE_NUMERICAL_ACCEPTANCE.json','SNAPSHOT_IMPORTED.json']:
        p=root/'evidence'/name
        if p.exists():out.add(p)
    out.update(p for p in (root/'evidence/samples').rglob('*') if p.is_file())
    for work in (root/'evidence/jobs').glob('*'):
        out.update(p for p in (work/'workflow').glob('*') if p.is_file())
        for name in ['environment.txt','workflow.sha256','launch_amd.sh','launch_perseus.sh']:
            p=work/name
            if p.exists():out.add(p)
    return sorted(out)

def identity(path):
    s=path.stat();return s,(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--source-commit',required=True);args=parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'),'Allocated compute job required'
    root=args.root.resolve();assert root.parent in [AMD,PERSEUS],root
    acceptance=load(root/'evidence/CASE_NUMERICAL_ACCEPTANCE.json');case=acceptance['case'];assert root.name==case['key']
    work=root/'runs'/case['id'];state=load(work/'state.json');assert state['complete'] and acceptance['numerical_checks_passed']
    final=load(work/'FINAL_FIELDS_OK.json');assert final['no_evolution']
    expected=state['latest']['sha256'];assert expected==acceptance['checkpoint']['sha256']==final['checkpoint']['sha256']
    packet=load(root/'review_packet/MANIFEST.json');assert packet['checkpoint']['sha256']==expected
    scopes={'case':root}
    initial=root/'evidence/INITIAL_CONSTRAINT_ADMISSION.json'
    if initial.exists():
        control=Path(load(initial)['control_root']).resolve();assert control.is_relative_to(root.parent)
        scopes['matched_control']=control
    files=[(name,p) for name,scope in scopes.items() for p in selected(scope)]
    started=now();clock=time.monotonic();rows=[];seen=set();owned=0;processed=0;next_report=0
    for index,(name,path) in enumerate(files):
        before,key=identity(path);value=digest(path);after,key_after=identity(path);assert key==key_after,('File changed during inventory',str(path))
        real=path.resolve();inside=any(real.is_relative_to(scope) for scope in scopes.values())
        if inside and key[:2] not in seen:owned+=after.st_blocks*512;seen.add(key[:2])
        row=dict(scope=name,path=str(path),relative=str(path.relative_to(scopes[name])),realpath=str(real),
          bytes=after.st_size,sha256=value,mtime_ns=after.st_mtime_ns,allocated_source_inode_bytes=after.st_blocks*512,
          owned_inode=inside,symlink_target=os.readlink(path) if path.is_symlink() else None)
        rows.append(row);processed+=row['bytes']
        if processed>=next_report:
            print(json.dumps(dict(event='manifest_progress',files=index+1,total_files=len(files),hashed_bytes=processed,elapsed_seconds=time.monotonic()-clock)),flush=True)
            next_report=processed+4*1024**3
    assert load(work/'state.json')['latest']['sha256']==expected and digest(Path(state['latest']['path']))==expected
    # Check the complete fixed-file set again, including files added or moved
    # during collection. Transient job telemetry is deliberately outside scope.
    assert [(name,p) for name,scope in scopes.items() for p in selected(scope)]==files
    for row in rows:
        s=Path(row['path']).stat();assert s.st_size==row['bytes'] and s.st_mtime_ns==row['mtime_ns']
    record=dict(format='ST8_closed_output_manifest_v1',started_UTC=started,finished_UTC=now(),host=socket.gethostname(),
      job=os.environ['SLURM_JOB_ID'],step=os.environ.get('SLURM_STEP_ID'),source_commit=args.source_commit,
      source_script=dict(path=str(Path(__file__)),sha256=digest(Path(__file__))),command=sys.argv,
      case=case,checkpoint=state['latest'],scopes={name:str(p) for name,p in scopes.items()},files=rows,
      file_count=len(rows),logical_bytes=processed,allocated_owned_unique_file_bytes=owned,
      elapsed_seconds=time.monotonic()-clock,stable_files_verified=True,
      scope='Closed raw outputs, reductions, exact inputs, fixed scientific provenance, native workflow snapshots, restart tests and matched control. Symlink targets are read-only dependencies, separately marked. Excludes transient telemetry, this inventory and unrelated batch/previous roots. Integrity inventory only; numerical/visual acceptance is separate.')
    output=root/'evidence/CLOSED_OUTPUT_MANIFEST.json'
    if output.exists():
        old=load(output);assert old['checkpoint']['sha256']==expected
        assert [(r['scope'],r['relative'],r['bytes'],r['sha256']) for r in old['files']]==[(r['scope'],r['relative'],r['bytes'],r['sha256']) for r in rows]
        print(json.dumps(dict(event='existing_manifest_verified',path=str(output),sha256=digest(output))),flush=True);return
    temporary=output.with_suffix('.json.tmp');assert not temporary.exists();temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(output)
    print(json.dumps(dict(event='manifest_complete',path=str(output),sha256=digest(output),file_count=len(rows),logical_bytes=processed,elapsed_seconds=record['elapsed_seconds'])),flush=True)

if __name__=='__main__':main()
