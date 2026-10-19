"""User-authorized Session 6 AMD cleanup; bulk work requires Slurm.

Inventory both fixed roots, archive reproduction evidence, freshly checksum
existing Perseus production copies, and delete only after a hash-bound receipt.
Existing scientific manifests and Perseus native outputs are never rewritten.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess

BASE = Path('/data/jiaxiwu/NRPIC/reconstruction_ST/06_yc0819_long_stability_20261016')
FOLLOW = BASE / 'followups/paired_scaling_20261017'
WORK = Path('/work1/eliasmost/jiaxiwu')
CONTROL = WORK / '.st_s06_cleanup_20261019'
ARCHIVE = BASE / 'archive/amd_cleanup_20261019'
SPECS = [dict(name='session6', amd=WORK/'st_reconstruction_s06_20261016', local=BASE),
         dict(name='paired_scaling', amd=WORK/'st_reconstruction_s06_paired_scaling_20261017', local=FOLLOW)]
AUTHORIZATION = 'Can you clean up the AMD data for both session 6 and session 6 followups? As long as the local doucments on perseus can reproduce those runs, delete them.'
EXE = '5fb4fb9f2131984854b3dd2eef455e8acc6808152e800698fcf5349e5bc35482'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def run(args):
    return subprocess.check_output([str(x) for x in args], text=True).strip()


def require_job():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Allocated Slurm job required')


def inventory(root):
    rows = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs.sort(); files.sort()
        for name in list(dirs)+files:
            p = Path(directory)/name
            if p.is_dir() and not p.is_symlink():
                continue
            s = p.lstat()
            row = dict(path=str(p.relative_to(root)), bytes=s.st_size,
                       allocated_bytes=s.st_blocks*512, mtime_ns=s.st_mtime_ns,
                       kind='symlink' if p.is_symlink() else 'file')
            if p.is_symlink():
                row['target'] = os.readlink(p)
                if not p.resolve().is_relative_to(root):
                    raise RuntimeError('External dependency requires explicit preservation: '+str(p))
            rows.append(row)
    return sorted(rows, key=lambda r:r['path'])


def final_paths(root):
    paths = set()
    for completion in (root/'runs').rglob('completion.json'):
        if completion.parent.name != 'production':
            continue
        state = max(load(completion)['checkpoints'], key=lambda r:r['time'])
        assert abs(state['time']-1000) < 1e-7
        paths.add(str(Path(state['path']).relative_to(root)))
    assert paths
    return paths


def disposition(row, finals):
    p = Path(row['path'])
    if '.git' in p.parts:
        return 'discard', 'Git administrative data replaced by complete verified source bundles'
    if '__pycache__' in p.parts or p.suffix == '.pyc':
        return 'discard', 'Regenerable Python bytecode'
    if p.parts[0] == 'runs' and 'out' in p.parts:
        if 'production' in p.parts and (p.suffix != '.rst' or row['path'] in finals):
            return 'existing_perseus', 'Production diagnostics and final restart already copied to Perseus'
        if p.suffix in ('.rst', '.vtk', '.bin'):
            return 'discard', 'Reproducible intermediate checkpoint or preflight native output'
    return 'archive', 'Reproduction input, initialization evidence, executable, source, log or provenance'


def queue_guard():
    queue = run(['squeue', '-u', 'jiaxiwu', '-h', '-o', '%i|%j|%T'])
    assert not any(('st_s06' in s or 'st_reconstruction_s06' in s)
                   and 'cleanup' not in s for s in queue.splitlines()), queue
    return queue


def prepare():
    require_job(); queue_guard()
    whole_before = int(run(['du', '-s', '-B1', WORK]).split()[0])
    for spec in SPECS:
        root = spec['amd']; stage = CONTROL/spec['name']
        assert root.is_dir() and root.resolve() == root
        stage.mkdir(parents=True, exist_ok=True)
        assert not (stage/'PREPARED.json').exists(), 'Inspect previous preparation before retry'
        sources = []
        repo = root/'src/athenak'
        if (repo/'.git').exists():
            for label, source in [('athenak', repo), ('kokkos', repo/'kokkos')]:
                status = run(['git', '-C', source, 'status', '--porcelain'])
                assert not status, status
                bundle = stage/(label+'.bundle')
                subprocess.run(['git', '-C', str(source), 'bundle', 'create', str(bundle), '--all'], check=True)
                subprocess.run(['git', '-C', str(source), 'bundle', 'verify', str(bundle)], check=True)
                sources.append(dict(name=label, head=run(['git','-C',source,'rev-parse','HEAD']),
                                    bundle=bundle.name, sha256=sha(bundle), status=status))
        rows = inventory(root); finals = final_paths(root)
        for i, row in enumerate(rows):
            row['action'], row['reason'] = disposition(row, finals)
            if row['action'] != 'discard' and row['kind'] == 'file':
                row['sha256'] = sha(root/row['path'])
            if i % 500 == 0:
                print('SOURCE_HASH', spec['name'], i, len(rows), flush=True)
        allocated = int(run(['du', '-s', '-B1', root]).split()[0])
        doc = dict(created_utc=now(), host=socket.gethostname(), job=os.environ['SLURM_JOB_ID'],
                   source_root=str(root), destination_root=str(spec['local']), files=rows,
                   allocated_bytes=allocated, logical_bytes=sum(r['bytes'] for r in rows),
                   sources=sources, final_restarts=sorted(finals), whole_user_bytes_before=whole_before,
                   script_sha256=sha(__file__), authorization=AUTHORIZATION)
        write(stage/'INVENTORY.json', doc)
        with (stage/'ARCHIVE_FILES0').open('wb') as stream:
            for row in rows:
                if row['action'] == 'archive':
                    stream.write(row['path'].encode()+b'\0')
        write(stage/'PREPARED.json', dict(status='PREPARED', utc=now(),
              inventory_sha256=sha(stage/'INVENTORY.json'), files=len(rows),
              archived_files=sum(r['action']=='archive' for r in rows),
              existing_perseus_files=sum(r['action']=='existing_perseus' for r in rows),
              discarded_files=sum(r['action']=='discard' for r in rows), allocated_bytes=allocated))
        print('PREPARED', spec['name'], allocated, flush=True)
    write(CONTROL/'PREPARE_OK.json', dict(utc=now(), job=os.environ['SLURM_JOB_ID'], status='PREPARED_BOTH'))


def transfer_verify():
    require_job()
    # Slurm may preempt even debug allocations here. Keep independently verified
    # per-file progress; a changed inode, size, mtime, ctime or source digest
    # invalidates the cached result. No digest is inferred from metadata alone.
    ledger = ARCHIVE/'HASH_PROGRESS.jsonl'
    progress = {}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            try:
                entry = json.loads(line)
                progress[entry['path']] = entry
            except (ValueError, KeyError):
                pass  # A partial last line after interruption has no authority.
    def verify_hash(path, expected):
        stat = path.stat()
        signature = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino]
        old = progress.get(str(path), {})
        if old.get('signature') == signature and old.get('sha256') == expected:
            return
        assert sha(path) == expected, path
        entry = dict(path=str(path), signature=signature, sha256=expected,
                     job=os.environ['SLURM_JOB_ID'], verified_utc=now())
        with ledger.open('a') as stream:
            stream.write(json.dumps(entry)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        progress[str(path)] = entry
    results = []
    for spec in SPECS:
        dest = ARCHIVE/spec['name']; records = dest/'records'
        records.mkdir(parents=True, exist_ok=True)
        subprocess.run(['rsync','-a',f'hpcfund.amd.com:{CONTROL/spec["name"]}/',str(records)+'/'],check=True)
        prepared = load(records/'PREPARED.json')
        assert sha(records/'INVENTORY.json') == prepared['inventory_sha256']
        doc = load(records/'INVENTORY.json')
        assert doc['source_root'] == str(spec['amd']) and doc['destination_root'] == str(spec['local'])
        subprocess.run(['rsync','-a','--partial','--from0','--files-from='+str(records/'ARCHIVE_FILES0'),
                        f'hpcfund.amd.com:{spec["amd"]}/',str(dest)+'/'],check=True)
        checked = []
        for i, row in enumerate(doc['files']):
            if row['action'] == 'discard':
                continue
            path = (dest if row['action']=='archive' else spec['local'])/row['path']
            if row['kind'] == 'symlink':
                assert path.is_symlink() and os.readlink(path)==row['target'], path
            else:
                assert path.is_file() and path.stat().st_size==row['bytes'], path
                verify_hash(path, row['sha256'])
            checked.append(dict(source=row['path'], destination=str(path), bytes=row['bytes'],
                                sha256=row.get('sha256'), kind=row['kind']))
            if i % 500 == 0:
                print('DESTINATION_HASH', spec['name'], i, len(doc['files']), flush=True)
        for source in doc['sources']:
            bundle = records/source['bundle']
            assert sha(bundle)==source['sha256']
            local_repo = spec['local']/'src/athenak'
            if source['name']=='kokkos': local_repo /= 'kokkos'
            subprocess.run(['git','-C',str(local_repo),'bundle','verify',str(bundle)],check=True)
        assert sha(dest/'protected/athena')==EXE
        results.append(dict(name=spec['name'], inventory_sha256=prepared['inventory_sha256'],
                            checked_files=len(checked), checked_logical_bytes=sum(r['bytes'] for r in checked),
                            source_allocated_bytes=doc['allocated_bytes']))
        write(dest/'VERIFIED_FILES.json',checked)
    # Revalidate actual initialization and complete final restart payloads without any evolution.
    import sys
    import numpy as np
    sys.path.insert(0, str(FOLLOW/'src/athenak/scripts/reconstruction_s05'))
    sys.path.insert(0, str(FOLLOW/'src/athenak/vis/python'))
    from run_case import checkpoint
    from common import sample
    payloads = []
    for spec in SPECS:
        dest = ARCHIVE/spec['name']; doc = load(dest/'records/INVENTORY.json')
        for rel in doc['final_restarts']:
            p = spec['local']/rel
            if spec['name']=='session6':
                initial = dest/'samples/proper_stratified_s1.bin'; n=1500000
            else:
                case = Path(rel).parts[1]; initial=dest/'samples'/(case+'.bin')
                _, arr=sample(initial); n=len(arr)
                init = dest/'runs'/case/'production/initial_particles.bin'
                with init.open('rb') as stream:
                    header=stream.readline(); actual=np.fromfile(stream,dtype='<f8').reshape(n,9)
                assert header.startswith(b'STMIGP05') and np.isfinite(actual).all()
                assert np.array_equal(actual[:,0],arr[:,0]) and np.all(actual[:,1]==0)
                assert np.array_equal(actual[:,2:8],arr[:,1:7]) and np.array_equal(actual[:,8],arr[:,7])
                del actual
            result=checkpoint(p,expected=1000,initial=initial,n=n)
            payloads.append(dict(path=str(p),time_M=1000,n=n,sha256=result['sha256'],payload_verified=True))
        assert (spec['local']/'src/athenak/src/pgen/particles/relativistic_cluster.cpp').is_file()
    restart = ARCHIVE/'session6/restart/proper_stratified_s1.00003.rst'
    result=checkpoint(restart,expected=300,initial=ARCHIVE/'session6/samples/proper_stratified_s1.bin',n=1500000)
    payloads.append(dict(path=str(restart),time_M=300,n=1500000,sha256=result['sha256'],payload_verified=True))
    docs = [BASE/'reports/REPORT_session6.md', FOLLOW/'reports/REPORT_paired_scaling.md',
            BASE/'scripts/README.md', FOLLOW/'scripts/README.md',
            BASE/'evidence/SOURCE_PROVENANCE.json', FOLLOW/'evidence/SOURCE_PIN.json']
    assert all(p.is_file() and p.stat().st_size>0 for p in docs)
    receipt=dict(status='VERIFIED',utc=now(),host=socket.gethostname(),job=os.environ['SLURM_JOB_ID'],
                 authorization=AUTHORIZATION,roots=results,checkpoints=payloads,
                 reports_and_reproduction_inputs=[dict(path=str(p),sha256=sha(p)) for p in docs],
                 script_sha256=sha(__file__),hash_progress_sha256=sha(ledger),
                 hash_progress='Each retained destination freshly hashed at least once across allocations; reuse requires identical inode/size/mtime/ctime and expected source SHA256.',
                 scope='Retained files freshly SHA256 verified at source and destination; full initial/final payload checks. No simulation rerun.')
    write(ARCHIVE/'VERIFIED.json', receipt)
    print('BOTH_ARCHIVES_VERIFIED',flush=True)


def delete():
    require_job(); queue=queue_guard()
    approval=load(CONTROL/'VERIFIED.json')
    assert approval['status']=='VERIFIED' and approval['authorization']==AUTHORIZATION
    assert approval['script_sha256']==sha(__file__)
    # Check every root before removing either one; an unexpected writer vetoes cleanup.
    for spec, verified in zip(SPECS,approval['roots']):
        root=spec['amd']; stage=CONTROL/spec['name']
        assert root.is_dir() and not root.is_symlink() and root.resolve()==root
        assert verified['name']==spec['name'] and sha(stage/'INVENTORY.json')==verified['inventory_sha256']
        frozen=load(stage/'INVENTORY.json')['files']; current=inventory(root)
        assert len(frozen)==len(current)
        for a,b in zip(frozen,current):
            assert all(a.get(k)==b.get(k) for k in ('path','bytes','mtime_ns','kind','target')),a['path']
    before=int(run(['du','-s','-B1',WORK]).split()[0]); removed=[]
    for spec in SPECS:
        root=spec['amd']; allocated=int(run(['du','-s','-B1',root]).split()[0])
        shutil.rmtree(root)
        assert not os.path.lexists(root)
        removed.append(dict(root=str(root),allocated_bytes_before=allocated,allocated_bytes_after=0))
    result=dict(status='REMOVED',utc=now(),host=socket.gethostname(),job=os.environ['SLURM_JOB_ID'],
                roots=removed,freed_allocated_bytes=sum(r['allocated_bytes_before'] for r in removed),
                whole_user_bytes_before=before,whole_user_bytes_after=int(run(['du','-s','-B1',WORK]).split()[0]),
                approval_sha256=sha(CONTROL/'VERIFIED.json'),queue_at_cleanup=queue,
                other_campaigns_untouched=True,script_sha256=sha(__file__))
    write(CONTROL/'REMOVED.json', result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','transfer_verify','delete'])
    mode=parser.parse_args().mode
    {'prepare':prepare,'transfer_verify':transfer_verify,'delete':delete}[mode]()
