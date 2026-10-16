"""Archive the explicitly approved Session 5 subset, then gate AMD removal.

Bulk preparation, transfer, validation and reproduction require Slurm. The delete
mode only unlinks the exact inventoried campaign tree and prints its receipt;
its caller must retain stdout on Perseus, never in the tree being removed.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

AMD = Path('/work1/eliasmost/jiaxiwu/st_reconstruction_s05_20261014')
SESSION = Path('/data/jiaxiwu/NRPIC/reconstruction_ST/05_sampling_shell_response_20261014')
ARCHIVE = Path('archive/amd_closeout_20261016')
STAGE = 'closeout_transfer'
RESTART_SOURCE = 'runs/proper_stratified_s1/out/rst/proper_stratified_s1.00003.rst'
RESTART_DEST = 'restarts/proper_stratified_s1/proper_stratified_s1.00003.rst'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def require_job():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Bulk archive work requires Slurm')


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def disposition(rel, size):
    p = Path(rel)
    parts = p.parts
    if '.git' in parts:
        return False, 'Git object/administrative data replaced by verified bundles and source records'
    if rel == RESTART_SOURCE:
        return True, 'Approved proper stratified seed1 final restart'
    if parts[0] == 'runs':
        if p.name == 'initial_particles.bin':
            return True, 'Full double-precision production initialization evidence'
        if p.suffix in ('.rst', '.vtk', '.bin'):
            return False, 'Native production particle/field/checkpoint output outside approved retention'
    if parts[0] in ('preflight', 'preflight_attempts'):
        if p.name == 'initial_particles.bin' and size < 1000000:
            return True, 'Small historical-default initialization evidence'
        if p.suffix in ('.rst', '.vtk', '.bin'):
            return False, 'Bulk preflight native state; logs and verification receipts retained'
    if parts[0] in ('build', 'baseline_build'):
        keep = p.suffix in ('.txt', '.cmake', '.log', '.json', '.yaml', '.yml')
        return keep, 'Build metadata retained; intermediates and duplicate binaries discarded'
    if '__pycache__' in parts or p.suffix == '.pyc':
        return False, 'Regenerable Python bytecode'
    return True, 'Document, exact input, compact product, protected executable or source snapshot'


def inventory(root):
    rows = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        files.sort()
        if Path(directory) == root:
            dirs[:] = [d for d in dirs if d != STAGE]
        for name in list(dirs)+files:
            p = Path(directory)/name
            if p.is_dir() and not p.is_symlink():
                continue
            s = p.lstat()
            rel = str(p.relative_to(root))
            keep, why = disposition(rel, s.st_size)
            row = {'path':rel, 'bytes':s.st_size, 'allocated_bytes':s.st_blocks*512,
                   'mtime_ns':s.st_mtime_ns, 'kind':'symlink' if p.is_symlink() else 'file',
                   'retain':keep, 'reason':why}
            if p.is_symlink():
                row['target'] = os.readlink(p)
                if not p.resolve().is_relative_to(root):
                    raise RuntimeError('Unexpected external symlink: '+rel)
            rows.append(row)
    return sorted(rows, key=lambda r:r['path'])


def prepare():
    require_job()
    assert AMD.is_dir() and AMD.resolve() == AMD
    stage = AMD/STAGE
    stage.mkdir(exist_ok=True)
    repo = AMD/'src/athenak'
    provenance = {'created_utc':now(), 'host':socket.gethostname(),
                  'job':os.environ['SLURM_JOB_ID'], 'sources':[]}
    worktrees = command(['git', '-C', str(repo), 'worktree', 'list', '--porcelain'])
    provenance['worktrees_porcelain'] = worktrees
    for source in [repo, *sorted((AMD/'qa_source').iterdir())]:
        if not source.is_dir():
            continue
        status = command(['git', '-C', str(source), 'status', '--porcelain'])
        if status:
            raise RuntimeError('Unrecorded source changes: '+str(source)+' '+status)
        provenance['sources'].append({'path':str(source.relative_to(AMD)),
            'commit':command(['git','-C',str(source),'rev-parse','HEAD']), 'status':status})
    for name, source in [('athenak',repo), ('kokkos',repo/'kokkos')]:
        bundle = stage/(name+'.bundle')
        assert not bundle.exists(), bundle
        subprocess.run(['git','-C',str(source),'bundle','create',str(bundle),'--all'],check=True)
        subprocess.run(['git','-C',str(source),'bundle','verify',str(bundle)],check=True)
        provenance[name+'_bundle'] = {'path':bundle.name,'bytes':bundle.stat().st_size,'sha256':sha(bundle)}
    provenance['toolchain'] = (stage/'toolchain.txt').read_text()
    provenance['protected_libraries'] = command(['ldd',str(AMD/'protected/athena')])
    write(stage/'SOURCE_PROVENANCE.json',provenance)
    rows = inventory(AMD)
    retained = []
    expected = {}
    for completion in (AMD/'runs').glob('*/completion.json'):
        for row in json.loads(completion.read_text())['manifest']:
            expected[row['path']] = row
    for receipt in (AMD/'evidence/samples').glob('*.json'):
        d = json.loads(receipt.read_text())
        expected['samples/'+receipt.stem+'.bin'] = d
    for i,row in enumerate(rows):
        if not row['retain']:
            continue
        out = dict(row)
        out['destination'] = RESTART_DEST if row['path']==RESTART_SOURCE else str(ARCHIVE/row['path'])
        if row['kind'] == 'file':
            out['sha256'] = sha(AMD/row['path'])
            if row['path'] in expected:
                e = expected[row['path']]
                assert out['bytes']==e['bytes'] and out['sha256']==e['sha256'], row['path']
        retained.append(out)
        if i%500 == 0:
            print('HASH',i,len(rows),flush=True)
    inv = {'created_utc':now(),'root':str(AMD),'job':os.environ['SLURM_JOB_ID'],
           'files':rows,'stage_excluded':STAGE,'files_count':len(rows),
           'logical_bytes':sum(r['bytes'] for r in rows),
           'allocated_file_bytes':sum(r['allocated_bytes'] for r in rows)}
    write(stage/'INVENTORY.json',inv)
    manifest = {'created_utc':now(),'source_root':str(AMD),'destination_root':str(SESSION),
                'job':os.environ['SLURM_JOB_ID'],'files':retained,
                'retained_logical_bytes':sum(r['bytes'] for r in retained),
                'inventory_sha256':sha(stage/'INVENTORY.json'),
                'scope':'Approved seed1 restart and compact products; exact production initialization inputs/evidence retained'}
    write(stage/'KEEP_MANIFEST.json',manifest)
    with (stage/'KEEP_FILES0').open('wb') as f:
        for r in retained:
            if r['path'] != RESTART_SOURCE:
                f.write(r['path'].encode()+b'\0')
    summary = {'status':'PREPARED','created_utc':now(),'job':os.environ['SLURM_JOB_ID'],
               'files':len(rows),'retained_files':len(retained),
               'retained_logical_bytes':manifest['retained_logical_bytes'],
               'manifest_sha256':sha(stage/'KEEP_MANIFEST.json'),
               'inventory_sha256':sha(stage/'INVENTORY.json')}
    write(stage/'PREPARED.json',summary)
    print(json.dumps(summary),flush=True)


def transfer():
    require_job()
    archive = SESSION/ARCHIVE
    records = archive/'closeout_records'
    records.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync','-a','--partial','hpcfund.amd.com:'+str(AMD/STAGE)+'/',str(records)+'/'],check=True)
    prep = json.loads((records/'PREPARED.json').read_text())
    assert sha(records/'KEEP_MANIFEST.json')==prep['manifest_sha256']
    assert sha(records/'INVENTORY.json')==prep['inventory_sha256']
    subprocess.run(['rsync','-a','--partial','--from0','--files-from='+str(records/'KEEP_FILES0'),
                    'hpcfund.amd.com:'+str(AMD)+'/',str(archive)+'/'],check=True)
    target = SESSION/RESTART_DEST
    target.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','--partial','hpcfund.amd.com:'+str(AMD/RESTART_SOURCE),str(target)],check=True)
    verify()


def verify():
    require_job()
    import numpy as np
    records = SESSION/ARCHIVE/'closeout_records'
    manifest = json.loads((records/'KEEP_MANIFEST.json').read_text())
    assert manifest['source_root']==str(AMD) and manifest['destination_root']==str(SESSION)
    for i,row in enumerate(manifest['files']):
        p = SESSION/row['destination']
        if row['kind']=='symlink':
            assert p.is_symlink() and os.readlink(p)==row['target'], row['path']
        else:
            assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'], row['path']
        if i%500==0:
            print('VERIFY',i,len(manifest['files']),flush=True)
    prov = json.loads((records/'SOURCE_PROVENANCE.json').read_text())
    for name in ('athenak','kokkos'):
        bundle = records/(name+'.bundle')
        assert sha(bundle)==prov[name+'_bundle']['sha256']
        subprocess.run(['git','-C',str(SESSION/'src/athenak' if name=='athenak' else SESSION/'src/athenak/kokkos'),
                        'bundle','verify',str(bundle)],check=True)
    initial_checks = []
    for case_dir in sorted((SESSION/'runs').iterdir()):
        if not (case_dir/'completion.json').is_file():
            continue
        case = case_dir.name
        path = SESSION/ARCHIVE/'samples'/(case+'.bin')
        with path.open('rb') as f:
            header = f.readline().decode().split()
            a = np.fromfile(f,'<f8').reshape(1500000,8)
        assert header[0]=='STSAMP05' and np.isfinite(a).all()
        assert np.array_equal(a[:,0],np.arange(1500000)) and np.all(a[:,7]>0)
        dump = SESSION/ARCHIVE/'runs'/case/'initial_particles.bin'
        with dump.open('rb') as f:
            h = f.readline().decode()
            b = np.fromfile(f,'<f8').reshape(1500000,9)
        assert h.startswith('STMIGP05') and np.isfinite(b).all()
        assert np.array_equal(b[:,0],a[:,0]) and np.all(b[:,1]==0)
        assert np.array_equal(b[:,2:8],a[:,1:7]) and np.array_equal(b[:,8],a[:,7]),case
        initial_checks.append({'case':case,'count':len(a),'mass':float(a[:,7].sum()),
                               'native_initialization_exact':True})
        print('INITIAL_OK',case,flush=True)
    assert len(initial_checks)==24
    from run_case import checkpoint
    result = checkpoint(SESSION/RESTART_DEST,300.,SESSION/ARCHIVE/'samples/proper_stratified_s1.bin')
    expected = json.loads((SESSION/'runs/proper_stratified_s1/completion.json').read_text())['checkpoints'][-1]
    assert result['bytes']==expected['bytes'] and result['sha256']==expected['sha256']
    import s8_checkpoint
    _,deck = s8_checkpoint.header(SESSION/RESTART_DEST)
    restart_dir = (SESSION/RESTART_DEST).parent
    (restart_dir/'embedded.athinput').write_bytes(deck)
    write(restart_dir/'VALIDATED.json',result)
    # Restore the authoritative actual-AMD receipts at the plotting API's local path.
    for source in sorted((SESSION/ARCHIVE/'evidence/samples').glob('*.json')):
        target = SESSION/'evidence/samples'/source.name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            assert sha(target)==sha(source),target
        else:
            shutil.copy2(source,target)
    result = {'status':'VERIFIED','created_utc':now(),'job':os.environ['SLURM_JOB_ID'],
              'host':socket.gethostname(),'manifest_sha256':sha(records/'KEEP_MANIFEST.json'),
              'verified_files':len(manifest['files']),
              'retained_logical_bytes':manifest['retained_logical_bytes'],
              'initial_checks':initial_checks,'restart':result,
              'source_bundles_verified':True,'source_script_sha256':sha(Path(__file__))}
    write(SESSION/'evidence/AMD_ARCHIVE_VERIFIED.json',result)
    print('ARCHIVE_VERIFIED',len(manifest['files']),flush=True)


def reproduce():
    require_job()
    archive_ok = json.loads((SESSION/'evidence/AMD_ARCHIVE_VERIFIED.json').read_text())
    assert archive_ok['status']=='VERIFIED'
    work = SESSION/'archive_validation'/os.environ['SLURM_JOB_ID']
    work.mkdir(parents=True,exist_ok=False)
    for name in ['reduced','reduced_interim','runs','samples','reference']:
        (work/name).symlink_to(SESSION/name,target_is_directory=True)
    (work/'evidence').mkdir()
    (work/'evidence/samples').symlink_to(SESSION/'evidence/samples',target_is_directory=True)
    shutil.copy2(SESSION/'evidence/PREFLIGHT_OK',work/'evidence/PREFLIGHT_OK')
    (work/'reports').mkdir()
    cache = Path('particle_visualization_data/464699')
    (work/cache.parent).mkdir()
    (work/cache).symlink_to(SESSION/cache,target_is_directory=True)
    scripts = SESSION/'src/athenak/scripts/reconstruction_s05'
    env = os.environ.copy()
    env['ROOT'] = str(work)
    for args in [['plot_report.py'],['plot_report.py','--interim'],
                 ['particle_visual_render.py',str(work/cache/'DATA_OK.json')]]:
        subprocess.run([sys.executable,str(scripts/args[0]),*args[1:]],env=env,check=True)
    pairs = {}
    for which in ['final','interim100']:
        files = list((work/'plots'/which).glob('f*.png'))
        assert len(files)==17
        assert all(p.with_suffix('.pdf').is_file() for p in files)
        pairs[which] = len(files)
    render = json.loads((work/'particle_visualization_render'/os.environ['SLURM_JOB_ID']/'RENDER_OK.json').read_text())
    assert len(render['promoted_artifacts'])==6 and render['frames']==151
    receipt = {'status':'REPRODUCED','created_utc':now(),'job':os.environ['SLURM_JOB_ID'],
               'host':socket.gethostname(),'work':str(work.relative_to(SESSION)),
               'figure_pairs':pairs,'movie_frames':151,'movie_and_initial_outputs':6,
               'source_commit':command(['git','-C',str(SESSION/'src/athenak'),'rev-parse','HEAD']),
               'manifest_sha256':archive_ok['manifest_sha256'],
               'scope':'All five original figure families at both endpoints and both movies plus initial figures regenerated solely from Perseus compact data; no AMD access'}
    write(SESSION/'evidence/AMD_ARCHIVE_REPRODUCED.json',receipt)
    print('PERSEUS_ONLY_REPRODUCED',flush=True)


def delete(receipt_path):
    # Explicit fixed-root deletion, gated on both independent Perseus receipts.
    assert AMD.resolve()==AMD and AMD.is_dir() and not AMD.is_symlink()
    approval = json.loads(Path(receipt_path).read_text())
    assert approval['authorization']=='User approved Session5 archive and deletion plan, then instructed Implement the plan.'
    stage = AMD/STAGE
    digest = sha(stage/'KEEP_MANIFEST.json')
    assert approval['verification']['status']=='VERIFIED'
    assert approval['reproduction']['status']=='REPRODUCED'
    assert approval['verification']['manifest_sha256']==approval['reproduction']['manifest_sha256']==digest
    frozen = json.loads((stage/'INVENTORY.json').read_text())['files']
    current = inventory(AMD)
    assert len(current)==len(frozen)
    for a,b in zip(current,frozen):
        assert all(a.get(k)==b.get(k) for k in ('path','bytes','mtime_ns','kind','target')),a['path']
    queue = command(['squeue','-u','jiaxiwu','-h','-o','%i|%j|%T'])
    assert not any('st_s05' in s or 'st_reconstruction_s05' in s for s in queue.splitlines()),queue
    processes = command(['ps','-u','jiaxiwu','-o','pid=,ppid=,args='])
    # Permit this process and its ssh shell ancestors; reject other campaign users.
    permitted = {os.getpid()}
    pid = os.getppid()
    while pid>1:
        permitted.add(pid)
        try:
            fields = Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
            pid = int(fields[1])
        except (OSError,ValueError):
            break
    for line in processes.splitlines():
        fields = line.strip().split(None,2)
        if len(fields)<3:
            continue
        if ('st_reconstruction_s05' in fields[2] or 'reconstruction_s05' in fields[2]) and int(fields[0]) not in permitted:
            raise RuntimeError('Active Session5 process: '+line)
    campaign_before = int(command(['du','-s','-B1',str(AMD)]).split()[0])
    user_before = int(command(['du','-s','-B1',str(AMD.parent)]).split()[0])
    shutil.rmtree(AMD)
    assert not os.path.lexists(AMD)
    candidates = []
    for base in [AMD.parent,Path.home()]:
        for p in base.iterdir():
            if any(s in p.name.lower() for s in ['st_reconstruction_s05','st_s05','reconstruction_s05','session5','sampling_shell_response']):
                candidates.append(str(p))
    assert not candidates,candidates
    result = {'status':'REMOVED','created_utc':now(),'host':socket.gethostname(),
              'deleted_root':str(AMD),'campaign_allocated_bytes_before':campaign_before,
              'campaign_allocated_bytes_after':0,'inventoried_original_files':len(frozen),
              'source_manifest_sha256':digest,'whole_user_bytes_before':user_before,
              'whole_user_bytes_after':int(command(['du','-s','-B1',str(AMD.parent)]).split()[0]),
              'other_campaigns_untouched':True,'remaining_named_candidates':candidates,
              'queue_at_cleanup':queue,'scope':'Session5-owned files only; unrelated AMD campaigns remain'}
    print(json.dumps(result),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','transfer','verify','reproduce','delete'])
    p.add_argument('--receipt')
    a = p.parse_args()
    if a.mode=='delete':
        delete(a.receipt)
    else:
        globals()[a.mode]()


if __name__=='__main__':
    main()
