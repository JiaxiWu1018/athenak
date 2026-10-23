"""One-way compact AMD review import; run on a Perseus compute allocation."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

E = Path('/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_long_and_inward_20261023')
AMD = '/work1/eliasmost/jiaxiwu/st_s08_yc0500_extensions_20261023'
HOST = 'hpcfund.amd.com'


def remote(args):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', HOST,
                                    shlex.join(list(map(str, args)))], timeout=180)


def read_remote(relative):
    return json.loads(remote(['cat', AMD + '/' + relative]))


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Compute allocation required'
    start = time.monotonic()
    record = E / 'evidence/perseus_jobs' / os.environ['SLURM_JOB_ID']
    record.mkdir(parents=True, exist_ok=False)
    (record / 'executed_mirror.py').write_bytes(Path(__file__).read_bytes())
    env = subprocess.check_output(['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID']], text=True)
    (record / 'environment.txt').write_text(socket.gethostname() + '\n' + env)
    jobs = read_remote('evidence/jobs.json')
    text = remote(['sacct', '-X', '-j', ','.join(str(r['id']) for r in jobs), '-n', '-P',
                   '-o', 'JobIDRaw,State,ExitCode,ElapsedRaw,AllocNodes,Start,End,Partition,NodeList']).decode()
    rows = [line.split('|') for line in text.splitlines() if line.strip()]
    assert {r[0] for r in rows} == {str(r['id']) for r in jobs}
    assert all(r[1].split('+')[0] in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT',
                                      'OUT_OF_MEMORY', 'PREEMPTED', 'NODE_FAIL'] for r in rows), rows
    (record / 'amd_terminal_sacct.psv').write_text(text)
    total = sum(int(r[3] or 0) * int(r[4] or 0) / 3600 for r in rows)
    assert total < 39, 'Keep finalization inside the 40-node-hour review allowance'
    manifest = read_remote('review/evidence/REVIEW_PRODUCT_MANIFEST.json')
    files = {}
    for item in manifest['products']:
        relative = Path(item['relative_path'])
        assert relative.parts[0] == 'review' and '..' not in relative.parts
        target = str(relative.relative_to('review'))
        assert target not in files
        files[target] = item
    assert sum(r['bytes'] for r in files.values()) < 1024 ** 3, 'Thin review copy exceeded 1GiB'
    # Verification receipts remain compact; no raw/reduced arrays, logs, source
    # tree, lossless frame sequence or checkpoint is imported in this mirror.
    evidence = ['REVIEW_PRODUCT_MANIFEST', 'CLOSED_DATA_MANIFEST', 'PROVENANCE',
                'SOURCE_MAP', 'INDEPENDENT_NUMERIC_QA', 'COVERAGE_AND_GLOBAL_CONSTRAINTS', 'PLOTS_RENDERED',
                'MOVIES_RENDERED', 'PDF_RASTERS', 'BOUNDARY_DIAGNOSTICS',
                'PRODUCTS_READY', 'ACCOUNTING_OBSERVED', 'STORAGE_REVIEW']
    dest = E / 'review_mirror'
    (dest / 'evidence').mkdir(parents=True, exist_ok=True)
    for name in evidence:
        payload = remote(['cat', AMD + '/review/evidence/' + name + '.json'])
        assert len(payload) < 64 * 1024 ** 2
        (dest / 'evidence' / (name + '.json')).write_bytes(payload)
    listpath = record / 'review.files'
    listpath.write_text('\n'.join(sorted(files)) + '\n')
    with (record / 'rsync.log').open('w') as f:
        subprocess.run(['rsync', '-a', '--stats', '--files-from=' + str(listpath),
                        HOST + ':' + AMD + '/review/', str(dest) + '/'],
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    verified = []
    for relative, item in sorted(files.items()):
        path = dest / relative
        assert path.is_file() and path.stat().st_size == item['bytes'], relative
        digest = sha(path)
        assert digest == item['sha256'], relative
        verified.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest,
                             source=AMD + '/review/' + relative))
    amdbytes = int(remote(['du', '-s', '-B1', AMD]).split()[0])
    localbytes = int(subprocess.check_output(['du', '-s', '-B1', str(E)]).split()[0])
    assert amdbytes + localbytes < 200 * 1024 ** 3
    write(E / 'evidence/AMD_FINAL_ACCOUNTING.json', dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        terminal=True, allocated_node_hours=total,
        scheduler_rows=rows, fields=['JobIDRaw', 'State', 'ExitCode', 'ElapsedRaw',
                                    'AllocNodes', 'Start', 'End', 'Partition', 'NodeList'],
        scope='All registered extension AMD jobs, including failed and unstarted cancelled allocations. Excludes the separate Perseus mirror allocation, whose terminal accounting follows.'))
    write(E / 'evidence/MIRROR_VERIFIED.json', dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        job=os.environ['SLURM_JOB_ID'], host=socket.gethostname(),
        files=verified, count=len(verified), bytes=sum(r['bytes'] for r in verified),
        source_root=AMD + '/review', mirror_root=str(dest), one_way=True,
        AMD_extension_allocated_bytes=amdbytes, Perseus_extension_allocated_bytes=localbytes,
        combined_allocated_bytes=amdbytes + localbytes,
        Mac_delivery=False, actual_visual_review='pending',
        elapsed_before_exit_seconds=time.monotonic() - start))
    (dest / 'README.md').write_text(
        '# Session 8 extension review mirror\n\n'
        'Reports, exact inputs, final static figures and movies are copied one-way from the AMD extension review root. '
        'MIRROR_VERIFIED.json in the parent evidence folder binds every copied product by SHA256. '
        'Raw data, reduced arrays, lossless frame sequences, code and logs remain on the clusters. '
        'Actual scientific/visual acceptance and the final combined scheduler account are separate closing records. '
        'No Mac delivery is claimed.\n')
    print(json.dumps(dict(files=len(verified), bytes=sum(r['bytes'] for r in verified),
                          AMD_node_hours=total, combined_storage_GiB=(amdbytes + localbytes) / 1024 ** 3)))


if __name__ == '__main__':
    main()
