"""Transfer final media only; delegate destination hashing to owned AMD Slurm."""
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

E = Path('/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_long_and_inward_20261023')
X = '/work1/eliasmost/jiaxiwu/st_s08_yc0500_extensions_20261023'
HOST = 'hpcfund.amd.com'


def remote(args):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', HOST,
        shlex.join(['env', 'TZ=UTC', *map(str, args)])], timeout=120)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    jid = sys.argv[1]
    assert jid.isdigit()
    registry = json.loads((E / 'evidence/jobs.json').read_text())
    assert any(r['id'] == jid and r['mode'] == 'thin_verify' for r in registry)
    source = X + '/thin_verify/evidence/jobs/' + jid
    ready = json.loads(remote(['cat', source + '/READY.json']))
    assert ready['job'] == jid and ready['source_root'] == X + '/review'
    out = E / 'evidence/transfer_jobs' / jid
    out.mkdir(parents=True, exist_ok=False)
    (out / 'executed_client.py').write_bytes(Path(__file__).read_bytes())
    write(out / 'READY.json', ready)
    paths = sorted(ready['files'])
    for name in paths:
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert path.parts[0] in ['plots', 'movies', 'reports', 'inputs', 'evidence']
    listing = out / 'FILES.nul'
    listing.write_bytes(b'\0'.join(name.encode() for name in paths) + b'\0')
    destination = E / 'review_mirror'
    destination.mkdir(exist_ok=True)
    assert destination.resolve().parent == E.resolve()
    with (out / 'rsync.log').open('w') as log:
        subprocess.run(['rsync', '-a', '--from0', '--files-from=' + str(listing),
            HOST + ':' + X + '/review/', str(destination) + '/'],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    for name in paths:
        path = destination / name
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == ready['files'][name]['bytes']
    with (out / 'stream.log').open('w') as log:
        archive = subprocess.Popen(['tar', '-C', str(destination), '-cf', '-',
            '--null', '-T', str(listing)], stdout=subprocess.PIPE, stderr=log)
        args = ['env', 'TZ=UTC', 'srun', '--overlap', '--jobid=' + jid,
            '-N', '1', '-n', '1', '-c', '1', 'python3',
            source + '/workflow/thin_verify.py', '--stream']
        verify = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', HOST, shlex.join(args)],
            stdin=archive.stdout, stdout=log, stderr=subprocess.STDOUT)
        archive.stdout.close()
        assert verify.wait(timeout=180) == 0
        assert archive.wait(timeout=30) == 0
    checked = json.loads(remote(['cat', source + '/DESTINATION_VERIFIED.json']))
    assert checked['files'] == ready['files']
    write(out / 'DESTINATION_VERIFIED.json', checked)
    fields = ['JobIDRaw', 'State', 'ExitCode', 'ElapsedRaw', 'AllocNodes', 'Start', 'End', 'Partition', 'NodeList']
    deadline = time.monotonic() + 120
    live = {'RUNNING', 'PENDING', 'COMPLETING', 'CONFIGURING', 'SUSPENDED'}
    while True:
        rows = [line.split('|') for line in remote(['sacct', '-X', '-j',
            ','.join(r['id'] for r in registry), '-n', '-P', '-o', ','.join(fields)]).decode().splitlines() if line.strip()]
        assert {r[0] for r in rows} == {r['id'] for r in registry}
        if all(r[1].split()[0].rstrip('+') not in live for r in rows):
            break
        assert time.monotonic() < deadline, 'Terminal observation timeout; poll the existing job, do not resubmit'
        time.sleep(2)
    assert next(r for r in rows if r[0] == jid)[1:3] == ['COMPLETED', '0:0']
    total = sum(int(r[3] or 0) * int(r[4] or 0) / 3600 for r in rows)
    assert total + 3 < 40, 'Include the preempted Perseus job full requested upper bound'
    write(E / 'evidence/AMD_FINAL_ACCOUNTING.json', dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), terminal=True,
        allocated_node_hours=total, Slurm_timestamp_timezone='UTC', fields=fields,
        scheduler_rows=rows, scope='All registered AMD extension jobs including destination verification; Perseus preemption accounted separately'))
    amdbytes = int(remote(['du', '-s', '-B1', X]).split()[0])
    localbytes = int(subprocess.check_output(['du', '-s', '-B1', str(E)]).split()[0])
    assert amdbytes + localbytes < 200 * 1024 ** 3
    files = [dict(path=str(destination / name), source=X + '/review/' + name, **row)
             for name, row in sorted(checked['files'].items())]
    write(E / 'evidence/MIRROR_VERIFIED.json', dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), job=jid,
        verification_host=checked['host'], verification_step=checked['step'],
        source_root=X + '/review', mirror_root=str(destination), one_way=True,
        files=files, count=len(files), bytes=checked['bytes'],
        verification_method='Actual destination files streamed read-only to owned allocated AMD Slurm step for SHA256; no media hashing/reduction/rendering on Perseus login',
        AMD_extension_allocated_bytes=amdbytes, Perseus_extension_allocated_bytes=localbytes,
        combined_allocated_bytes=amdbytes + localbytes, Mac_delivery=False,
        actual_visual_review='pending final joint acceptance'))
    print(json.dumps(dict(files=len(files), bytes=checked['bytes'], AMD_node_hours=total,
        combined_storage_GiB=(amdbytes + localbytes) / 1024 ** 3)))


if __name__ == '__main__':
    main()
