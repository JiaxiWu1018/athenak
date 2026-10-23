"""Verify a streamed Perseus review copy inside an allocated AMD Slurm step."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tarfile
import time


X = Path('/work1/eliasmost/jiaxiwu/st_s08_yc0500_extensions_20261023')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def record():
    jid = os.environ['SLURM_JOB_ID']
    assert jid.isdigit()
    return X / 'thin_verify/evidence/jobs' / jid


def main():
    out = record()
    manifest = json.loads((X / 'review/evidence/REVIEW_PRODUCT_MANIFEST.json').read_text())
    files = {}
    for row in manifest['products']:
        relative = Path(row['relative_path']).relative_to('review')
        assert not relative.is_absolute() and '..' not in relative.parts
        files[str(relative)] = dict(bytes=row['bytes'], sha256=row['sha256'])
    evidence = ['REVIEW_PRODUCT_MANIFEST', 'CLOSED_DATA_MANIFEST', 'PROVENANCE',
        'SOURCE_MAP', 'INDEPENDENT_NUMERIC_QA', 'COVERAGE_AND_GLOBAL_CONSTRAINTS',
        'REMOVAL_GEOMETRY', 'PLOTS_RENDERED', 'MOVIES_RENDERED', 'PDF_RASTERS',
        'BOUNDARY_DIAGNOSTICS', 'PRODUCTS_READY', 'ACCOUNTING_OBSERVED', 'STORAGE_REVIEW']
    if (X / 'review/evidence/PRIMARY_MOVIE_PROMOTION.json').exists():
        evidence.append('PRIMARY_MOVIE_PROMOTION')
    for name in evidence:
        path = X / 'review/evidence' / (name + '.json')
        files['evidence/' + path.name] = dict(bytes=path.stat().st_size, sha256=sha(path))
    assert sum(r['bytes'] for r in files.values()) < 1024 ** 3
    write(out / 'READY.json', dict(job=os.environ['SLURM_JOB_ID'], files=files,
        source_root=str(X / 'review'), host=socket.gethostname(),
        verifier_sha256=sha(Path(__file__)), source_review_job=manifest['job']))
    deadline = time.monotonic() + 3000
    while not (out / 'DESTINATION_VERIFIED.json').exists():
        assert not (out / 'FAILED.json').exists(), 'Destination stream verification failed'
        assert not (X / 'thin_verify/evidence/RESOURCE_STOP.json').exists()
        assert time.monotonic() < deadline, 'Verification step was not supplied before deadline'
        time.sleep(2)


def stream():
    out = record()
    ready = json.loads((out / 'READY.json').read_text())
    assert ready['job'] == os.environ['SLURM_JOB_ID']
    expected = ready['files']
    verified = {}
    try:
        with tarfile.open(fileobj=sys.stdin.buffer, mode='r|') as archive:
            for member in archive:
                name = member.name[2:] if member.name.startswith('./') else member.name
                assert member.isfile() and name in expected and name not in verified, name
                assert member.size == expected[name]['bytes'], name
                h = hashlib.sha256()
                with archive.extractfile(member) as f:
                    for block in iter(lambda: f.read(8 * 1024 ** 2), b''):
                        h.update(block)
                assert h.hexdigest() == expected[name]['sha256'], name
                verified[name] = expected[name]
        assert set(verified) == set(expected), 'Incomplete destination stream'
        result = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            job=os.environ['SLURM_JOB_ID'], step=os.environ.get('SLURM_STEP_ID'),
            host=socket.gethostname(), files=verified, count=len(verified),
            bytes=sum(v['bytes'] for v in verified.values()),
            ready_sha256=sha(out / 'READY.json'),
            scope='SHA256 of actual Perseus destination files streamed to an allocated AMD step; no raw scientific arrays or lossless frame sequences transferred')
        write(out / 'DESTINATION_VERIFIED.json', result)
        print(json.dumps(dict(verified=len(verified), bytes=result['bytes'])))
    except Exception as error:
        write(out / 'FAILED.json', dict(error=repr(error)))
        raise


if __name__ == '__main__':
    stream() if '--stream' in sys.argv else main()
