"""Verify the selected thin products on an allocated Perseus compute node."""
import argparse
import datetime
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Allocated compute job required'
    root = args.destination.resolve()
    manifest_path = root / 'evidence/REVIEW_PRODUCT_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    acceptance = json.loads((root / 'evidence/FINAL_ACCEPTANCE.json').read_text())
    assert acceptance['product_manifest']['sha256'] == digest(manifest_path)
    assert acceptance['all_five_numerical_acceptance']
    assert acceptance['actual_PDF_and_encoded_frame_inspection']
    assert acceptance['resource_limits_passed']
    rows = []
    for record in manifest['files']:
        relative = Path(record['relative'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path = root / relative
        assert path.resolve().is_relative_to(root) and path.is_file()
        assert path.stat().st_size == record['bytes']
        actual = digest(path)
        assert actual == record['sha256'], str(relative)
        rows.append(dict(relative=str(relative), bytes=record['bytes'], sha256=actual))
    assert len(rows) == manifest['file_count']
    assert sum(row['bytes'] for row in rows) == manifest['logical_bytes']
    output = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  host=socket.gethostname(), job=os.environ['SLURM_JOB_ID'],
                  destination=str(root), source_manifest_sha256=digest(manifest_path),
                  file_count=len(rows), bytes=sum(row['bytes'] for row in rows),
                  destination_SHA256_verified=True, files=rows,
                  allocated_destination_bytes=int(subprocess.check_output(
                      ['du', '-s', '-B1', str(root)], text=True).split()[0]),
                  publisher_terminal_addendum_required=True,
                  scope='Actual destination bytes checked against the fixed AMD product manifest. No raw data/source/frames mirrored; no Mac delivery claimed.')
    path = root / 'evidence/THIN_REVIEW_VERIFIED.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(output, indent=2) + '\n')
    temporary.replace(path)
    print(json.dumps(dict(file_count=len(rows), bytes=output['bytes'], verified=True)))


if __name__ == '__main__':
    main()
