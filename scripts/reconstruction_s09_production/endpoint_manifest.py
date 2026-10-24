"""Close an AMD endpoint inventory inside an existing continuation allocation."""
import re
import shutil
import socket
import sys
import time
from common import *


def finish(c):
    job()
    if MACHINE != 'amd':
        return
    work = ROOT / 'runs' / c['id']
    state = load(work / 'state.json')
    accept = load(ROOT / 'evidence/CASE_NUMERICAL_ACCEPTANCE.json')
    packet = load(ROOT / 'review_packet/MANIFEST.json')
    final = load(work / 'FINAL_FIELDS_OK.json')
    expected = state['latest']['sha256']
    assert state['complete'] and accept['numerical_checks_passed'] and final['no_evolution']
    assert expected == accept['checkpoint']['sha256'] == packet['checkpoint']['sha256'] == final['checkpoint']['sha256']
    record = ROOT / 'evidence/jobs' / job()
    deployment_file=record/'DEPLOYMENT.json'
    deployment=load(deployment_file if deployment_file.exists() else EXT/'evidence/ENDPOINT_MANIFEST_DEPLOYMENT.json')
    assert re.fullmatch('[0-9a-f]{40}', deployment['source_commit'])
    for name, digest in deployment['python_files'].items():
        assert Path(name).name == name
        assert sha(Path(__file__).parent / name) == digest, name
    atomic(record / 'ENDPOINT_MANIFEST_SOURCE.json', deployment)
    marker = ROOT / 'evidence/CLOSED_OUTPUT_MANIFEST.json'
    reused = marker.exists()
    if not reused:
        remaining = int(os.environ['ST8X_WALL_SECONDS']) - (time.time() - float(os.environ['ST8X_START_EPOCH']))
        if remaining < 180:
            atomic(record / 'ENDPOINT_MANIFEST_DEFERRED.json', dict(utc=now(), job=job(), checkpoint=state['latest'], remaining_seconds=remaining,
                reason='Retain the completed scientific endpoint; close its raw inventory in a later allocated continuation.', no_native_evolution_here=True))
            return
        cmd([sys.executable, Path(__file__).with_name('closed_output_manifest.py'), '--root', ROOT, '--source-commit', deployment['source_commit']])
    inventory = load(marker)
    assert inventory['stable_files_verified'] and inventory['checkpoint']['sha256'] == expected
    sizes = {}
    for name, path in [('case', ROOT), ('new_batch', EXT), ('whole_AMD_user', Path('/work1/eliasmost/jiaxiwu'))]:
        sizes[name] = int(subprocess.check_output(['du', '-s', '-B1', str(path)], text=True, timeout=180).split()[0])
    result = dict(utc=now(), job=job(), host=socket.gethostname(), case=c, checkpoint=state['latest'], closed_output_manifest=receipt(marker),
        reused_existing_inventory=reused, file_count=inventory['file_count'], logical_selected_bytes=inventory['logical_bytes'],
        allocated_bytes=sizes, filesystem_free_bytes=shutil.disk_usage(EXT).free, retention='Retained on AMD; no data deletion by this postprocessor.',
        postprocessor_source=deployment, no_native_evolution_here=True,
        scope='Closed source inventory and current allocated-byte metadata. A reused inventory retains its original fixed-file scope; this worker adds its own separate provenance receipt. Numerical and actual visual acceptance are separate.')
    atomic(record / 'ENDPOINT_MANIFEST_OK.json', result)
    atomic(ROOT / 'evidence/ENDPOINT_RETENTION.json', result)
