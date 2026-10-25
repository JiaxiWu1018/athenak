"""Audit a saved endpoint on the serial CPU lane after native GPU field export."""
from common import *


def ready(root, c):
    job()
    work = root / 'runs' / c['id']
    state = load(work / 'state.json')
    last = state['latest']
    assert state['complete'] and last['payload_verified']
    expected = min(c['target'], last['ah_onset'] + 20) if last['ah_onset'] >= 0 else c['target']
    assert abs(last['time'] - expected) < 1e-7
    fields = load(work / 'FINAL_FIELDS_OK.json')
    assert fields['no_evolution'] and fields['checkpoint']['sha256'] == last['sha256']
    assert fields['executable']['sha256'] == state['segments'][-1]['executable']['sha256']
    assert len(fields['files']) == 2
    for record in fields['files']:
        assert record['full_payload_finite'] and record['nblocks'] == last['nblocks']
        assert record['time'] == last['time'] and record['cycle'] == last['cycle']
        path = Path(record['path'])
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256']
    return state, fields


def main(key):
    job()
    c = case_config(key)
    assert ROOT == EXT / key and CASE['key'] == key
    assert not (ROOT / 'evidence/RESOURCE_STOP.json').exists()
    state, fields = ready(ROOT, c)
    record = ROOT / 'evidence/jobs' / job()
    assert load(record / 'DEPLOYMENT.json')['native_source_unchanged']
    packet = ROOT / 'review_packet/MANIFEST.json'
    if packet.exists():
        from products import verify
        verify(packet.parent)
    else:
        from reduce import reduce_case
        from case_review import review
        from review_packet import export
        reduce_case(c)
        assert not (ROOT / 'evidence/RESOURCE_STOP.json').exists()
        review(c)
        assert not (ROOT / 'evidence/RESOURCE_STOP.json').exists()
        export(c)
    assert not (ROOT / 'evidence/RESOURCE_STOP.json').exists()
    from endpoint_manifest import finish
    finish(c)
    closed = load(ROOT / 'evidence/CLOSED_OUTPUT_MANIFEST.json')
    assert closed['stable_files_verified'] and closed['checkpoint']['sha256'] == state['latest']['sha256']
    atomic(record / 'CPU_CLOSEOUT_OK.json', dict(utc=now(), job=job(), case=c,
           checkpoint=state['latest'], native_fields=receipt(ROOT / 'runs' / c['id'] / 'FINAL_FIELDS_OK.json'),
           numerical_acceptance=receipt(ROOT / 'evidence/CASE_NUMERICAL_ACCEPTANCE.json'),
           packet=receipt(packet), raw_manifest=receipt(ROOT / 'evidence/CLOSED_OUTPUT_MANIFEST.json'),
           no_native_evolution=True, all_existing_numerical_checks_retained=True))
