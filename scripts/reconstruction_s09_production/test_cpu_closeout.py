"""Allocated integration check against real final fields and an unclosed case."""
from common import *
from closeout_cpu import ready
import extension_resources as resources
import shutil

job()
c = case_config('yc0900_u0005')
state, fields = ready(EXT / c['key'], c)
assert state['complete'] and fields['no_evolution']
other = case_config('yc0819_u0005')
try:
    ready(EXT / other['key'], other)
except AssertionError:
    pass
else:
    raise AssertionError('Unfinished0.819 endpoint was admitted for CPU closeout')
actual = resources.cpu_watchdog(once=True)
assert actual['whole'] < 1.7*TIB and 'storage_error' not in actual
original = resources.storage
def transient_failure():
    raise RuntimeError('Intentional transient inventory test')
try:
    resources.storage = transient_failure
    failure = resources.cpu_watchdog(once=True)
    assert 'Intentional transient inventory test' in failure['storage_error']
finally:
    resources.storage = original
recovered = resources.cpu_watchdog(once=True)
assert 'storage_error' not in recovered
assert not (ROOT / 'evidence/RESOURCE_STOP.json').exists()
marker = EXT / 'evidence/CPU_CLOSEOUT_TEST_OK.json'
if marker.exists():
    destination = ROOT / 'evidence/jobs' / job() / 'PREVIOUS_CPU_CLOSEOUT_TEST_OK.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(marker, destination)
atomic(EXT / 'evidence/CPU_CLOSEOUT_TEST_OK.json', dict(utc=now(), job=job(),
       real_closed_endpoint_and_full_field_hashes_verified=True,
       actual_unfinished_endpoint_rejected=True,
       real_whole_user_storage_sample_verified=True,
       transient_inventory_failure_and_recovery_verified=True,
       source=receipt(Path(__file__).with_name('closeout_cpu.py')),
       storage_guard_source=receipt(Path(resources.__file__)),
       reference_checkpoint=state['latest']['sha256'],
       scope='Integration/read-only validation under Slurm. No native call, reduction rewrite or scientific acceptance for0.819.'))
