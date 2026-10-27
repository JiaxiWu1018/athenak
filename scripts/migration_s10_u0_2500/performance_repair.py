"""One bounded, same-trajectory admission after a recorded slow-node stop."""
import socket
import time
from common import *


def pending():
    return (EXT/'evidence/PERFORMANCE_REPAIR_PLAN.json').exists() and not (
        ROOT/'evidence/PERFORMANCE_REPAIR_OK.json').exists()


def reviewed_stop():
    if not pending():
        return False
    plan = load(EXT/'evidence/PERFORMANCE_REPAIR_PLAN.json')
    path = ROOT/'evidence/RESOURCE_STOP.json'
    if not path.exists():
        return False
    d = load(path)
    terminal = load(EXT/'evidence/allocations'/plan['stopped_job']/'scheduler_terminal.json')
    return (d['job'] == plan['stopped_job'] and d['reason'] == plan['stop_reason'] and
        'JobState=FAILED' in terminal['scheduler'] and 'ExitCode=1:0' in terminal['scheduler'])


def admit(state, run):
    if not pending():
        return state
    from extension_resources import admission, stop
    from restart_identity import gate
    plan = load(EXT/'evidence/PERFORMANCE_REPAIR_PLAN.json')
    assert plan['kind'] == 'same_trajectory_known_node_admission'
    assert socket.gethostname().split('.')[0] == plan['validated_node']
    assert int(os.environ['SLURM_NTASKS']) == 8
    assert sha(ROOT/'protected/athena') == OLD_EXE_SHA
    assert state['latest']['sha256'] == plan['checkpoint']['sha256']
    assert state['latest']['cycle'] == plan['checkpoint']['cycle']
    terminal = load(EXT/'evidence/allocations'/plan['stopped_job']/'scheduler_terminal.json')
    assert 'JobState=FAILED' in terminal['scheduler'] and 'ExitCode=1:0' in terminal['scheduler']
    assert state['segments'][-1]['returncode'] == 0
    assert state['segments'][-1]['stop_reason'] == plan['stop_reason']
    # Reject a node whose inherited module settings name unavailable IB ports.
    requested = os.environ.get('UCX_NET_DEVICES', 'all')
    if requested != 'all':
        for device in requested.split(','):
            name, port = device.split(':')
            assert (Path('/sys/class/infiniband')/name/'ports'/port).exists(), device
    disk = admission()
    identity = gate(state['latest']['path'])
    identity.update(executable_sha256=OLD_EXE_SHA, machine='amd')
    atomic(ROOT/'evidence/PRODUCTION_ADMISSION.json', identity)
    assert identity['physical_fields_exact'] and identity['zero_steps']
    assert identity['interior_grid_max_scaled_error'] < 1e-12
    # Only the specifically reviewed stop is cleared, with exact bytes retained.
    archived = ROOT/'evidence/performance_repair'/plan['stopped_job']
    archived.mkdir(parents=True, exist_ok=False)
    stopfile = ROOT/'evidence/RESOURCE_STOP.json'
    request = ROOT/'runs/yc0500/REQUEST_STOP'
    d = load(stopfile)
    assert d['job'] == plan['stopped_job'] and d['reason'] == plan['stop_reason']
    assert request.read_text().strip() == plan['stop_reason']
    originals = [receipt(stopfile), receipt(request)]
    atomic(archived/'BEFORE_CLEAR.json', dict(utc=now(), job=job(), files=originals,
        plan=receipt(EXT/'evidence/PERFORMANCE_REPAIR_PLAN.json'), identity=identity,
        admission=disk, action='Archive only the reviewed performance stop before bounded same-trajectory pilot'))
    for path in [stopfile, request]:
        target = archived/path.name
        path.rename(target)
        assert sha(target) == next(x['sha256'] for x in originals if x['path'] == str(path))
    start = time.monotonic()
    initial = state['latest']['time']
    target = min(initial+float(plan['pilot_M']), 2500.)
    state = run(CASE, target=target, wall_seconds=1200, suppress_segment_3d=True)
    elapsed = time.monotonic()-start
    advanced = state['latest']['time']-initial
    rate = elapsed/advanced if advanced > 0 else float('inf')
    remaining = 2500.-state['latest']['time']
    available_wall = int(os.environ['ST8X_WALL_SECONDS'])-(time.time()-float(os.environ['ST8X_START_EPOCH']))-240
    endpoint = state['complete']
    passed = endpoint or (advanced >= float(plan['pilot_M'])-1e-7 and
        rate <= float(plan['maximum_seconds_per_M']) and
        remaining*rate*1.05+120 < available_wall)
    record = dict(utc=now(), job=job(), source_checkpoint=plan['checkpoint'],
        checkpoint=state['latest'], host=socket.gethostname(), identity=identity,
        native_segment=state['segments'][-1], measured_workflow_seconds_per_M=rate,
        pilot_advanced_M=advanced, elapsed_seconds=elapsed,
        projected_remaining_native_hours=remaining*rate/3600,
        available_wall_seconds=available_wall, performance_passed=passed,
        physical_settings_unchanged=True, original_anchors_untouched=True)
    atomic(ROOT/'evidence'/('PERFORMANCE_PILOT_'+job()+'.json'), record)
    if not passed:
        stop('performance_repair_pilot_failed')
        raise RuntimeError('Same-trajectory performance admission failed; no automatic retry')
    assert not (ROOT/'evidence/RESOURCE_STOP.json').exists()
    assert state['segments'][-1].get('stop_reason') in [None, 'qualified_AH_checkpoint', 'ah_tail_complete']
    atomic(ROOT/'evidence/PERFORMANCE_REPAIR_OK.json', record)
    return state
