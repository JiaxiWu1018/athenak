"""One serial production allocation, with explicit pilot and milestone gates."""
import shutil
import time
from common import *
from checkpoint import validate
from setup_case import setup
from restart_identity import gate
from run_segment import run
from extension_resources import admission

def advance(c, work, target):
    from recover import recover
    state = recover(c, work)
    start = float(os.environ['ST8X_START_EPOCH'])
    limit = int(os.environ['ST8X_WALL_SECONDS'])
    while not state.get('complete') and state['latest']['time'] < target-1e-8:
        remaining = limit-(time.time()-start)-240
        if remaining < 180:
            break
        state = run(c, target=target, wall_seconds=int(min(1200, remaining)), suppress_segment_3d=True)
        if state['segments'][-1].get('stop_reason') not in [None, 'ah_tail_complete']:
            break
        assert not (ROOT/'evidence/RESOURCE_STOP.json').exists()
    return state

def control_gate():
    """The short U0 check is separate from the historical production control."""
    if (ROOT/'evidence/SHORT_CONTROL_OK.json').exists():
        return load(ROOT/'runs/yc0500/state.json')
    from fresh_initial import initialize
    initialize(CASE, control=True)
    work = ROOT/'runs/yc0500'
    if not (ROOT/'evidence/SHORT_CONTROL_OK.json').exists():
        if not (ROOT/'evidence/PRODUCTION_ADMISSION.json').exists():
            gate(load(work/'state.json')['latest']['path'])
        c = dict(CASE, U=0., target=20.)
        state = advance(c, work, 20)
        if state['latest']['time'] >= 20-1e-8:
            atomic(ROOT/'evidence/SHORT_CONTROL_OK.json', dict(utc=now(), job=job(), checkpoint=state['latest'],
                scope='Actual20M matched U0 numerical check; not a third long production control'))
    return load(work/'state.json')

def import_control():
    work = setup(CASE)
    if (work/'state.json').exists():
        return work
    source = DONOR/'runs/yc0500/rst/yc0500.00058.rst'
    old = validate(source)
    assert old['sha256'] == ANCHOR_SHA and old['time'] == 700
    assert sha(ROOT/'protected/athena') == OLD_EXE_SHA
    destination = work/'rst/yc0500.00058.rst'
    destination.parent.mkdir(exist_ok=True)
    assert not destination.exists()
    shutil.copy2(source, destination)
    copied = validate(destination)
    assert copied['sha256'] == ANCHOR_SHA
    copied.pop('parameters')
    atomic(work/'checkpoints.json', [copied])
    atomic(work/'state.json', dict(segments=[], latest=copied, complete=False, inherited_time=700))
    atomic(work/'INITIAL_OK.json', dict(utc=now(), job=job(), inherited=True,
        accepted_original_endpoint=ANCHOR_SHA, original_t0_not_regenerated=True,
        count=2000000, positions_ids_weights_exact=True))
    atomic(ROOT/'evidence/INHERITED_700M.json', dict(utc=now(), job=job(), source=str(source),
        destination=copied, original_preserved=True, executable=receipt(ROOT/'protected/athena')))
    return work

def main():
    job()
    admission()
    from artifact_provenance import audit
    audit()
    mode = os.environ.get('S10_MODE', 'production')
    if mode == 'short_control':
        state = control_gate()
    else:
        if CASE.get('inherited_time'):
            work = import_control()
        else:
            from fresh_initial import admit
            work = setup(CASE)
            admit(CASE, work)
            initial = load(ROOT/'evidence/INITIAL_CONSTRAINT_ADMISSION.json')
            short = Path(initial['control_root'])/'evidence/SHORT_CONTROL_OK.json'
            if not short.exists():
                env = os.environ.copy()
                env.update(ROOT=initial['control_root'], ST8X_U='0', S10_MODE='short_control')
                cmd(['python3', Path(__file__).resolve()], env=env)
                if not short.exists():
                    atomic(ROOT/'evidence/SHORT_CONTROL_PENDING.json', dict(utc=now(), job=job()))
                    return
        prior = ROOT/'evidence/PRODUCTION_ADMISSION.json'
        old = load(prior) if prior.exists() else {}
        if not (old.get('zero_steps') and old.get('ranks') == 8 and old.get('executable_sha256') == sha(ROOT/'protected/athena')):
            record = gate(load(work/'state.json')['latest']['path'])
            record.update(executable_sha256=sha(ROOT/'protected/athena'), machine=MACHINE)
            atomic(prior, record)
        assert not (ROOT/'evidence/RESOURCE_STOP.json').exists()
        target = float(os.environ['S10_TARGET'])
        required = [1000] if CASE.get('inherited_time') else [20, 100, 500, 1000]
        for milestone in required:
            if target > milestone:
                review = load(ROOT/'evidence'/f'MILESTONE_{milestone:05d}_NUMERICAL.json')
                assert review['numerical_passed'] and abs(review['endpoint']-milestone)<1e-7
        state = advance(CASE, work, target)
        if state['latest']['time'] >= 20-1e-8 and not CASE.get('inherited_time'):
            atomic(ROOT/'evidence/OUTWARD_20M_OK.json', dict(utc=now(), job=job(),
                checkpoint=state['latest'], same_trajectory=True,
                initial_constraint_admission=load(ROOT/'evidence/INITIAL_CONSTRAINT_ADMISSION.json')))
        if state.get('complete'):
            from final_fields import export
            export(CASE['id'])
    atomic(ROOT/'evidence/jobs'/job()/'CASE_RESULT.json', dict(utc=now(), job=job(), case=CASE,
        latest=state['latest'], complete=state.get('complete', False), mode=mode))

if __name__ == '__main__':
    main()
