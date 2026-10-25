"""Small timing arithmetic and fail-closed applicability checks, under Slurm."""
import copy
from closeout_reserve import seconds, estimate
from common import *


def main():
    job()
    assert seconds(0) == 300 and seconds(314) == 1122
    assert seconds(314, 400, 314) == 720
    assert seconds(300, 400, 314) == 720
    assert seconds(628, 400, 314) == 1320
    for invalid in [float('nan'), float('inf'), 0., -1.]:
        try:
            seconds(314, invalid, 314)
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid measured time accepted')
    c = case_config('yc0819_u0005')
    work = EXT / c['key'] / 'runs' / c['id']
    state = load(work / 'state.json')
    assert not state.get('complete'), 'This forecast test must precede the last evolution'
    fallback = estimate(c, work, state, 8)
    assert fallback['seconds'] == seconds(fallback['full_particle_snapshots'])
    future = copy.deepcopy(state)
    future['complete'] = True
    future['latest']['time'] = 300.
    measured = estimate(c, work, future, 8)
    assert 'reference' in measured, measured
    assert 700 <= measured['seconds'] < 900
    assert measured['measured_closeout_seconds'] > 400
    assert estimate(c, work, future, 4)['seconds'] == fallback['seconds']
    for field, value in [('nblocks', 400), ('particle_count', 1999999),
                         ('ah_onset', 299.), ('removal_counts', [0, 0, 1, 0])]:
        bad = copy.deepcopy(future)
        bad['latest'][field] = value
        assert 'reference' not in estimate(c, work, bad, 8), field
    result = dict(utc=now(), job=job(), arithmetic_and_rejection_checks_passed=True,
                  actual_incomplete_state_falls_back=True, matching_reference_forecast=measured,
                  scope='Timing forecast only using a synthetic300M state. This does not establish a300M endpoint or numerical acceptance, and runs no native evolution.')
    atomic(EXT / 'evidence/CLOSEOUT_TIMING_TEST_OK.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
