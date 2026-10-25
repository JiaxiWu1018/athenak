"""Explicit coverage and time-weighted coordinate-profile/shell summaries.

Piecewise-linear interpolation uses native snapshots, with no extrapolation.
Incomplete requested windows retain coverage information but no mean or excursion.
"""
import numpy as np

PROFILE_WINDOWS = ((750., 1000.), (1000., 1250.), (1250., 1500.))


def time_window(times, values, lo, hi):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    assert hi > lo and times.ndim == 1 and len(times) == len(values)
    assert np.isfinite(times).all() and np.isfinite(values).all()
    # Native output counters can restart; filename order is not time order.
    # Verify equal-time payloads before keeping one, rather than silently replacing.
    order = np.argsort(times, kind='stable')
    times, values = times[order], values[order]
    unique = np.r_[True, np.diff(times) > 0] if len(times) else np.array([], dtype=bool)
    for i in np.flatnonzero(~unique):
        assert np.allclose(values[i], values[i-1], rtol=1e-10, atol=1e-12), 'Conflicting duplicate native profiles'
    times, values = times[unique], values[unique]
    result = dict(start=float(lo), end=float(hi), complete=False, samples=0,
        covered_start=None, covered_end=None, covered_duration=0., coverage_fraction=0.,
        max_native_gap=None, mean=None, peak_to_peak=None)
    if len(times) < 2:
        return result
    left, right = max(lo, times[0]), min(hi, times[-1])
    if right <= left:
        return result
    select = (times >= left) & (times <= right)
    # Include interpolation brackets when reporting the actual native gap.
    first = max(0, int(np.searchsorted(times, left, side='right'))-1)
    last = min(len(times)-1, int(np.searchsorted(times, right, side='left')))
    result.update(covered_start=float(left), covered_end=float(right),
        covered_duration=float(right-left), coverage_fraction=float((right-left)/(hi-lo)),
        samples=int(select.sum()), max_native_gap=float(np.diff(times[first:last+1]).max()))
    if times[0] > lo or times[-1] < hi:
        return result
    interior = (times > lo) & (times < hi)
    at = np.r_[lo, times[interior], hi]
    flat = values.reshape(len(times), -1)
    boundary = np.array([[np.interp(t, times, flat[:, k]) for k in range(flat.shape[1])]
                         for t in [lo, hi]])
    y = np.concatenate([boundary[:1], flat[interior], boundary[1:]])
    mean = np.sum((y[1:]+y[:-1])*.5*np.diff(at)[:, None], axis=0)/(hi-lo)
    shape = values.shape[1:]
    result.update(complete=True, mean=mean.reshape(shape),
        peak_to_peak=np.ptp(y, axis=0).reshape(shape))
    return result


def coverage(result):
    return {k: v for k, v in result.items() if k not in ['mean', 'peak_to_peak']}


def profile_windows(times, masses, initial_mass):
    assert initial_mass > 0
    profiles = [time_window(times, masses, lo, hi) for lo, hi in PROFILE_WINDOWS]
    drift = []
    for earlier, later in zip(profiles, profiles[1:]):
        valid = earlier['complete'] and later['complete']
        difference = later['mean']-earlier['mean'] if valid else None
        drift.append(dict(earlier_start=earlier['start'], earlier_end=earlier['end'],
            later_start=later['start'], later_end=later['end'], complete=valid,
            total_variation_M0=(float(.5*np.abs(difference).sum()/initial_mass) if valid else None),
            max_enclosed_mass_change_M0=(float(np.abs(np.cumsum(difference)).max()/initial_mass) if valid else None)))
    return profiles, drift


def analytic_checks():
    # Extra restart snapshots along an unchanged linear signal must not bias means.
    reference = None
    for times in [np.array([0., 1., 2.]), np.array([0., .01, .02, .1, .9, 1., 1.8, 2.])]:
        masses = np.column_stack([1-times/2, times/2])
        first = time_window(times, masses, 0., 1.)
        second = time_window(times, masses, 1., 2.)
        assert np.allclose(first['mean'], [.75, .25])
        assert np.allclose(second['mean'], [.25, .75])
        delta = second['mean']-first['mean']
        assert np.isclose(.5*abs(delta).sum(), .5)
        assert np.isclose(abs(np.cumsum(delta)).max(), .5)
        scalar = time_window(times, 2+times, 0., 2.)
        assert np.isclose(scalar['mean'], 3.) and np.isclose(scalar['peak_to_peak'], 2.)
        if reference is not None:
            assert np.allclose(first['mean'], reference)
        reference = first['mean']
    partial = time_window([1., 2.], [3., 4.], 0., 2.)
    assert not partial['complete'] and partial['mean'] is None and partial['coverage_fraction'] == .5
    missing = time_window([1., 2.], [3., 4.], 3., 4.)
    assert not missing['complete'] and missing['covered_duration'] == 0 and missing['mean'] is None
    unsorted = time_window([2., 0., 1., 1.], [4., 2., 3., 3.], 0., 2.)
    assert unsorted['complete'] and np.isclose(unsorted['mean'], 3.) and unsorted['samples'] == 3
    try:
        time_window([0., 1., 1., 2.], [2., 3., 9., 4.], 0., 2.)
    except AssertionError as e:
        assert 'Conflicting duplicate' in str(e)
    else:
        raise AssertionError('Conflicting duplicate was not rejected')
    t = np.array([750., 1000., 1250., 1500.])
    x = (t-750)/750
    profiles, drift = profile_windows(t, np.column_stack([1-x, x]), 1.)
    assert all(r['complete'] for r in profiles)
    assert all(r['complete'] and np.isclose(r['total_variation_M0'], 1/3)
               and np.isclose(r['max_enclosed_mass_change_M0'], 1/3) for r in drift)
    return dict(linear_profiles=True, irregular_sampling_invariance=True,
        scalar_mean_and_excursion=True, partial_and_missing_coverage=True,
        adjacent_profile_window_drift=True, verified_duplicate_and_unordered_inputs=True)
