"""Allocated independent arithmetic/native checks after actual new reductions.

Do not call the production shell/window/comparison reducers. Check exported
results without introducing scientific stability thresholds.
"""
import bisect
import csv
import math
import numpy as np
from cfg import *
import reduce_run as native

NAMES = ['ordinary_1p5m_s1'] + [c['id'] for c in CASES]
MAX_ERRORS = {}


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def dest(name):
    return BASE/'reduced' if name == NAMES[0] else ROOT/'reduced'/name


def close(label, actual, expected, atol=2e-12, rtol=2e-11):
    actual, expected = float(actual), float(expected)
    assert math.isfinite(actual) and math.isfinite(expected), label
    error = abs(actual-expected)
    MAX_ERRORS[label] = max(MAX_ERRORS.get(label, 0.), error)
    assert error <= atol+rtol*abs(expected), (label, actual, expected, error)


def interp(t, times, values):
    """Scalar interpolation independent of the reducers' NumPy calls."""
    if t == times[-1]:
        return values[-1]
    j = bisect.bisect_right(times, t)-1
    assert 0 <= j < len(times)-1
    w = (t-times[j])/(times[j+1]-times[j])
    return values[j]+w*(values[j+1]-values[j])


def integral(t, values):
    return math.fsum((b-a)*(x+y)/2 for a, b, x, y in
                     zip(t[:-1], t[1:], values[:-1], values[1:]))


def check_windows(name, shells):
    times = [float(r['time']) for r in shells]
    rows = read(dest(name)/'windows.csv')
    expected_keys = {(family, j*width, (j+1)*width, q)
                     for family, width in [('100M', 100.), ('P', P), ('2P', 2*P)]
                     for j in range(int(1000/width)) for q in range(10, 100, 10)}
    keys = {(r['family'], float(r['start']), float(r['end']), int(r['quantile'])) for r in rows}
    assert keys == expected_keys and len(keys) == len(rows) == 153
    for row in rows:
        lo, hi, q = float(row['start']), float(row['end']), int(row['quantile'])
        t = [lo]+[v for v in times if lo < v < hi]+[hi]
        source = [float(r[f'r{q}']) for r in shells]
        v = [interp(tt, times, source) for tt in t]
        initial = source[0]
        mean = integral(t, v)/(hi-lo)
        weights = [(t[1]-t[0])/2]+[(t[j+1]-t[j-1])/2 for j in range(1, len(t)-1)]+[(t[-1]-t[-2])/2]
        centered = [tt-(lo+hi)/2 for tt in t]
        slope = math.fsum(w*x*(y-mean) for w, x, y in zip(weights, centered, v))/math.fsum(w*x*x for w, x in zip(weights, centered))
        residual = [y-mean-x*slope for x, y in zip(centered, v)]
        values = dict(half_range_fraction=(max(v)-min(v))/(2*initial),
                      mean_shift_fraction=(mean-initial)/initial,
                      detrended_half_range_fraction=(max(residual)-min(residual))/(2*initial),
                      slope_fraction_per_M=slope/initial)
        for key, value in values.items():
            close('window_'+key, row[key], value)
    return len(rows)


def check_native(case, shells):
    name, n = case['id'], case['n']
    work = ROOT/'runs'/name/'production'
    sources = load(dest(name)/'summary.json')['source_particle_outputs']
    native_paths = sorted((work/'out/pvtk').glob('*.part.vtk'))
    assert len(native_paths) == 501
    cache = ROOT/'movie_cache'/name
    with np.load(cache/'trajectories.npz', allow_pickle=False) as data:
        display = data['ids'].copy()
        clouds = data['positions'][[0, 250, 500]].copy()
        initial_radius = data['initial_radius'].copy()
        cache_times = data['times'].copy()
    assert len(np.unique(display)) == len(display) == 20000
    if case['paired']:
        parents = display[display < n//2]
        assert len(parents) == 10000 and np.array_equal(display[display >= n//2], parents+n//2)
    _, initial = sample(ROOT/'samples'/(name+'.bin'))
    radius = np.sqrt(np.sum(initial[display, 1:4]**2, axis=1))
    assert np.allclose(initial_radius, radius, rtol=0, atol=2e-14)
    del initial
    with np.load(dest(name)/'profiles.npz') as profiles:
        core_mass = profiles['mass_bins'][[0, 250, 500], 0].copy()
    pair_rows = read(dest(name)/'pair_symmetry.csv') if case['paired'] else None
    checks = []
    for index, slot in enumerate([0, 250, 500]):
        path = native_paths[slot]
        bound = sources[slot]
        assert sha(path) == bound['sha256']
        t, cycle, count, data = native.read_pvtk(str(path))
        assert count == n and t == float(shells[slot]['time']) == cache_times[slot]
        tags = data['ptag'].astype(np.int64)
        assert np.array_equal(tags, data['ptag']) and np.all((tags >= 0) & (tags < n))
        assert np.all(np.bincount(tags, minlength=n) == 1)
        assert all(np.isfinite(v).all() for v in data.values())
        mass = data['prtcl_mass']
        assert np.all(mass == mass[0]) and mass[0] > 0
        pos = np.column_stack([data[c] for c in 'xyz'])
        # Equal native particle weights permit an independently computed mean.
        center = pos.mean(axis=0)
        for component, value in zip('xyz', center):
            close('native_COM', shells[slot]['com_'+component], value)
        close('native_COM_magnitude', shells[slot]['com_mag'], np.sqrt(np.dot(center, center)))
        radius_sets = [np.sqrt(np.sum((pos-center)**2, axis=1)), np.sqrt(np.sum(pos**2, axis=1))]
        brackets = {}
        # Partial selection is independent of the production sorted mass CDF.
        # Adjacent ranks allow its accumulated finite-precision CDF rounding.
        ranks = sorted({int(n*q/100)+offset for q in range(10, 100, 10) for offset in range(-2, 3)})
        for suffix, radii in zip(['', '_origin'], radius_sets):
            selected = np.partition(radii, ranks)
            for q in range(10, 100, 10):
                j = int(n*q/100)
                neighbors = selected[j-2:j+3]
                lo, hi = float(neighbors.min()), float(neighbors.max())
                value = float(shells[slot][f'r{q}'+suffix])
                assert lo-1e-10 <= value <= hi+1e-10
                brackets[f'r{q}'+suffix] = [lo, hi]
        close('native_central_mass', core_mass[index], np.count_nonzero(radius_sets[0] < .5)*mass[0])
        inverse = np.empty(n, dtype=np.int32)
        inverse[tags] = np.arange(n, dtype=np.int32)
        assert np.array_equal(clouds[index], pos[inverse[display]].astype(np.float32))
        if pair_rows is not None:
            first, second = inverse[:n//2], inverse[n//2:]
            defect = pos[first]+pos[second]
            lengths2 = np.sum(defect**2, axis=1)
            close('pair_position_RMS', pair_rows[slot]['position_pair_sum_rms_M'], np.sqrt(lengths2.mean()))
            close('pair_position_max', pair_rows[slot]['position_pair_sum_max_M'], np.sqrt(lengths2.max()))
            u = np.column_stack([data['u'+c] for c in 'xyz'])
            close('pair_momentum_RMS', pair_rows[slot]['covariant_momentum_pair_sum_rms'], np.sqrt(np.mean(np.sum((u[first]+u[second])**2, axis=1))))
        checks.append(dict(case=name, slot=slot, time_M=t, cycle=cycle, n=n,
                           path=str(path), sha256=bound['sha256'], independent_rank_brackets=brackets,
                           exact_display_positions=True, central_bin_mass_verified=True))
        print('AUDIT_NATIVE', name, slot, t, flush=True)
    return checks


def check_differences(data):
    rows = read(ROOT/'comparison/shell_differences.csv')
    with np.load(ROOT/'samples/model.npz') as model:
        proper, radius = model['proper'].tolist(), model['riso_over_m'].tolist()
    verified = {}
    for row in rows:
        first, second, q = row['first'], row['second'], int(row['quantile'])
        lo, hi = float(row['start']), float(row['end'])
        a, b = data[first], data[second]
        ta, tb = [float(v['time']) for v in a], [float(v['time']) for v in b]
        va, vb = [float(v[f'r{q}']) for v in a], [float(v[f'r{q}']) for v in b]
        t = sorted({lo, hi} | {v for v in ta+tb if lo < v < hi})
        rref = interp(q/100, proper, radius)
        close('difference_equilibrium_radius', row['common_equilibrium_radius_M'], rref)
        delta = [(interp(tt, tb, vb)-interp(tt, ta, va))/rref for tt in t]
        rms = math.sqrt(integral(t, [v*v for v in delta])/(hi-lo))
        close('difference_RMS', row['RMS_difference_fraction'], rms)
        close('difference_mean', row['signed_mean_difference_fraction'], integral(t, delta)/(hi-lo))
        close('difference_max', row['max_abs_difference_fraction'], max(abs(v) for v in delta))
        key = (first, second, q, lo, hi, row['family'])
        assert key not in verified
        verified[key] = rms
    expected_windows = {(0., 1000., 'full'), (P, 2*P, 'early_P'), (4*P, 5*P, 'late_P')} | {(100.*j, 100.*(j+1), '100M') for j in range(10)}
    pairs = [(NAMES[0], NAMES[2]), (NAMES[2], NAMES[3]), (NAMES[1], NAMES[2])]
    assert set(verified) == {(a, b, q, lo, hi, family) for a, b in pairs for q in range(10, 100, 10) for lo, hi, family in expected_windows}
    ratios = read(ROOT/'comparison/N_difference_ratios.csv')
    seen = set()
    for row in ratios:
        q, lo, hi, family = int(row['quantile']), float(row['start']), float(row['end']), row['family']
        coarse = verified[(*pairs[0], q, lo, hi, family)]
        fine = verified[(*pairs[1], q, lo, hi, family)]
        close('ratio_coarse_RMS', row['coarse_RMS_fraction'], coarse)
        close('ratio_fine_RMS', row['fine_RMS_fraction'], fine)
        if coarse > 0:
            close('N_difference_ratio', row['fine_over_coarse_RMS'], fine/coarse)
        else:
            assert row['fine_over_coarse_RMS'] == ''
        key = (q, lo, hi, family)
        assert key not in seen
        seen.add(key)
    assert seen == {(q, lo, hi, family) for q in range(10, 100, 10) for lo, hi, family in expected_windows}
    return dict(difference_rows=len(rows), ratio_rows=len(ratios))


def main():
    require_job()
    target = ROOT/'evidence/RESULTS_ARITHMETIC_QA.json'
    assert not target.exists(), 'Preserve the previous audit receipt before a documented new attempt'
    data = {name: read(dest(name)/'shells.csv') for name in NAMES}
    windows = {name: check_windows(name, data[name]) for name in NAMES}
    checks = [check for case in CASES for check in check_native(case, data[case['id']])]
    differences = check_differences(data)
    inputs = [dest(name)/file for name in NAMES for file in ['shells.csv', 'windows.csv']]
    inputs += [ROOT/'comparison'/file for file in ['shell_differences.csv', 'N_difference_ratios.csv']]
    atomic(target, dict(utc=now(), job=os.environ['SLURM_JOB_ID'], source=source(), passed=True,
                       scope='Independent scalar window/N-difference arithmetic and representative full-N native checks; no scientific pass threshold or actual visual review claimed.',
                       window_rows=windows, native_checks=checks, comparison_checks=differences,
                       maximum_absolute_arithmetic_errors=MAX_ERRORS,
                       inputs=[dict(path=str(p), sha256=sha(p)) for p in inputs],
                       visual_review_still_required=True))
    print('RESULTS_ARITHMETIC_QA_OK', flush=True)


if __name__ == '__main__':
    main()
