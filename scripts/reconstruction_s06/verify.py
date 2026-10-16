"""Allocated independent arithmetic checks and reduction-interface regression."""
import argparse
from session import *
import numpy as np

def arithmetic():
    from analyze_extension import stitch, window_metrics
    # Irregular sampling, interpolated boundaries and a known linear radius drift.
    t = np.array([0., .3, 1.7, 2.8, 4.])
    initial = np.arange(1., 10.)
    values = initial[None, :] * (1 + .02 * t[:, None])
    rows = window_metrics(t, values, initial, [(.5, 3.5)], 'synthetic')
    for row in rows:
        assert abs(row['half_range_fraction'] - .03) < 1e-12
        assert abs(row['mean_shift_fraction'] - .04) < 1e-12
        assert abs(row['slope_fraction_per_M'] - .02) < 1e-12
        assert row['detrended_half_range_fraction'] < 1e-12
    prefix = [{'time': str(i)} for i in range(0, 301, 2)]
    assert len(stitch(prefix, [{'time': '302'}, {'time': '304'}])) == 153
    for bad in ([{'time': '300'}], [{'time': '304'}], [{'time': '302'}, {'time': '302'}]):
        try:
            stitch(prefix, bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing/overlap/duplicate case was accepted')
    # Historical first-three window values must reproduce without changing q0.
    old = ROOT / 'baseline' / CASE
    shells = read_csv(old / 'shells.csv')
    times = np.array([float(r['time']) for r in shells])
    radii = np.array([[float(r[f'r{q}']) for q in range(10, 100, 10)] for r in shells])
    actual = window_metrics(times, radii, radii[0], [(0, 100), (100, 200), (200, 300)], '100M')
    expected = read_csv(old / 'windows.csv')
    error = max(abs(a[key] - float(b[key])) for a, b in zip(actual, expected)
                for key in ('half_range_fraction', 'mean_shift_fraction', 'detrended_half_range_fraction', 'slope_fraction_per_M'))
    assert len(actual) == len(expected) and error < 1e-12
    return {'synthetic_linear_drift': 'passed', 'stitch_missing_duplicate_overlap': 'passed', 'prefix_window_max_abs_error': error}

def final():
    import reduce_run as R
    import analyze as A
    dest = ROOT / 'reduced'
    rows = read_csv(dest / 'shells.csv')
    assert len(rows) == 501 and float(rows[-1]['time']) == END
    assert np.array_equal(np.rint(np.array([float(r['time']) for r in rows]) / 2), np.arange(501))
    _, initial = sample(ROOT / 'samples' / (CASE + '.bin'))
    maximum = 0.
    # Reconstruct selected quantiles directly from sorted particle rest mass.
    for nominal in (302, 500, 1000):
        path = ROOT / 'runs/production/out/pvtk' / f'{CASE}.prtcl_all.{nominal//2:05d}.part.vtk'
        t, cycle, n, data = R.read_pvtk(str(path))
        tags = data['ptag'].astype(np.int64)
        mass = data['prtcl_mass']
        assert n == N and np.array_equal(mass, initial[tags, 7].astype(np.float32).astype(float))
        pos = np.stack([data[a] for a in 'xyz'], axis=1)
        com = np.sum(mass[:, None] * pos, axis=0) / mass.sum()
        radius = np.linalg.norm(pos - com, axis=1)
        order = np.argsort(radius)
        cumulative = np.cumsum(mass[order])
        saved = rows[nominal // 2]
        for q in (10, 50, 90):
            value = radius[order[np.searchsorted(cumulative, q / 100 * cumulative[-1])]]
            error = abs(value - float(saved[f'r{q}']))
            assert error < 1e-10
            maximum = max(maximum, error)
    # Direct time-weighted arithmetic, including a non-grid-aligned P window.
    times = np.array([float(r['time']) for r in rows])
    windows = read_csv(dest / 'windows.csv')
    max_window_error = 0.
    for w in windows:
        q = int(w['quantile']); lo, hi = float(w['start']), float(w['end'])
        rr = np.array([float(r[f'r{q}']) for r in rows]); r0 = rr[0]
        tt = np.r_[lo, times[(times > lo) & (times < hi)], hi]
        vv = np.interp(tt, times, rr)
        area = sum((vv[i] + vv[i+1]) * .5 * (tt[i+1] - tt[i]) for i in range(len(tt)-1))
        mean = area / (hi - lo)
        for got, expected in ((.5 * (max(vv) - min(vv)) / r0, float(w['half_range_fraction'])),
                              ((mean - r0) / r0, float(w['mean_shift_fraction']))):
            error = abs(got - expected)
            assert error < 1e-12
            max_window_error = max(max_window_error, error)
    health = read_csv(dest / 'health.csv')
    native = np.atleast_2d(np.loadtxt(ROOT / 'runs/production/out' / (CASE + '.user.hst')))
    for row, v in zip(health[151:], native):
        assert abs(float(row['H_rms_r10']) - np.sqrt(max(v[3], 0) / v[2])) < 1e-14
        assert abs(float(row['M_rms_r10']) - np.sqrt(max(v[4], 0) / v[2])) < 1e-14
        assert (str(row['constraint_usable']).lower() == 'true') == (v[18] > 1e-6)
    # Genuine interior 500M constraint cube vs the history reduction.
    path = ROOT / 'runs/production/out/bin' / f'{CASE}.con3d.00005.bin'
    t, cube = A.cube(path)
    from reference import coordinates
    rr = np.linalg.norm(coordinates(), axis=1)
    hrow = health[250]
    assert abs(float(hrow['time']) - t) < 1e-7
    for rad in (10, 16):
        keep = rr < rad
        h = np.sqrt(np.mean(cube['con_H'][keep]**2))
        m = np.sqrt(np.mean(np.maximum(cube['con_M'][keep], 0)))
        assert np.isclose(h, float(hrow[f'H_rms_r{rad}']), rtol=2e-6)
        assert np.isclose(m, float(hrow[f'M_rms_r{rad}']), rtol=2e-6)
    return {'stitched_states': 501, 'independent_shell_max_abs_error': maximum,
            'independent_window_max_abs_error': max_window_error, 'native_history_norms': 'passed',
            'interior_500M_3D_constraint_crosscheck': 'passed', 'endpoint_constraint_usable': health[-1]['constraint_usable']}

def main():
    require_job()
    parser = argparse.ArgumentParser(); parser.add_argument('--final', action='store_true'); args = parser.parse_args()
    result = arithmetic()
    if args.final:
        result.update(final())
    result.update(utc=now(), job=os.environ['SLURM_JOB_ID'])
    atomic(ROOT / 'evidence' / ('QA_FINAL_OK.json' if args.final else 'QA_METHODS_OK.json'), result)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
