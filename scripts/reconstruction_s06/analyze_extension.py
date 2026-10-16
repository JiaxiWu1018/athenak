"""Join preserved session-5 summaries to unchanged-definition continuation reductions."""
import argparse
from session import *
import numpy as np
import analyze as A
import reduce_run as R
import st_migration_lib as L

def stitch(prefix, extension):
    if not prefix or float(prefix[0]['time']) != 0 or abs(float(prefix[-1]['time']) - 300) > 1e-7:
        raise ValueError('Invalid preserved prefix')
    if extension and float(extension[0]['time']) <= 300:
        raise ValueError('Duplicate/overlapping restart sample')
    rows = list(prefix) + list(extension)
    times = np.array([float(r['time']) for r in rows])
    if np.any(np.diff(times) <= 0):
        raise ValueError('Non-monotonic or duplicate time')
    slots = np.rint(times / 2).astype(int)
    if not np.array_equal(slots, np.arange(slots[-1] + 1)):
        raise ValueError('Missing nominal 2M sample')
    if np.any((times - 2 * slots < -.03) | (times - 2 * slots > .25)):
        raise ValueError('Cadence skew exceeds preserved bounds')
    return rows

def window_metrics(times, values, initial, bounds, family):
    rows = []
    for lo, hi in bounds:
        if lo < times[0] or hi > times[-1] + 1e-7:
            continue
        tt = np.r_[lo, times[(times > lo) & (times < hi)], hi]
        qq = np.column_stack([np.interp(tt, times, v) for v in values.T])
        mean = np.trapz(qq, tt, axis=0) / (hi - lo)
        w = np.r_[np.diff(tt)[0] / 2, (tt[2:] - tt[:-2]) / 2, np.diff(tt)[-1] / 2]
        centered = tt - (hi + lo) / 2
        slope = np.sum(w[:, None] * centered[:, None] * (qq - mean), axis=0) / np.sum(w * centered**2)
        residual = qq - mean - centered[:, None] * slope
        for j, quantile in enumerate(range(10, 100, 10)):
            rows.append({'family': family, 'start': float(lo), 'end': float(hi), 'quantile': quantile,
                         'half_range_fraction': float(np.ptp(qq[:, j]) / (2 * initial[j])),
                         'mean_shift_fraction': float((mean[j] - initial[j]) / initial[j]),
                         'detrended_half_range_fraction': float(np.ptp(residual[:, j]) / (2 * initial[j])),
                         'slope_fraction_per_M': float(slope[j] / initial[j])})
    return rows

def compare_interior(out):
    checks = []
    for nominal in (302, 304):
        number = nominal // 2
        name = f'{CASE}.prtcl_all.{number:05d}.part.vtk'
        first = ROOT / 'runs/smoke/out/pvtk' / name
        second = out / 'pvtk' / name
        ta, ca, na, a = R.read_pvtk(str(first))
        tb, cb, nb, b = R.read_pvtk(str(second))
        if abs(ta - tb) > 1e-8 or ca != cb or na != N or nb != N:
            raise ValueError('Smoke/production interior times/cycles/counts differ')
        ia = np.argsort(a['ptag']); ib = np.argsort(b['ptag'])
        maxima = {}
        for key in a:
            aa, bb = a[key][ia], b[key][ib]
            if not np.allclose(aa, bb, rtol=1e-7, atol=1e-7):
                raise ValueError('Smoke/production per-ID mismatch: ' + key)
            maxima[key] = float(np.max(np.abs(aa - bb)))
        checks.append({'time': ta, 'cycle': ca, 'max_abs_differences': maxima,
                       'smoke_sha256': sha(first), 'production_sha256': sha(second)})
    return checks

def main():
    require_job()
    parser = argparse.ArgumentParser()
    parser.add_argument('--until', type=float, default=END)
    args = parser.parse_args()
    out = ROOT / 'runs/production/out'
    dest = ROOT / ('reduced' if args.until == END else f'reduced_{args.until:g}')
    dest.mkdir(parents=True, exist_ok=False)
    baseline = ROOT / 'baseline' / CASE
    model = dict(np.load(ROOT / 'samples/model.npz'))
    _, initial = sample(ROOT / 'samples' / (CASE + '.bin'))
    im, ix = initial[:, 7], initial[:, 1:4]
    origin = np.average(ix, weights=im, axis=0)
    order = np.argsort(np.linalg.norm(ix - origin, axis=1))
    cohort = np.empty(N, dtype=np.int8)
    cohort[order] = np.minimum((np.cumsum(im[order]) / im.sum() * 10).astype(int), 9)
    shells, profiles, cohorts, occupancy, mixing, source = [], [], [], [], [], []
    sampling_checks = []
    for path in sorted((out / 'pvtk').glob('*.part.vtk')):
        t, cycle, n, data = R.read_pvtk(str(path))
        if t > args.until + .25:
            continue
        tags = data['ptag'].astype(np.int64)
        if n != N or not np.array_equal(np.sort(tags), np.arange(N)) or any(not np.isfinite(v).all() for v in data.values()):
            raise ValueError('Invalid particle state: ' + str(path))
        masses = data['prtcl_mass']
        if not np.array_equal(masses, initial[tags, 7].astype(np.float32).astype(float)):
            raise ValueError('Per-ID mass changed')
        pos = np.column_stack([data[a] for a in 'xyz'])
        row, mass_bins, _, radii = A.sample_state(pos, masses, model, 'proper')
        slot = round(t / 2)
        row.update(time=t, cycle=cycle, nominal_time=2 * slot)
        shells.append(row); profiles.append(mass_bins)
        cc = cohort[tags]
        cm = np.bincount(cc, weights=masses, minlength=10)
        means = np.bincount(cc, weights=masses * radii, minlength=10) / cm
        cohorts.append(dict(time=t, **{f'cohort{i+1}': float(v) for i, v in enumerate(means)}))
        if slot % 50 == 0:
            ss, check = A.sampling_state(pos, masses)
            occupancy.extend([dict(time=t, **v) for v in ss])
            sampling_checks.append(dict(time=t, **check))
            ordered = np.argsort(radii)
            current = np.empty(N, dtype=np.int8)
            current[ordered] = np.minimum((np.cumsum(masses[ordered]) / masses.sum() * 10).astype(int), 9)
            transfer = np.bincount(cc * 10 + current, weights=masses, minlength=100).reshape(10, 10) / cm[:, None]
            mixing.extend({'time': t, 'initial_cohort': a + 1, 'current_mass_decile': b + 1, 'mass_fraction': float(transfer[a, b])}
                          for a in range(10) for b in range(10))
        source.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'time': t, 'cycle': cycle})
        if len(shells) % 25 == 0:
            print('Reduced continuation particles', len(shells), 'through', t, flush=True)
    joined = stitch(read_csv(baseline / 'shells.csv'), shells)
    if abs(float(joined[-1]['time']) - args.until) > .25:
        raise ValueError('Requested endpoint not covered')
    A.write_csv(dest / 'shells.csv', joined)
    A.write_csv(dest / 'cohorts.csv', stitch(read_csv(baseline / 'cohorts.csv'), cohorts))
    A.write_csv(dest / 'occupancy.csv', read_csv(baseline / 'occupancy.csv') + occupancy)
    A.write_csv(dest / 'cohort_mixing.csv', read_csv(baseline / 'cohort_mixing.csv') + mixing)
    old = dict(np.load(baseline / 'profiles.npz'))
    times = np.array([float(r['time']) for r in joined])
    np.savez_compressed(dest / 'profiles.npz', time=times, edges=old['edges'],
                        mass_bins=np.concatenate([old['mass_bins'], np.array(profiles)]),
                        target_mass=old['target_mass'], equilibrium_mass=old['equilibrium_mass'])
    new_health = [h for h in A.history(out / (CASE + '.user.hst')) if h['time'] <= args.until + .25]
    health = stitch(read_csv(baseline / 'health.csv'), new_health)
    A.write_csv(dest / 'health.csv', health)
    ga = np.atleast_2d(np.loadtxt(out / (CASE + '.z4c.user.hst')))
    ga = ga[ga[:, 0] <= args.until + .25]
    if len(ga) != len(new_health) or not np.array_equal(ga[:, 0], np.array([h['time'] for h in new_health])):
        raise ValueError('History timestamp mismatch')
    global_rows = [{'time': float(v[0]), 'dt_completed': h['dt_completed'], 'constraint_usable': h['constraint_usable'],
                    'H_rms': float(np.sqrt(max(v[3], 0) / v[10])), 'M_rms': float(np.sqrt(max(v[4], 0) / v[10]))}
                   for v, h in zip(ga, new_health)]
    A.write_csv(dest / 'global_constraints.csv', stitch(read_csv(baseline / 'global_constraints.csv'), global_rows))
    errors, checks = [], []
    for path in sorted((out / 'bin').glob('*.tme3d.*.bin')):
        rows, check = A.native_errors(path, 'proper')
        if check['time'] <= args.until + .25:
            errors.extend(rows); checks.append(check)
    A.write_csv(dest / 'field_errors.csv', read_csv(baseline / 'field_errors.csv') + errors)
    central = []
    for path in sorted((out / 'bin').glob('*.tme2d.*.bin')):
        fields = L.read_binary(str(path))
        if fields['time'] > args.until + .25:
            continue
        rr, vv, ww = L.bin_flatten(fields, 'tmunu_E', slice_x3=0.0)
        mask = rr < .75
        if not mask.any():
            raise ValueError('Central slice region missing')
        central.append({'time': float(fields['time']), 'E_central_slice_r075': float(np.average(vv[mask], weights=ww[mask])),
                        'cells': int(mask.sum()), 'radius_M': .75})
    A.write_csv(dest / 'central_energy.csv', central)
    values = np.array([[float(r[f'r{j}']) for j in range(10, 100, 10)] for r in joined])
    windows = []
    for width, family in ((100., '100M'), (P, 'P'), (2 * P, '2P')):
        bounds = [(k * width, (k + 1) * width) for k in range(int((args.until + 1e-7) / width))]
        windows.extend(window_metrics(times, values, values[0], bounds, family))
    A.write_csv(dest / 'windows.csv', windows)
    original = load(baseline / 'summary.json')
    atomic(dest / 'summary.json', {'case': CASE, 'endpoint_M': float(times[-1]), 'states': len(joined),
           'period_M': P, 'reference': 'Original t=0 radii, fixed masses and session5 metric/CIC reference',
           'field_checks': original['field_checks'] + checks, 'sampling_checks': original['sampling_checks'] + sampling_checks,
           'source_particle_outputs': source, 'smoke_production_interior': compare_interior(out),
           'prefix_manifest': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for p in sorted(baseline.glob('*')) if p.is_file()],
           'max_outside_finest_mass_fraction': max(float(r['outside_finest_mass_fraction']) for r in joined),
           'central_energy_definition': 'Coordinate-area-weighted deposited Eulerian energy on z=0, r<0.75M; new slices only; not rest-mass density',
           'utc': now()})
    print('REDUCED', len(joined), 'states; endpoint', times[-1], flush=True)

if __name__ == '__main__':
    main()
