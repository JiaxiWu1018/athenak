"""Streaming output coverage and layout checks; no scientific evolution or rendering."""
import csv
import heapq
import itertools
import math
import re
import struct
from pathlib import Path

from workflow import read_rows, startup_census


def coverage(times, interval, start, end, *, slack=1.e-9, initial=True, final=True):
    times = sorted(set(times))
    if not times or not all(map(math.isfinite, times)):
        raise RuntimeError('Missing/nonfinite output times')
    gaps = [[a, b] for a, b in zip(times, times[1:]) if b - a > interval + slack]
    if times[0] < start - slack or times[-1] > end + slack:
        raise RuntimeError('Output times outside the run')
    if initial and times[0] > start + slack:
        raise RuntimeError('Missing initial output state')
    if final and abs(times[-1] - end) > slack:
        raise RuntimeError('Missing final output state')
    if gaps:
        raise RuntimeError(f'Output cadence gaps: {gaps[:8]}')
    return dict(states=len(times), first=times[0], last=times[-1],
                endpoint_gap=end - times[-1], max_gap=max((b-a for a, b in zip(times, times[1:])), default=0),
                nominal_interval=interval, cadence_gaps=gaps)


def csv_states(path, *, time_key='time', cycle_key='cycle', group_rows=None,
               finite=(), allow_empty=False):
    times, cycles, count, previous, group_count = [], [], 0, None, 0
    with Path(path).open() as stream:
        reader = csv.DictReader(stream)
        required = {time_key, cycle_key, *finite}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError('Missing diagnostic columns: ' + str(path))
        for row in reader:
            if None in row or None in row.values():
                raise RuntimeError('Incomplete diagnostic row: ' + str(path))
            tm, cycle = float(row[time_key]), int(row[cycle_key])
            if not math.isfinite(tm) or any(not math.isfinite(float(row[k])) for k in finite):
                raise RuntimeError('Nonfinite diagnostic: ' + str(path))
            state = cycle, tm
            if state != previous:
                if previous is not None:
                    if cycle <= previous[0] or tm <= previous[1]:
                        raise RuntimeError('Nonmonotonic diagnostic states: ' + str(path))
                    if group_rows is not None and group_count != group_rows:
                        raise RuntimeError('Incomplete diagnostic state: ' + str(path))
                times.append(tm); cycles.append(cycle); previous = state; group_count = 0
            group_count += 1; count += 1
        if group_rows is not None and count and group_count != group_rows:
            raise RuntimeError('Incomplete final diagnostic state: ' + str(path))
    if not count and not allow_empty:
        raise RuntimeError('Empty diagnostic: ' + str(path))
    return dict(times=times, cycles=cycles, rows=count, bytes=Path(path).stat().st_size)


def smoke_diagnostics(out):
    out = Path(out); base = 's11_smoke'
    component = read_rows(out / (base + '.components.csv'))
    first = sorted(component[:3], key=lambda r: int(r['component']))
    if ([int(r['count']) for r in first] != [3000000, 1000000, 1000000]
            or any(int(r['cycle']) != 0 or float(r['time']) != 0 for r in first)):
        raise RuntimeError('Compact GI initial census failed')
    summaries = {}
    for suffix, width, finite in [('components', 3, ('rest_mass', 'Jz')),
                                   ('radial_profiles', 1500, ('rest_mass', 'Jz')),
                                   ('trackers', 2, ('x', 'y', 'z')),
                                   ('cohort_updates', None, ('delta_x', 'delta_u1_cov'))]:
        result = csv_states(out / f'{base}.{suffix}.csv', group_rows=width, finite=finite)
        if result['cycles'][0] != 0 or result['cycles'][-1] != 2:
            raise RuntimeError('Compact GI diagnostic does not cover both endpoints: ' + suffix)
        summaries[suffix] = result
    tags = set()
    for rank in range(4):
        for row in read_rows(out / f'{base}.cohort.rank{rank}.csv'):
            if int(row['cycle']) == 0:
                tag = int(row['tag'])
                if tag in tags:
                    raise RuntimeError('Duplicate compact GI cohort tag')
                tags.add(tag)
    expected = {c + i * stride for c, stride in ((0, 750), (3000000, 250), (4000000, 250))
                for i in range(4000)}
    if tags != expected:
        raise RuntimeError('Compact GI immutable cohort census failed')
    return summaries


def binary_state(path):
    """Check native bin/cbin header and integral record length without loading fields."""
    with Path(path).open('rb') as stream:
        if stream.readline() != b'Athena binary output version=1.1\n':
            raise RuntimeError('Unexpected binary output version')
        meta = {}
        for _ in range(16):
            line = stream.readline(8192).decode()
            if '=' in line:
                key, value = line.strip().split('=', 1); meta[key] = value
            if 'header offset' in meta:
                break
        else:
            raise RuntimeError('Missing binary parameter header')
        stream.seek(int(meta['header offset']), 1)
        offset = stream.tell()
        dims = struct.unpack('<6i', stream.read(24))
        cells = math.prod(dims[i+1] - dims[i] + 1 for i in (0, 2, 4))
        record = 40 + 6 * int(meta['size of location']) + cells * int(meta['number of variables']) * int(meta['size of variable'])
        size = Path(path).stat().st_size
        if cells <= 0 or size <= offset or (size - offset) % record:
            raise RuntimeError('Truncated binary output: ' + str(path))
    return float(meta['time']), int(meta['cycle'])


def vtk_state(path):
    with Path(path).open('rb') as stream:
        def line():
            value = stream.readline(4096).decode().strip()
            while not value:
                value = stream.readline(4096).decode().strip()
                if stream.tell() >= Path(path).stat().st_size:
                    raise RuntimeError('Truncated particle VTK')
            return value
        if line() != '# vtk DataFile Version 2.0':
            raise RuntimeError('Unexpected particle VTK version')
        header = line()
        tm = float(re.search(r'time=\s*(\S+)', header)[1])
        cycle = int(re.search(r'cycle=\s*(\d+)', header)[1])
        if line() != 'BINARY' or line() != 'DATASET UNSTRUCTURED_GRID':
            raise RuntimeError('Unexpected particle VTK structure')
        points = line().split()
        if len(points) != 3 or points[0] != 'POINTS' or points[2] != 'float':
            raise RuntimeError('Invalid particle VTK points')
        n = int(points[1]); stream.seek(n * 12, 1)
        if line() != f'POINT_DATA {n}':
            raise RuntimeError('Invalid particle VTK data count')
        for name in ('gid', 'ptag', 'prtcl_vel', 'prtcl_energy', 'prtcl_mass'):
            vector = name == 'prtcl_vel'
            expected = ('VECTORS' if vector else 'SCALARS') + ' ' + name + ' float'
            if line() != expected:
                raise RuntimeError('Missing particle VTK field: ' + name)
            if not vector and line() != 'LOOKUP_TABLE default':
                raise RuntimeError('Invalid particle VTK lookup table')
            stream.seek(n * (12 if vector else 4), 1)
        if stream.tell() != Path(path).stat().st_size:
            raise RuntimeError('Truncated/incompatible particle VTK payload')
    return tm, cycle


def table_times(path, width=None):
    result = []
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip() or line.startswith('#'):
                continue
            values = list(map(float, line.split()))
            if ((width is not None and len(values) != width)
                    or not all(map(math.isfinite, values))):
                raise RuntimeError('Invalid native text output: ' + str(path))
            if result and values[0] <= result[-1]:
                raise RuntimeError('Nonmonotonic native text output')
            result.append(values[0])
    return result


def removal_ledger(path, final_particles):
    expected = 5_000_000 - final_particles
    if not path.exists():
        if expected:
            raise RuntimeError('Missing native particle-removal ledger')
        return dict(records=0)
    seen = bytearray(5_000_000); count = 0
    with path.open() as stream:
        fields = stream.readline().lstrip('# ').strip().split(',')
        for row in csv.DictReader(stream, fieldnames=fields):
            tag = int(row['tag'])
            component = 0 if tag < 3000000 else 1 if tag < 4000000 else 2
            if (not 0 <= tag < 5_000_000 or seen[tag]
                    or int(row['component']) != component
                    or row['reason'] not in ('exit', 'sphere', 'lapse')
                    or not all(math.isfinite(float(row[k])) for k in
                               ('time', 'x1', 'x2', 'x3', 'u1_cov', 'u2_cov', 'u3_cov', 'mass'))
                    or float(row['mass']) <= 0):
                raise RuntimeError('Invalid/duplicate particle-removal record')
            seen[tag] = 1; count += 1
    if count != expected:
        raise RuntimeError('Particle-removal ledger/native final census mismatch')
    return dict(records=count, native_final_particles=final_particles)


def quadrature(out, base, horizon, published, ranks):
    """Verify each published surface angle has exactly one owner; never sum copies."""
    def records(path):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                yield (int(row['cycle']), float(row['time'])), row
    with_files = [out / f'{base}.ah_quadrature_{horizon}.rank{r}.csv' for r in range(ranks)]
    merged = heapq.merge(*(records(p) for p in with_files), key=lambda item: item[0])
    checked = set()
    for state, group in itertools.groupby(merged, key=lambda item: item[0]):
        angles = set()
        for _, row in group:
            if state not in published:
                continue
            angle = int(row['angle'])
            if angle in angles or not all(math.isfinite(float(v)) for v in row.values()):
                raise RuntimeError('Duplicate owner/nonfinite accepted surface quadrature')
            angles.add(angle)
        if state in published:
            if angles != set(range(200)):
                raise RuntimeError('Incomplete accepted surface quadrature')
            checked.add(state)
    if checked != published:
        raise RuntimeError('Missing accepted surface quadrature states')
    return dict(published_states=len(checked), unique_owned_angles_per_state=200)


def audit_production(out, checkpoints, ranks=16):
    out = Path(out); base = 's11_orbit_gw'; end = 150.
    report = dict(passed=False, products={}, errors=[], scientific_analysis_pending=True)
    products, errors = report['products'], report['errors']
    def inspect(label, fn):
        try:
            products[label] = fn()
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, struct.error) as exc:
            errors.append(label + ': ' + str(exc))
    with (out / f'{base}.components.csv').open() as stream:
        initial = list(itertools.islice(csv.DictReader(stream), 3))
    startup_census(initial)
    comp = csv_states(out / f'{base}.components.csv', group_rows=3, finite=('dt', 'rest_mass', 'Jz'))
    # Per-output times may be rounded, or occur on the completed step after a target.
    max_dt = 0.
    with (out / f'{base}.components.csv').open() as stream:
        for row in csv.DictReader(stream):
            max_dt = max(max_dt, float(row['dt']))
    slack = max_dt + .002
    inspect('components', lambda: coverage(comp['times'], .025, 0, end, slack=slack))
    for suffix, width, finite in [('radial_profiles', 1500, ('rest_mass', 'Jz')),
                                   ('trackers', 2, ('x', 'y', 'z')),
                                   ('cohort_updates', None, ('delta_x', 'delta_u1_cov'))]:
        def diagnostic(suffix=suffix, width=width, finite=finite):
            result = csv_states(out / f'{base}.{suffix}.csv', group_rows=width, finite=finite)
            # Sparse cohort updates are emitted only for changed surviving/removing tags.
            if suffix == 'cohort_updates':
                return dict(rows=result['rows'], first=result['times'][0], last=result['times'][-1], sparse=True)
            return coverage(result['times'], .025, 0, end, slack=slack)
        inspect(suffix, diagnostic)
    for rank in range(ranks):
        inspect(f'cohort_rank{rank}', lambda rank=rank: csv_states(
            out / f'{base}.cohort.rank{rank}.csv', finite=('mass', 'x', 'u1_cov'), allow_empty=True) | {'sparse': True})
    inspect('history', lambda: coverage(table_times(out / f'{base}.z4c.user.hst'), .025, 0, end, slack=slack))
    ids = {f'{field}_{plane}': .25 for field in ('tmunu', 'con', 'z4c', 'weyl') for plane in ('xy', 'xz', 'yz')}
    ids.update({name: 10. for name in ('E3d', 'con3d', 'z4c3d', 'weyl3d')})
    for name, interval in ids.items():
        inspect(name, lambda name=name, interval=interval: coverage(
            [binary_state(p)[0] for p in sorted((out / 'bin').glob(f'{base}.{name}.*.bin'))],
            interval, 0, end, slack=slack))
    inspect('mesh', lambda: coverage([binary_state(p)[0] for p in sorted(
        (out / 'cbin_mesh_32').glob(f'{base}.mesh.*.cbin'))], .025, 0, end, slack=slack))
    inspect('particles', lambda: coverage([vtk_state(p)[0] for p in sorted(
        (out / 'pvtk').glob(f'{base}.*.part.vtk'))], .5, 0, end, slack=slack))
    inspect('checkpoints', lambda: coverage([cp['time'] for cp in checkpoints], 10, 0, end, slack=slack))
    final_cp = max(checkpoints, key=lambda cp: cp['time'])
    inspect('particle_removal_ledger', lambda: removal_ledger(out / f'{base}.prtcl_destroy.csv', final_cp['particles']))
    def waves():
        mapping = read_rows(out / 'wave_state_times.csv')
        legacy = [float(r['legacy_time']) for r in mapping]
        state = [float(r['state_time']) for r in mapping]
        for radius in (40, 50, 60):
            for part in ('real', 'imag'):
                saved = table_times(out / f'waveforms/rpsi4_{part}_{radius:04d}.txt', 78)
                if saved != legacy:
                    raise RuntimeError('Raw waveform/state sidecar mismatch')
        result = coverage(state, .025, 0, end, slack=slack, initial=False, final=False)
        if state[0] > .025 + slack or end - state[-1] > .025 + slack:
            raise RuntimeError('Waveform endpoint coverage exceeds one nominal interval plus timestep')
        return result | dict(radii=[40, 50, 60], modes='l=2..8', exact_t150_sample=state[-1] == end)
    inspect('waves', waves)
    for horizon in range(3):
        def ah(horizon=horizon):
            path = out / f'{base}.ah_{horizon}.csv'
            if horizon == 2 and not path.exists():
                return dict(attempts=0, note='Conditional common finder produced no attempts; inspect trackers')
            attempts = read_rows(path)
            csv_states(path, time_key='state_time', cycle_key='state_cycle')
            published = {(int(r['state_cycle']), float(r['state_time'])) for r in attempts if r['published'] == '1'}
            accepted_times = sorted(t for _, t in published)
            result = dict(attempts=len(attempts), published=len(published),
                          first_attempt=float(attempts[0]['state_time']), last_attempt=float(attempts[-1]['state_time']),
                          accepted_gaps=[[a,b] for a,b in zip(accepted_times, accepted_times[1:]) if b-a > .025+slack])
            if published:
                coeff = out / f'{base}.ah_coefficients_{horizon}.csv'
                if not coeff.is_file() or not coeff.stat().st_size:
                    raise RuntimeError('Missing accepted surface coefficients')
                result['quadrature'] = quadrature(out, base, horizon, published, ranks)
            return result
        inspect(f'ah_{horizon}', ah)
    report['passed'] = not errors
    report['max_recorded_completed_dt'] = max_dt
    report['coverage_slack'] = slack
    return report
