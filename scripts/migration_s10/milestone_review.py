"""Allocated trend evidence and a conditional forecast at each later milestone."""
import math
import shutil
import numpy as np
from common import *
from reduce import csv_rows, write_csv


def science_trends(c, red, destination):
    job()
    summary = load(red/'summary.json')
    shells = csv_rows(red/'shells.csv')
    flow = csv_rows(red/'coordinate_radial_flow.csv') if (red/'coordinate_radial_flow.csv').exists() else []
    z = np.load(red/'coordinate_rest_mass_profiles.npz')
    # Final dumps/restarts can repeat a native time. Give each actual time one
    # equal-weight observation, preserving the most recent identical state.
    indices = {round(float(t), 7): i for i, t in enumerate(z['time'])}
    times = np.array(sorted(indices))
    bins = np.array([z['mass_per_bin'][indices[t]] for t in times])
    m0 = summary['rest_mass_initial']
    assert len(times) and np.isfinite(bins).all() and (bins >= 0).all()
    assert np.allclose(bins.sum(axis=1), m0, rtol=1e-12, atol=1e-14)
    drift = .5*np.abs(bins-bins[0]).sum(axis=1)/m0
    assert np.isfinite(drift).all() and (drift >= 0).all() and (drift <= 1+1e-12).all()
    windows = []
    start = math.ceil((shells[0]['time']-1e-7)/100)*100
    for lo in range(start, math.floor((shells[-1]['time']+1e-7)/100)*100, 100):
        rows = [r for r in shells if lo-1e-7 <= r['time'] <= lo+100+1e-7]
        assert rows and abs(rows[0]['time']-lo) < 1e-6 and abs(rows[-1]['time']-lo-100) < 1e-6
        for q in [10, 50, 90]:
            values = [r[f'r{q}'] for r in rows]
            mean = float(np.mean(values))
            windows.append(dict(start=lo, end=lo+100, shell=q, samples=len(values),
                minimum=min(values), maximum=max(values), mean=mean,
                fractional_peak_to_peak=(max(values)-min(values))/mean))
    if windows:
        write_csv(destination/'window_shell_excursions.csv', windows)
    write_csv(destination/'profile_redistribution.csv', [dict(time=float(t),
        fraction=float(v)) for t, v in zip(times, drift)])
    import matplotlib.pyplot as plt
    from publication_style import style
    style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    centers = .5*(z['edges'][1:]+z['edges'][:-1])
    volumes = 4*np.pi/3*np.diff(z['edges']**3)
    colors = ['#245d88', '#bd5e28', '#7b6340']
    for k, color, line in zip(sorted({0, len(times)//2, len(times)-1}), colors, ['-', '--', ':']):
        density = bins[k]/volumes
        axes[0, 0].loglog(centers, np.where(density > 0, density, np.nan),
            color=color, ls=line, label=f'Native {times[k]:g}M')
    occupied = np.flatnonzero(bins.sum(axis=0) > 0)
    axes[0, 0].set(xlabel='COM coordinate radius [M]',
        ylabel='Rest mass / coordinate shell volume\n[M^-2]',
        xlim=(.03, max(8., float(centers[occupied[-1]])*1.1)))
    axes[0, 1].plot(times, 100*drift, color=colors[0])
    axes[0, 1].set(ylabel=f'Binned rest-mass redistribution\nfrom {times[0]:g}M [%]')
    if flow:
        for name, label, color, line in [('mean_coordinate_dr_dt', 'Mean', colors[0], '-'),
                ('rms_coordinate_dr_dt', 'RMS', colors[1], '--')]:
            axes[1, 0].plot([r['time'] for r in flow], [r[name] for r in flow], label=label, color=color, ls=line)
        axes[1, 0].axhline(0, color='.5', lw=.6)
    axes[1, 0].set(ylabel='Fixed-ID coordinate dr/dt')
    for q, color, line in zip([10, 50, 90], colors, ['-', '--', ':']):
        rows = [r for r in windows if r['shell'] == q]
        if rows:
            axes[1, 1].plot([r['end'] for r in rows], [100*r['fractional_peak_to_peak'] for r in rows],
                marker='o', color=color, ls=line, label=f'{q}% shell')
    excursion_max = max([100*r['fractional_peak_to_peak'] for r in windows]+[1.])
    axes[1, 1].set(ylabel='100M window shell excursion [%]',
        xlim=(0, max(1., summary['endpoint'])*1.02), ylim=(0, excursion_max*1.12))
    if not windows:
        axes[1, 1].text(.5, .5, 'No complete 100M window', ha='center', transform=axes[1, 1].transAxes)
    for ax in axes.flat:
        ax.grid(alpha=.18)
        if ax is not axes[0, 0]:
            ax.set_xlabel('t / M_ref')
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.suptitle(f"{c['key']}: profiles, flow and finite-window excursions through {summary['endpoint']:g}M")
    fig.supxlabel('Coordinate diagnostics; shell/profile curves stop before first removal. Window excursions include drift and phase, not isolated damping.', fontsize=9)
    for suffix in ['png', 'pdf']:
        fig.savefig(destination/f'trends.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    # Keep the exact compact inputs for every milestone figure/table even as
    # the current reduced/ directory later grows with the next trajectory tail.
    saved = destination/'data'
    saved.mkdir()
    for p in sorted(red.iterdir()):
        if p.is_file() and (p.suffix in ['.csv', '.npz'] or p.name in ['summary.json', 'REDUCTION_OK.json']):
            shutil.copy2(p, saved/p.name)
    record = dict(utc=now(), job=job(), case=c, endpoint=summary['endpoint'],
        initial_r50=shells[0]['r50'], final_r50=shells[-1]['r50'], final_r90=shells[-1]['r90'],
        last_uncensored_shell_time=shells[-1]['time'], first_removal=summary['first_removal'],
        profile_baseline_time=float(times[0]), last_profile_time=float(times[-1]),
        final_profile_redistribution=float(drift[-1]),
        window_definition='Closed adjacent 100M windows; (maximum-minimum)/sample mean. Includes drift and phase; not an isolated oscillation amplitude or a damping/stability test.',
        shell_windows=windows, source_inputs=[receipt(p) for p in sorted(saved.iterdir())],
        profiles_are_coordinate_rest_mass=True, numerical_checks_passed=True,
        actual_visual_review=False)
    atomic(destination/'SCIENCE_TRENDS.json', record)
    return record


def forecast(destination):
    job()
    from extension_resources import storage, cost_seconds
    from resource_projection import project
    disk = storage()
    spent = cost_seconds()
    snapshots = destination/'projection_inputs'
    snapshots.mkdir()
    cases = []
    for c in ALL_CASES:
        root = EXT/c['id']
        work = root/'runs/yc0500'
        state_path = work/'state.json'
        state = None
        source = None
        if state_path.exists():
            # Copy the bytes read once: another case can advance concurrently.
            data = state_path.read_bytes()
            frozen = snapshots/(c['id']+'_state.json')
            frozen.write_bytes(data)
            state = json.loads(data)
            source = dict(native_path=str(state_path), frozen=receipt(frozen))
        t = float(state['latest']['time']) if state else float(c['start'])
        complete = bool(state and state.get('complete'))
        future = 0. if complete else max(0., 1500-t)
        new_anchors = 0 if complete else sum(x > t+1e-7 for x in range(100, 1501, 100))
        if not state:
            assert c['start'] == 700, 'Fresh outward pilot must already exist'
            new_anchors += 1  # Exact inherited700M copy.
        row = dict(case=c['id'], latest=t, complete=complete, remaining=future,
            future_scientific_anchors=new_anchors, source_state=source,
            observed_native_seconds_per_M=None, particle_bytes=0, cohort_bytes=0,
            checkpoint_bytes=0, slice_bytes={name: 0 for name in ['z4c2d', 'tme2d', 'con2d']})
        cases.append(row)
        if not state:
            continue
        advanced = [s for s in state['segments'] if s.get('simulation_advanced', 0) > 0]
        segments = [s for s in advanced if s.get('elapsed') is not None
                    and math.isfinite(s['elapsed']) and s['elapsed'] > 0]
        row['segments_without_valid_elapsed'] = len(advanced)-len(segments)
        if segments:
            rates = [sum(s['elapsed'] for s in segments)/sum(s['simulation_advanced'] for s in segments)]
            rates.extend(s['elapsed']/s['simulation_advanced'] for s in segments if s['simulation_advanced'] >= 5)
            row['observed_native_seconds_per_M'] = max(rates)
        for p in (work/'pvtk').glob('*.vtk'):
            row['particle_bytes'] = max(row['particle_bytes'], p.stat().st_size)
        for p in (work/'cohort').glob('*.s8c'):
            row['cohort_bytes'] = max(row['cohort_bytes'], p.stat().st_size)
        for p in (work/'checkpoint_receipts').glob('*.json'):
            row['checkpoint_bytes'] = max(row['checkpoint_bytes'], load(p)['bytes'])
        for name in row['slice_bytes']:
            for p in (work/'bin').glob('*.'+name+'.*.bin'):
                row['slice_bytes'][name] = max(row['slice_bytes'][name], p.stat().st_size)
    record = dict(utc=now(), job=job(), storage=disk, **project(cases, disk['own'], spent),
        projection_scope='Conditional per-case maxima of observed output/checkpoint sizes and native rates. Rates include aggregate accepted evolution and chunks advancing at least5M; unobserved case measurements fall back to the global observed maximum. Current bytes, all future science anchors, two extra recoveries per case, one additional maximum-sized identity checkpoint,20GiB products and4nodeh overhead remain included. Future AMR or throughput changes can invalidate the forecast; shared50h/350GiB and live guards remain unchanged.')
    atomic(destination/'RESOURCE_FORECAST.json', record)
    return record
