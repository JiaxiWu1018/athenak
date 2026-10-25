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
    axes[1, 1].set(ylabel='100M window shell excursion [%]',
        xlim=(0, max(1., summary['endpoint'])*1.02), ylim=(0, None))
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
    disk = storage()
    remaining = 0.
    anchors = 0
    rates = []
    particle_bytes = cohort_bytes = checkpoint_bytes = 0
    slice_sizes = {name: 0 for name in ['z4c2d', 'tme2d', 'con2d']}
    cases = []
    for c in ALL_CASES:
        root = EXT/c['id']
        work = root/'runs/yc0500'
        state = load(work/'state.json') if (work/'state.json').exists() else None
        t = float(state['latest']['time']) if state else float(c['start'])
        complete = bool(state and state.get('complete'))
        future = 0. if complete else max(0., 1500-t)
        new_anchors = 0 if complete else sum(x > t+1e-7 for x in range(100, 1501, 100))
        if not state:
            assert c['start'] == 700, 'Fresh outward pilot must already exist'
            new_anchors += 1  # Exact inherited700M copy.
        remaining += future
        anchors += new_anchors
        cases.append(dict(case=c['id'], latest=t, complete=complete, remaining=future,
            future_scientific_anchors=new_anchors))
        if not state:
            continue
        segments = [s for s in state['segments'] if s.get('simulation_advanced', 0) > 0]
        if segments:
            rates.append(sum(s['elapsed'] for s in segments)/sum(s['simulation_advanced'] for s in segments))
            rates.extend(s['elapsed']/s['simulation_advanced'] for s in segments if s['simulation_advanced'] >= 5)
        for p in (work/'pvtk').glob('*.vtk'):
            particle_bytes = max(particle_bytes, p.stat().st_size)
        for p in (work/'cohort').glob('*.s8c'):
            cohort_bytes = max(cohort_bytes, p.stat().st_size)
        for p in (work/'checkpoint_receipts').glob('*.json'):
            checkpoint_bytes = max(checkpoint_bytes, load(p)['bytes'])
        for name in slice_sizes:
            for p in (work/'bin').glob('*.'+name+'.*.bin'):
                slice_sizes[name] = max(slice_sizes[name], p.stat().st_size)
    assert rates and min(particle_bytes, cohort_bytes, checkpoint_bytes, *slice_sizes.values()) > 0
    rate = max(rates)
    cadence_bytes = math.ceil(remaining/2)*(particle_bytes+sum(slice_sizes.values()))
    cadence_bytes += math.ceil(remaining/.5)*cohort_bytes
    # Four future intermediate recoveries, plus one control restart-identity
    # checkpoint, deliberately retained in addition to all current bytes.
    checkpoint_allowance = (anchors+5)*checkpoint_bytes
    overhead = 20*GIB
    projected = disk['own']+cadence_bytes+checkpoint_allowance+overhead
    cost = cost_seconds()/3600+rate*remaining/3600+4
    record = dict(utc=now(), job=job(), cases=cases, storage=disk,
        measured_particle_bytes=particle_bytes, measured_slice_bytes=slice_sizes,
        measured_cohort_bytes=cohort_bytes, measured_checkpoint_bytes=checkpoint_bytes,
        future_scientific_anchors=anchors, remaining_long_evolution_M=remaining,
        future_cadence_bytes=cadence_bytes, checkpoint_allowance_bytes=checkpoint_allowance,
        product_and_overhead_allowance_bytes=overhead, conservative_projected_new_bytes=projected,
        native_seconds_per_M=rate, projected_remaining_native_node_hours=rate*remaining/3600,
        spent_or_reserved_node_hours=cost_seconds()/3600, cost_overhead_allowance_node_hours=4,
        conditional_projected_total_node_hours=cost,
        cost_projection_within_cap=bool(cost <= 50), storage_projection_within_cap=bool(projected <= 350*GIB),
        projection_scope='Conditional on the maximum observed output/checkpoint sizes and native rates. Rates include aggregate accepted evolution and chunks advancing at least5M. Current bytes, all future scientific anchors, five extra checkpoint allowances and20GiB products/overhead remain included. Finer future AMR or slower throughput can change this forecast; hard guards remain independent.')
    atomic(destination/'RESOURCE_FORECAST.json', record)
    return record
