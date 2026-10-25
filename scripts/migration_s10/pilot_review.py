"""Allocated matched-pilot comparison and measured, conditional resource forecast."""
import sys
import re
import numpy as np
from common import *
from reduce import reduce_case, csv_rows


def control_config():
    return dict(CASE, U=0., key='matched_u0_20M', target=20.)


def review(c, destination):
    job()
    initial = load(ROOT/'evidence/INITIAL_CONSTRAINT_ADMISSION.json')
    control = Path(initial['control_root'])
    short = load(control/'evidence/SHORT_CONTROL_OK.json')
    assert short['checkpoint']['time'] == 20
    env = os.environ.copy()
    env['ROOT'] = str(control)
    cmd(['python3', Path(__file__).resolve(), '--reduce-control'], env=env)
    roots = [control, ROOT]
    labels = ['Matched U=0', 'Outward U=+0.005']
    results = []
    import matplotlib.pyplot as plt
    from publication_style import style
    style()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for root, label in zip(roots, labels):
        red = root/'reduced/yc0500'
        summary = load(red/'summary.json')
        assert summary['endpoint'] == 20 and summary['first_removal'] is None
        assert summary['ah_first'] is None
        shell = csv_rows(red/'shells.csv')
        history = csv_rows(red/'history.csv')
        flow = csv_rows(red/'coordinate_radial_flow.csv')
        profile = np.load(red/'coordinate_rest_mass_profiles.npz')
        bins = profile['mass_per_bin']
        # This is a binned rest-mass redistribution statistic, not energy density.
        drift = .5*np.abs(bins-bins[0]).sum(axis=1)/summary['rest_mass_initial']
        axes[0, 0].plot([r['time'] for r in shell], [r['r50'] for r in shell], label=label)
        axes[0, 1].plot([r['time'] for r in history], [r['core_lapse'] for r in history], label=label)
        axes[0, 2].plot(profile['time'], drift, label=label)
        axes[1, 0].semilogy([r['time'] for r in history], [r['H_rms_r10'] for r in history], label=label)
        axes[1, 1].semilogy([r['time'] for r in history], [r['M_rms_r10'] for r in history], label=label)
        axes[1, 2].plot([r['time'] for r in flow], [r['mean_coordinate_dr_dt'] for r in flow], label=label)
        state = load(root/'runs/yc0500/state.json')
        elapsed = sum(s['elapsed'] for s in state['segments'] if s.get('simulation_advanced', 0) > 0)
        warnings = []
        for segment in state['segments']:
            p = Path(segment['log'])
            text = p.read_text(errors='replace')
            assert '### FATAL' not in text and not re.search(r'search_fail=[1-9]', text)
            warnings += [line for line in text.splitlines() if 'fell back' in line or 'search_fail=' in line]
        results.append(dict(label=label, root=str(root), summary=summary,
            initial_r50=shell[0]['r50'], final_r50=shell[-1]['r50'],
            final_r90=shell[-1]['r90'], initial_core_lapse=history[0]['core_lapse'],
            final_core_lapse=history[-1]['core_lapse'],
            final_binned_rest_mass_redistribution=float(drift[-1]),
            final_mean_coordinate_dr_dt=flow[-1]['mean_coordinate_dr_dt'],
            native_seconds=elapsed, native_seconds_per_M=elapsed/20,
            pusher_warning_lines=warnings))
        # Restart admission may originate in an earlier allocation.
        results[-1]['restart_identity'] = load(root/'evidence/PRODUCTION_ADMISSION.json')
    ylabels = ['COM half-rest-mass radius [M]', 'Volume-mean core lapse',
        'Binned rest-mass redistribution from t=0', 'Hamiltonian RMS, r<10M',
        'Momentum RMS, r<10M', 'Cohort mean coordinate dr/dt']
    for ax, ylabel in zip(axes.flat, ylabels):
        ax.set(xlabel='t / M_ref', ylabel=ylabel)
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)
    fig.suptitle('Session 10 matched 20M pilot; no damping inference from this short interval')
    for suffix in ['png', 'pdf']:
        fig.savefig(destination/f'matched_pilot.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    from extension_resources import storage, cost_seconds
    disk = storage()
    work = ROOT/'runs/yc0500'
    snapshots = load(ROOT/'reduced/yc0500/snapshots.json')
    particle_bytes = max(s['bytes'] for s in snapshots)
    slice_sizes = {}
    for name in ['z4c2d', 'tme2d', 'con2d']:
        files = list((work/'bin').glob('*.'+name+'.*.bin'))
        assert files
        slice_sizes[name] = max(p.stat().st_size for p in files)
    cohort_bytes = max(p.stat().st_size for p in (work/'cohort').glob('*.s8c'))
    checkpoint_bytes = load(work/'state.json')['latest']['bytes']
    # At most 26 scientific/pilot/long-control anchors, plus four intermediate
    # recoveries across the two long trajectories. Current bytes are retained
    # in full; these allowances deliberately overcount already-present anchors.
    future_cadence_bytes = (particle_bytes+sum(slice_sizes.values()))*(2300-20)/2
    future_cadence_bytes += cohort_bytes*(2300-20)/.5
    checkpoint_allowance = 30*checkpoint_bytes
    product_and_overhead_allowance = 20*GIB
    projected_bytes = disk['own']+future_cadence_bytes+checkpoint_allowance+product_and_overhead_allowance
    rate = max(r['native_seconds_per_M'] for r in results)
    forecast = dict(utc=now(), job=job(), storage=disk,
        measured_particle_bytes=particle_bytes, measured_slice_bytes=slice_sizes,
        measured_cohort_bytes=cohort_bytes, measured_checkpoint_bytes=checkpoint_bytes,
        future_cadence_bytes=future_cadence_bytes, checkpoint_allowance_bytes=checkpoint_allowance,
        product_and_overhead_allowance_bytes=product_and_overhead_allowance,
        conservative_projected_new_bytes=projected_bytes, storage_limit_bytes=350*GIB,
        native_seconds_per_M=rate, remaining_long_evolution_M=2280,
        projected_remaining_native_node_hours=rate*2280/3600,
        spent_or_reserved_node_hours=cost_seconds()/3600,
        projection_scope='Conditional on pilot grid/output sizes and rate; refresh at 100M and subsequent milestones. Future cadence covers both long trajectories. Additional 20GiB allows analysis, rendering and exports. Existing data and all 30 checkpoint allowances are retained in this deliberately conservative sum.',
        storage_projection_within_cap=bool(projected_bytes <= 350*GIB),
        cost_projection_within_cap=bool(cost_seconds()/3600+rate*2280/3600+4 <= 50),
        cost_overhead_allowance_node_hours=4.)
    atomic(destination/'MATCHED_PILOT.json', dict(utc=now(), job=job(), results=results,
        forecast=forecast, figures=[receipt(p) for p in sorted(destination.glob('matched_pilot.*'))],
        numerical_passed=True, actual_visual_review=False))
    atomic(EXT/'evidence/MEASURED_PILOT_FORECAST.json', forecast)
    return dict(comparison=str(destination/'MATCHED_PILOT.json'), forecast=forecast)


if __name__ == '__main__':
    assert sys.argv[1:] == ['--reduce-control']
    reduce_case(control_config(), require_complete=False)
