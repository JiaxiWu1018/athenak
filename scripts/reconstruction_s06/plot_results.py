"""Two static scientific figures and descriptive, auditable trend metrics."""
from session import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_extension import window_metrics

COLORS = ['#0072B2', '#D55E00', '#009E73']

def column(rows, name):
    return np.array([float(r[name]) for r in rows])

def usable(rows):
    return np.array([str(r['constraint_usable']).lower() == 'true' for r in rows])

def panel_axis(ax):
    ax.grid(alpha=.2)
    ax.set_xlim(0, END / P)
    ax.axvline(300 / P, color='.45', ls='--', lw=.9)
    for t in range(1, 6):
        ax.axvline(t, color='.7', lw=.5, alpha=.4)
    ax.set_xlabel(r'$t/P_{\rm surface}$')

def save_figure(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=180)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)

def weighted_mean(rows, key, lo, hi, mask=None):
    times, values = column(rows, 'time'), column(rows, key)
    use = (times >= lo - 1e-7) & (times <= hi + 1e-7)
    if mask is not None:
        use &= mask
    t, v = times[use], values[use]
    if len(t) < 2:
        return None
    return {'mean': float(np.trapz(v, t) / (t[-1] - t[0])), 'min': float(v.min()),
            'max': float(v.max()), 'first_time': float(t[0]), 'last_time': float(t[-1]), 'samples': len(t)}

def main():
    require_job()
    dest = ROOT / 'reduced'
    figures = ROOT / 'plots'
    figures.mkdir(exist_ok=True)
    shells = read_csv(dest / 'shells.csv')
    health = read_csv(dest / 'health.csv')
    constraints = read_csv(dest / 'global_constraints.csv')
    central = read_csv(dest / 'central_energy.csv')
    windows = read_csv(dest / 'windows.csv')
    summary = load(dest / 'summary.json')
    t = column(shells, 'time'); x = t / P
    th = column(health, 'time') / P
    hm = usable(health); gm = usable(constraints)
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'legend.fontsize': 8, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    ax = axes[0, 0]
    for quantile, color in zip((10, 50, 90), COLORS):
        radius = column(shells, f'r{quantile}')
        origin = column(shells, f'r{quantile}_origin')
        ax.plot(x, 100 * (radius / radius[0] - 1), color=color, label=f'r{quantile}, COM')
        ax.plot(x, 100 * (origin / origin[0] - 1), color=color, ls=':', lw=.9, alpha=.8)
    ax.axhline(0, color='.4', lw=.7)
    ax.set(title='Enclosing rest-mass shells (dotted: origin)', ylabel='Change from own t=0 radius [%]')
    ax.legend(ncol=3)
    ax = axes[0, 1]
    for name, color in zip(('com_x', 'com_y', 'com_z'), COLORS):
        ax.plot(x, column(shells, name), color=color, label=name[-1])
    ax.plot(x, column(shells, 'com_mag'), color='k', ls='--', label='|COM|')
    ax.set(title='Rest-mass centre motion', ylabel='Coordinate position [M]')
    ax.legend(ncol=4)
    ax = axes[1, 0]
    ax.plot(th, column(health, 'core_lapse'), color=COLORS[0], label='r<0.75M core')
    ax.set(title='Lapse (fixed origin regions)', ylabel='Core mean lapse')
    outer = ax.twinx()
    outer.plot(th, column(health, 'outer_lapse'), color=COLORS[1], alpha=.8, label='96<r/M<120')
    outer.set_ylabel('Outer mean lapse', color=COLORS[1])
    ax.legend(loc='upper left'); outer.legend(loc='upper right')
    ax = axes[1, 1]
    mean_e = column(health, 'E_mean_r10')
    ax.plot(th, 100 * (mean_e / mean_e[0] - 1), color=COLORS[0], label='r<10M mean E, relative to t=0')
    profiles = np.load(dest / 'profiles.npz')
    volume = 4 * np.pi / 3 * np.diff(profiles['edges']**3)
    rho = profiles['mass_bins'][:, 0] / volume[0]
    ax.plot(x, 100 * (rho / rho[0] - 1), color=COLORS[2], alpha=.7, lw=.8,
            label='COM r<0.5M coordinate rest-mass density, relative to t=0')
    ax.set(title='Density proxies (different definitions)', ylabel='Change [%]')
    ce = ax.twinx()
    ce.plot(column(central, 'time') / P, column(central, 'E_central_slice_r075'),
            color=COLORS[1], lw=.8, alpha=.8)
    ce.set_ylabel(r'Central slice $E$ [$M^{-2}$]', color=COLORS[1])
    ax.legend(loc='lower left', fontsize=7)
    ax.text(.02, .98, 'Orange: new z=0, r<0.75M E; no pre-300M slices', transform=ax.transAxes,
            va='top', fontsize=7)
    ax = axes[2, 0]
    for quantity, color in (('H', COLORS[0]), ('M', COLORS[1])):
        for radius, style in ((10, '-'), (16, '--')):
            ax.plot(th[hm], column(health, f'{quantity}_rms_r{radius}')[hm], style,
                    color=color, label=f'{quantity}, r<{radius}M', lw=1)
    ax.set_yscale('log')
    ax.set(title='Core constraints: coordinate-volume RMS', ylabel=r'RMS [$M^{-2}$]')
    ax.legend(ncol=2)
    ax = axes[2, 1]
    tc = column(constraints, 'time') / P
    for quantity, color in (('H', COLORS[0]), ('M', COLORS[1])):
        ax.plot(tc[gm], column(constraints, quantity + '_rms')[gm], color=color, label=quantity)
    ax.set_yscale('log')
    ax.set(title='Global constraints and density departure', ylabel=r'Proper-volume RMS [$M^{-2}$]')
    error = ax.twinx()
    checks = summary['field_checks']
    error.plot([c['time'] / P for c in checks], [100 * c['E_relative_L2_equilibrium'] for c in checks],
               'o-', color=COLORS[2], ms=3, label='E error to initial equilibrium')
    error.set_ylabel('r<10M E relative L2 error [%]', color=COLORS[2])
    ax.legend(loc='upper left'); error.legend(loc='upper right', fontsize=7)
    for ax in axes.flat:
        panel_axis(ax)
    fig.suptitle(r'ST $y_c=0.819$: proper profile, stratified seed 1, $N=1.5\times10^6$' +
                 '\nUnchanged live-Z4c setup; dashed vertical line: restart at 300M; reference period 197.626M', fontsize=13)
    save_figure(fig, figures / 'f1_long_baseline_histories')
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    styles = [('100M', 'o', COLORS[0]), ('P', 's', COLORS[1]), ('2P', 'D', COLORS[2])]
    for row, quantile in enumerate((10, 50, 90)):
        for family, marker, color in styles:
            selected = [r for r in windows if r['family'] == family and int(r['quantile']) == quantile]
            xx = np.array([(float(r['start']) + float(r['end'])) / (2 * P) for r in selected])
            axes[row, 0].plot(xx, 100 * column(selected, 'half_range_fraction'), marker + '-',
                              color=color, ms=4, label=family + ' raw')
            axes[row, 1].plot(xx, 100 * column(selected, 'mean_shift_fraction'), marker + '-',
                              color=color, ms=4, label=family + ' mean')
            if family == '100M':
                axes[row, 0].plot(xx, 100 * column(selected, 'detrended_half_range_fraction'), 'o:',
                                  color=color, ms=3, label='100M detrended')
        axes[row, 0].set(title=f'r{quantile}: window excursion', ylabel='Half peak-to-peak / initial radius [%]')
        axes[row, 1].set(title=f'r{quantile}: mean-radius shift', ylabel='Window mean shift / initial radius [%]')
        axes[row, 1].axhline(0, color='.4', lw=.7)
        for ax in axes[row]:
            panel_axis(ax)
        if row == 0:
            axes[row, 0].legend(fontsize=8, ncol=2); axes[row, 1].legend(fontsize=8)
    fig.suptitle('Oscillation excursion and mean drift are reported separately\n'
                 'Complete non-overlapping windows; markers at window centres; incomplete period tails excluded', fontsize=13)
    save_figure(fig, figures / 'f2_shell_amplitude_and_drift')
    values = np.array([[float(r[f'r{j}']) for j in range(10, 100, 10)] for r in shells])
    drift = window_metrics(t, values, values[0], [(P, 5 * P)], 'post_first_P')
    comparisons = []
    for quantile in range(10, 100, 10):
        period_rows = [r for r in windows if r['family'] == 'P' and int(r['quantile']) == quantile]
        early, late = period_rows[1], period_rows[4]
        raw = float(early['half_range_fraction']); det = float(early['detrended_half_range_fraction'])
        trend = next(r for r in drift if r['quantile'] == quantile)
        comparisons.append({'quantile': quantile, 'early_window_M': [float(early['start']), float(early['end'])],
                            'late_window_M': [float(late['start']), float(late['end'])],
                            'early_raw_fraction': raw, 'late_raw_fraction': float(late['half_range_fraction']),
                            'late_over_early_raw': float(late['half_range_fraction']) / raw if raw > 1e-12 else None,
                            'late_over_early_detrended': float(late['detrended_half_range_fraction']) / det if det > 1e-12 else None,
                            'mean_shift_change_fraction': float(late['mean_shift_fraction']) - float(early['mean_shift_fraction']),
                            'linear_slope_fraction_per_P': trend['slope_fraction_per_M'] * P,
                            'endpoint_change_fraction': values[-1, quantile // 10 - 1] / values[0, quantile // 10 - 1] - 1})
    health_metrics = {}
    for key in ('core_lapse', 'outer_lapse', 'E_mean_r10', 'H_rms_r10', 'M_rms_r10', 'H_rms_r16', 'M_rms_r16'):
        mask = hm if key.startswith(('H_', 'M_')) else None
        health_metrics[key] = {'early': weighted_mean(health, key, P, 2 * P, mask),
                               'late': weighted_mean(health, key, 4 * P, 5 * P, mask)}
    manifest = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                for p in sorted(figures.glob('f[12]*'))]
    metrics = {'utc': now(), 'period_M': P, 'final_time_M': float(t[-1]), 'total_periods': float(t[-1] / P),
               'shell_comparisons': comparisons, 'health': health_metrics,
               'com_initial_M': float(shells[0]['com_mag']), 'com_final_M': float(shells[-1]['com_mag']),
               'com_max_M': float(column(shells, 'com_mag').max()),
               'central_coordinate_density_definition': 'Rest mass / coordinate volume in COM-centred r<0.5M, same 0.5M profile bin as session5',
               'scientific_gate': 'Descriptive results only; no preset scientific pass/fail; user reviews later paper rerun',
               'figures': manifest}
    atomic(ROOT / 'evidence/DESCRIPTIVE_METRICS.json', metrics)
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
