"""Allocated independent audit and provisional review of one finished case.

Keep this separate from the final four-case comparison while 6M is evolving.
All diagnostic definitions and independent audit functions are unchanged.
"""
import argparse
import subprocess
import numpy as np
from cfg import *
import audit_results as audit


def entries(paths):
    return [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in paths]


def review(index, execution_source):
    require_job()
    case = CASES[index]
    name = case['id']
    dest = ROOT/'interim'/name
    dest.mkdir(parents=True, exist_ok=False)
    summary = load(ROOT/'reduced'/name/'summary.json')
    assert summary['states'] == 501 and summary['endpoint_M'] == 1000
    assert load(ROOT/'evidence/transfers'/(name+'.json'))['all_destination_checksums_verified']
    names = ['ordinary_1p5m_s1', name]
    data = {n: audit.read(audit.dest(n)/'shells.csv') for n in names}
    checked = {n: audit.check_windows(n, data[n]) for n in names}
    native_checks = audit.check_native(case, data[name])
    inputs = [audit.dest(n)/f for n in names for f in ['shells.csv', 'windows.csv', 'health.csv', 'profiles.npz', 'summary.json']]
    metrics = {}
    for n in names:
        windows = audit.read(audit.dest(n)/'windows.csv')
        early = [r for r in windows if r['family'] == 'P' and abs(float(r['start'])-P) < 1e-8]
        late = [r for r in windows if r['family'] == 'P' and abs(float(r['start'])-4*P) < 1e-8]
        shells = data[n]
        health = audit.read(audit.dest(n)/'health.csv')
        values = {}
        for q in [10, 50, 90]:
            a = next(r for r in early if int(r['quantile']) == q)
            b = next(r for r in late if int(r['quantile']) == q)
            values[str(q)] = dict(early={k: float(a[k]) for k in ['half_range_fraction', 'detrended_half_range_fraction', 'mean_shift_fraction', 'slope_fraction_per_M']},
                                 late={k: float(b[k]) for k in ['half_range_fraction', 'detrended_half_range_fraction', 'mean_shift_fraction', 'slope_fraction_per_M']},
                                 raw_growth=float(b['half_range_fraction'])/float(a['half_range_fraction']),
                                 detrended_growth=float(b['detrended_half_range_fraction'])/float(a['detrended_half_range_fraction']))
        with np.load(audit.dest(n)/'profiles.npz') as profiles:
            central = profiles['mass_bins'][:, 0]
            central_ratio_max = float((central/central[0]).max())
        field_checks = load(audit.dest(n)/'summary.json')['field_checks']
        context = dict(COM_initial_M=float(shells[0]['com_mag']), COM_final_M=float(shells[-1]['com_mag']),
                       COM_max_M=max(float(r['com_mag']) for r in shells),
                       central_coordinate_density_max_over_initial=central_ratio_max,
                       core_lapse_min=min(float(r['core_lapse']) for r in health),
                       core_lapse_max=max(float(r['core_lapse']) for r in health),
                       equilibrium_E_L2_final=field_checks[-1]['E_relative_L2_equilibrium'])
        valid = [r for r in health if r['constraint_usable'].lower() == 'true']
        times = [float(r['time']) for r in valid]
        for key in ['H_rms_r10', 'M_rms_r10']:
            values_native = [float(r[key]) for r in valid]
            means = {}
            for label, lo, hi in [('early', P, 2*P), ('late', 4*P, 5*P)]:
                tt = [lo]+[v for v in times if lo < v < hi]+[hi]
                vv = [audit.interp(t, times, values_native) for t in tt]
                means[label] = audit.integral(tt, vv)/(hi-lo)
            context[key] = dict(**means, late_over_early=means['late']/means['early'])
        metrics[n] = dict(shells=values, health=context)
    if case['paired']:
        pairs = audit.read(ROOT/'reduced'/name/'pair_symmetry.csv')
        metrics[name]['pair_symmetry'] = {key: dict(initial=float(pairs[0][key]), final=float(pairs[-1][key]), maximum=max(float(r[key]) for r in pairs))
                                        for key in ['position_pair_sum_rms_M', 'position_pair_sum_max_M', 'covariant_momentum_pair_sum_rms']}
        inputs.append(ROOT/'reduced'/name/'pair_symmetry.csv')
    # The execution commit is passed at submission; do not change the source pin
    # being used by a still-running reduction to describe this separate review.
    atomic(dest/'ARITHMETIC_QA.json', dict(utc=now(), job=os.environ['SLURM_JOB_ID'], execution_source=execution_source,
        reduction_source=summary['source'], passed=True, windows_checked=checked, native_checks=native_checks,
        maximum_absolute_arithmetic_errors=audit.MAX_ERRORS,
        audit_module=entries([Path(audit.__file__)])[0], review_script=entries([Path(__file__)])[0],
        scope='Existing independent scalar window arithmetic and full-N representative native/COM/core-mass/display-cache checks. No stability threshold or visual review claimed.'))
    atomic(dest/'MEASUREMENTS.json', dict(utc=now(), job=os.environ['SLURM_JOB_ID'], execution_source=execution_source,
        period_M=P, early_window_M=[P, 2*P], late_window_M=[4*P, 5*P], cases=metrics, inputs=entries(inputs),
        scope='Provisional comparison of available complete histories; six-million-particle comparison remains pending. Same fixed mesh/one public seed, nonnested samples.'))
    plot(data, case, dest)
    pdf = dest/'shells_com.pdf'
    subprocess.run(['pdftoppm', '-scale-to', '1600', '-singlefile', '-png', str(pdf), str(dest/'shells_com_pdf_preview')], check=True)
    atomic(dest/'PLOTS_RENDERED.json', dict(utc=now(), job=os.environ['SLURM_JOB_ID'], execution_source=execution_source,
        cases=names, files=entries([dest/'shells_com.png', pdf, dest/'shells_com_pdf_preview.png']),
        actual_visual_review_pending=True, all_final_comparisons_and_movies_still_required=True))
    print(json.dumps(metrics, indent=2), flush=True)
    print('READY_CASE_AUDIT_AND_PLOT_OK', name, flush=True)


def plot(data, case, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': .18, 'savefig.dpi': 240})
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, layout='constrained')
    labels = ['Ordinary N = 1,500,000', ('Antipodal' if case['paired'] else 'Ordinary')+f' N = {case["n"]:,}']
    colors = ['#656e76', '#ba7132' if case['paired'] else '#247caa']
    for i, q in enumerate([10, 50, 90]):
        for (_, rows), label, color, style in zip(data.items(), labels, colors, ['-', '--']):
            t = np.array([float(r['time']) for r in rows])
            radius = np.array([float(r[f'r{q}']) for r in rows])
            axes[i, 0].plot(t, radius, label=label, color=color, ls=style, lw=1.4)
            axes[i, 1].plot(t, 100*(radius/radius[0]-1), color=color, ls=style, lw=1.4)
        axes[i, 0].set_ylabel(f'$r_{{{q}}}/M$')
        axes[i, 1].set_ylabel(f'$100(r_{{{q}}}/r_{{{q},0}}-1)$ [%]')
        axes[i, 1].axhline(0, color='#444444', lw=.5)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].set_title('Enclosed rest-mass shell radius about the COM')
    axes[0, 1].set_title('Change relative to each case’s initial radius')
    for ax in axes.ravel():
        ax.set_xlim(0, 1000)
    for ax in axes[-1]:
        ax.set_xlabel('$t/M$')
    for ax in axes[0]:
        ax.secondary_xaxis('top', functions=(lambda t: t/P, lambda p: p*P)).set_xlabel('$t/P_{surface}$')
    fig.suptitle('ST $y_c=0.819$ · proper profile · fixed mesh · stratified seed 1\nProvisional completed-case comparison; N = 6,000,000 still pending', fontsize=14)
    fig.text(.5, -.025, 'Rows enclose 10%, 50% and 90% of rest mass. Radii are coordinate distances; particles may cross these boundaries.', ha='center', fontsize=10)
    fig.savefig(dest/'shells_com.png', bbox_inches='tight')
    fig.savefig(dest/'shells_com.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('index', type=int)
    p.add_argument('execution_source')
    a = p.parse_args()
    review(a.index, a.execution_source)
