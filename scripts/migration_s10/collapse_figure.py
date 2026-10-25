"""Allocated, source-backed presentation of an accepted collapse endpoint."""
import math
from common import *
from reduce import csv_rows


def render(root, case_key, destination):
    job()
    root, destination = Path(root), Path(destination)
    acceptance_path = root/'evidence/CASE_NUMERICAL_ACCEPTANCE.json'
    acceptance = load(acceptance_path)
    assert acceptance['numerical_checks_passed']
    onset = acceptance['first_accepted_AH_M']
    if onset is None:
        return None
    red = root/'reduced/yc0500'
    summary = load(red/'summary.json')
    assert summary['complete'] and abs(summary['endpoint']-acceptance['endpoint_M']) < 1e-7
    ah, mass, history = [csv_rows(red/name) for name in ['ah.csv', 'mass.csv', 'history.csv']]
    accepted = [r for r in ah if r['accepted'] == 1]
    converged = [r for r in ah if r['converged'] == 1 and math.isfinite(r['hrel'])]
    warmup = [r for r in converged if r['accepted'] != 1 and r['missing'] == 0
              and r['hrel'] <= .05 and r['rmin'] >= 4*r['dx']
              and 0 < r['streak'] < 3]
    assert len(accepted) == acceptance['accepted_AH_samples'] and accepted
    assert abs(accepted[0]['time']-onset) < 1e-7
    assert all(r['missing'] == 0 and r['streak'] >= 3 and r['hrel'] <= .05
               and r['rmin'] >= 4*r['dx'] for r in accepted)
    m0 = summary['rest_mass_initial']
    assert m0 > 0 and all(abs(r['closure']) < 1e-10 for r in mass)
    endpoint = acceptance['endpoint_M']
    first_removal = acceptance['first_removal_M']
    lo = min(accepted[0]['time'], converged[0]['time'],
             first_removal if first_removal is not None else onset)-1
    hi = endpoint+1
    history = [r for r in history if lo <= r['time'] <= hi]
    assert history
    from publication_style import style
    import matplotlib.pyplot as plt
    style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), layout='constrained')
    blue, orange = '#245d88', '#bd5e28'
    axes[0, 0].plot([r['time'] for r in converged], [r['mirr'] for r in converged],
                    c='.65', lw=1, label='Converged surface')
    axes[0, 0].plot([r['time'] for r in accepted], [r['mirr'] for r in accepted],
                    '.', c=blue, ms=4, label='Persistence accepted')
    if warmup:
        axes[0, 0].plot([r['time'] for r in warmup], [r['mirr'] for r in warmup],
                        'o', mfc='white', mec=orange, ms=4, label='Counter warmup')
    axes[0, 0].set(ylabel='Irreducible horizon mass / M_ref')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot([r['time'] for r in converged], [r['hrel'] for r in converged], c=blue)
    axes[0, 1].axhline(.05, c=orange, ls='--', lw=1, label='Acceptance threshold 0.05')
    axes[0, 1].set(ylabel='Dimensionless horizon residual hrel',
                    ylim=(0, max(.055, max(r['hrel'] for r in converged)*1.1)))
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot([r['time'] for r in mass], [r['m_alive']/m0 for r in mass],
                    c=blue, label='Alive rest mass / initial rest mass')
    axes[1, 0].plot([r['time'] for r in mass], [r['n_lapse']/2000000 for r in mass],
                    c=orange, ls='--', label='Native lapse-excised fraction')
    axes[1, 0].set(ylabel='Fraction of original population / rest mass', ylim=(-.02, 1.04))
    axes[1, 0].legend(fontsize=8)
    for key, color, label in [('H_rms_r10', blue, 'Hamiltonian'),
                               ('M_rms_r10', orange, 'Momentum')]:
        axes[1, 1].semilogy([r['time'] for r in history],
                            [r[key] for r in history], c=color, label=label)
    axes[1, 1].set(ylabel='Proper-volume RMS constraint, r < 10M')
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.axvline(onset, c='.35', ls=':', lw=.8)
        if first_removal is not None:
            ax.axvline(first_removal, c=orange, ls=':', lw=.8)
        ax.set(xlim=(lo, hi), xlabel='t / M_ref')
        ax.grid(alpha=.18)
    fig.suptitle(f'{case_key}: accepted horizon at {onset:g}M; stopped at {endpoint:g}M')
    fig.supxlabel('Dotted lines: first lapse removal (orange), first accepted AH (gray). '
                  'Excision is not escape; finite constraints do not establish convergence.', fontsize=9)
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for suffix in ['png', 'pdf']:
        path = destination/f'collapse_{case_key}.{suffix}'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        files.append(receipt(path))
    plt.close(fig)
    record = dict(utc=now(), job=job(), case=case_key, endpoint_M=endpoint,
        first_accepted_AH_M=onset, converged_samples=len(converged),
        accepted_samples=len(accepted), counter_warmup_samples=len(warmup),
        plots=files, sources=[receipt(p) for p in [acceptance_path,
            red/'ah.csv', red/'mass.csv', red/'history.csv', Path(__file__)]],
        numerical_acceptance_reused=True, actual_visual_review=False,
        scope='Presentation from the accepted endpoint; no evolution or new horizon criterion. '
              'The mass can still grow. Warmup markers include the restart counter reset; '
              'particle excision is not an escaped or unbound mass fraction.')
    atomic(destination/f'COLLAPSE_{case_key}.json', record)
    return record
