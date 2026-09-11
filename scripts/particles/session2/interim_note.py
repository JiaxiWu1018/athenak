#!/usr/bin/env python3
"""Write the interim status note required after every completed P_1/2.

The specification lists the questions each milestone must answer.  This turns each one
into a measured number with an explicit threshold and a verdict, so the note is a record
rather than an impression, and so a later milestone can be compared with an earlier one
line by line.

Deliberately conservative about growth.  Session 1 withdrew three fitted growth rates
because sliding one-period windows on the same data spanned -0.58 to +1.51 e-folds per
period, and its frozen-metric control produced a -0.85 slope from pure phase mixing.  So
this note reports amplification ratios and window-to-window scatter and explicitly
refuses to fit an exponential; a rate is a claim for the final report to make only if the
record supports a sustained law.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figs_session2 import dedup, read_hst    # noqa: E402

L = []


def w(s=''):
    L.append(s)


def verdict(ok, good, bad):
    return ('**%s**' % good) if ok else ('**%s**' % bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--milestone', type=float, required=True)
    ap.add_argument('--reduced', required=True)
    ap.add_argument('--rundir', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--figdir', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ref = json.load(open(a.ref))
    P = ref['P_half']
    K = a.milestone

    w('# Interim note: case %s, through %g $P_{1/2}$' % (a.case, K))
    w()
    w('Case **%s**: $(R/M)_{\\rm eff} = %.6g$, $b = %.9g\\,M$, $r_t = 20b = %.9g\\,M$, '
      '$P_{1/2} = %.9g\\,M$, $N = %d$, seed %d.'
      % (ref['label'], ref['RM_eff'], ref['b'], ref['rt'], P, ref['Npart'], ref['seed']))
    w('Covers $t = 0$ to $%.9g\\,M$. Figures: `%s`. Reduction: `%s`.'
      % (K * P, os.path.relpath(a.figdir), os.path.relpath(a.reduced)))
    w()
    w('| question | measurement | verdict |')
    w('|---|---|---|')

    # ---- dipole amplitude and localisation ---------------------------------------
    try:
        m = dedup(pd.read_csv(os.path.join(a.reduced, 'modes.csv')),
                  ['time', 'band', 'l'])
        g = m[(m.band == 'com') & (m.l == 1)].sort_values('time')
        a0, amax, alast = g.A_l.iloc[0], g.A_l.max(), g.A_l.iloc[-1]
        w('| Is $A_1$ growing, oscillating, or at its finite-N level? | '
          'global $A_1^{\\rm CoM}$: start %.4e, max %.4e (%.2fx start), '
          'last %.4e (%.2fx start); null %.4e | %s |'
          % (a0, amax, amax / a0, alast, alast / a0, ref['A_shot_global'],
             verdict(amax / a0 < 2.0, 'no global growth beyond 2x',
                     'global amplification %.2fx' % (amax / a0))))
        rows = []
        for q in range(4):
            s = m[(m.band == 'm%d' % q) & (m.l == 1)].sort_values('time')
            if not len(s):
                continue
            rows.append((q, s.A_l.iloc[0], s.A_l.max(), s.A_l.iloc[-1],
                         s.A_shot.iloc[-1]))
        if rows:
            txt = '; '.join('q%d %.2fx (last/null %.2f)'
                            % (q, mx / s0, la / sh) for q, s0, mx, la, sh in rows)
            amp = [mx / s0 for _, s0, mx, _, _ in rows]
            core_led = amp[0] == max(amp)
            w('| Is growth global or confined to the core? | per-quartile '
              'max/start: %s | %s |'
              % (txt, verdict(max(amp) < 2.0,
                              'no band above 2x',
                              'largest in %s (%.2fx)'
                              % ('the CORE band' if core_led else
                                 'band q%d' % rows[int(np.argmax(amp))][0],
                                 max(amp)))))
            # window-to-window scatter, the Session-1 lesson about fitting rates
            s = m[(m.band == 'm0') & (m.l == 1)].sort_values('time')
            if len(s) > 20:
                sl = []
                for k in range(int(K)):
                    win = s[(s.t_over_P >= k) & (s.t_over_P < k + 1)]
                    if len(win) > 5 and win.A_l.min() > 0:
                        c = np.polyfit(win.t_over_P, np.log(win.A_l), 1)[0]
                        sl.append(c)
                if sl:
                    w('| Is a growth LAW resolved? | core-band sliding 1-$P$ log slopes: '
                      '%s (sd %.2f) | %s |'
                      % (', '.join('%+.2f' % v for v in sl), float(np.std(sl)),
                         verdict(len(sl) >= 3 and np.std(sl) < 0.3 and min(sl) > 0,
                                 'consistent sign and small scatter: a rate may be '
                                 'defensible in the final report',
                                 'scatter or sign changes: NO rate is fitted, per the '
                                 'Session-1 lesson')))
    except Exception as e:
        w('| dipole amplitude | FAILED: %s | - |' % e)

    # ---- direction behaviour -----------------------------------------------------
    try:
        d = dedup(pd.read_csv(os.path.join(a.reduced, 'dipoles.csv')), ['time', 'band'])
        piv = {b: gg.set_index('time')[['Dx', 'Dy', 'Dz']]
               for b, gg in d.groupby('band')}
        t = np.array(sorted(set.intersection(*[set(v.index) for v in piv.values()])))
        A, B = piv['m0'].loc[t].to_numpy(), piv['m1'].loc[t].to_numpy()
        c01 = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))
        short = K < 0.5
        w('| Are inner shells aligned or anti-aligned? | '
          r'$\cos(\mathbf{D}_0,\mathbf{D}_1)$: mean %+.3f, last %+.3f, '
          'range [%+.3f, %+.3f] over %.3f $P_{1/2}$ | %s |'
          % (c01.mean(), c01[-1], c01.min(), c01.max(), K,
             'NOT YET MEANINGFUL (span < 0.5 $P_{1/2}$)' if short
             else verdict(abs(c01.mean()) < 0.5, 'no persistent alignment either way',
                          'persistently %s' % ('ALIGNED (translation-like)'
                                               if c01.mean() > 0 else
                                               'ANTI-ALIGNED (sloshing-like)'))))
        u = A / np.linalg.norm(A, axis=1)[:, None]
        mn = u.mean(0)
        mn /= np.linalg.norm(mn)
        ang = np.degrees(np.arccos(np.clip(u @ mn, -1, 1)))
        # Direction behaviour needs a decent fraction of a period to mean anything: over
        # a short span the dipole simply has not had time to turn, so "locked" and
        # "aligned" would be statements about the sampling, not the physics.  Say so
        # rather than letting a short milestone read as a result.
        short = K < 0.5
        w('| Does the dipole direction wander or lock? | core band: mean angle from its '
          'own time-mean direction %.1f deg over %.3f $P_{1/2}$ '
          '(isotropic wander would be 90) | %s |'
          % (ang.mean(), K,
             'NOT YET MEANINGFUL (span < 0.5 $P_{1/2}$)' if short
             else verdict(ang.mean() > 55, 'wandering', 'LOCKING')))
    except Exception as e:
        w('| direction | FAILED: %s | - |' % e)

    # ---- equilibrium, dispersion, CoM --------------------------------------------
    try:
        s = dedup(pd.read_csv(os.path.join(a.reduced, 'scalars.csv')), ['time'])
        dr = [(q, float(s['r_q%02d' % q].iloc[-1] / s['r_q%02d' % q].iloc[0] - 1.0))
              for q in (10, 25, 50, 75, 90) if 'r_q%02d' % q in s]
        worst = max(abs(v) for _, v in dr)
        w('| Is the cluster still in radial equilibrium? | enclosed-rest-mass radii '
          'change from $t=0$: %s | %s |'
          % ('; '.join('%d%% %+.2f%%' % (q, 100 * v) for q, v in dr),
             verdict(worst < 0.05, 'all within 5%',
                     'worst %.1f%% drift' % (100 * worst))))
        w('| Is radial dispersion growing? | $\\sigma_r$: %.4e at $t=0$ -> %.4e now; '
          '$\\sigma_r/\\sigma_t$ now %.4f | %s |'
          % (s.sigma_r_u.iloc[0], s.sigma_r_u.iloc[-1],
             s.sigma_r_u.iloc[-1] / max(s.sigma_t_u.iloc[-1], 1e-300),
             verdict(s.sigma_r_u.iloc[-1] / max(s.sigma_t_u.iloc[-1], 1e-300) < 0.71,
                     'still radially cold relative to tangential',
                     'radial dispersion comparable to tangential')))
        w('| Is the centre of mass moving? | $R_{\\rm CoM}$: %.4e at $t=0$ -> %.4e now '
          '(%.3f of $R_{1/2}$) | %s |'
          % (s.Rcom.iloc[0], s.Rcom.iloc[-1], s.Rcom.iloc[-1] / ref['R_half'],
             verdict(s.Rcom.iloc[-1] < 0.05 * ref['R_half'],
                     'below 5% of $R_{1/2}$', 'CoM has moved appreciably')))
        w('| Orbit invariants | $|dE|$ rms %.2e, $|dL|$ rms %.2e, non-finite %d | %s |'
          % (s.dE_rms.iloc[-1], s.dL_rms.iloc[-1], int(s.n_nonfinite.iloc[-1]),
             verdict(s.n_nonfinite.iloc[-1] == 0, 'no non-finite states',
                     'NON-FINITE STATES PRESENT')))
    except Exception as e:
        w('| equilibrium | FAILED: %s | - |' % e)

    # ---- ADM momentum ------------------------------------------------------------
    try:
        g = glob.glob(os.path.join(a.rundir, 'out', '*.plummer_admmom.csv'))
        mm = dedup(pd.read_csv(g[0], comment='#'), ['time', 'R'])
        radii = sorted(mm.R.unique())
        Rq = radii[-1]
        sq = mm[mm.R == Rq].sort_values('time')
        spread = []
        tf = mm.time.max()
        fin = mm[np.isclose(mm.time, tf)]
        if len(fin) > 1:
            spread = [fin.absP_adm.min(), fin.absP_adm.max()]
        w('| Is ADM linear momentum remaining near zero? | at $R = %.6g$: '
          '$|P|$ max %.3e, last %.3e; across %d radii at the final time '
          '$|P| \\in$ [%.3e, %.3e] | %s |'
          % (Rq, sq.absP_adm.abs().max(), sq.absP_adm.iloc[-1], len(radii),
             spread[0] if spread else float('nan'),
             spread[1] if spread else float('nan'),
             verdict(sq.absP_adm.abs().max() < 1e-4,
                     'consistent with zero (< 1e-4 M)',
                     'NONZERO: %.3e M' % sq.absP_adm.abs().max())))
        w('| Matter-side momenta | $|P^{\\rm matter}|$ last %.3e; '
          '$|P^{\\rm dep}|$ last %.3e | %s |'
          % (sq.absP_matter.iloc[-1],
             float(np.hypot(np.hypot(sq.Px_dep.iloc[-1], sq.Py_dep.iloc[-1]),
                            sq.Pz_dep.iloc[-1])),
             verdict(sq.absP_matter.iloc[-1] < 1e-4, 'at zero', 'nonzero')))
        ar = fin.area_ratio.to_numpy()
        pred = np.array([(1.0 + 0.5 / R) ** 4 for R in fin.R.to_numpy()])
        w('| Is the surface diagnostic itself healthy? | area control '
          '$A/4\\pi R^2$ vs $(1+M/2R)^4$: worst relative deviation %.2e | %s |'
          % (np.max(np.abs(ar / pred - 1.0)),
             verdict(np.max(np.abs(ar / pred - 1.0)) < 1e-3,
                     'quadrature and interpolation sound',
                     'CHECK the extraction spheres')))
    except Exception as e:
        w('| ADM momentum | FAILED: %s | - |' % e)

    # ---- field health and contamination ------------------------------------------
    try:
        hg = [p for p in glob.glob(os.path.join(a.rundir, 'out', '*.user.hst'))
              if '.z4c.' not in p]
        h = read_hst(hg[0])
        w('| Lapse, constraints, particle count | $\\min\\alpha$ %.6f (continuum '
          '%.6f); $\\|H\\|_2$ matter %.3e -> %.3e; $N_{\\rm alive}$ %d/%d; '
          'GR-Boris fallbacks %d | %s |'
          % (h['alpha_min'].iloc[-1], ref['alpha_min'],
             h['Ham_L2_mat'].iloc[0], h['Ham_L2_mat'].iloc[-1],
             int(round(h['N_alive'].iloc[-1])), ref['Npart'],
             int(h['boris_nfai'].iloc[-1]),
             verdict(abs(round(h['N_alive'].iloc[-1]) - ref['Npart']) < 1
                     and h['N_nonfinit'].iloc[-1] == 0,
                     'all particles alive, no non-finite states',
                     'PARTICLE LOSS OR NON-FINITE STATES')))
        tarr = (ref['L'] - ref['R_t']) / np.sqrt(2.0)
        w('| Boundary or refinement contamination? | earliest inward gauge-signal '
          'arrival at the matter edge: %.4g $M$ = %.3f $P_{1/2}$ (conservative '
          '%.3f $P_{1/2}$); now at %.3f $P_{1/2}$ | %s |'
          % (tarr, tarr / P, (ref['L'] - ref['R_t']) / np.sqrt(2.0 / ref['alpha_0']) / P,
             K, verdict(K < (ref['L'] - ref['R_t']) / np.sqrt(2.0 / ref['alpha_0']) / P,
                        'boundary still causally irrelevant',
                        'BOUNDARY MAY NOW BE IN CAUSAL CONTACT')))
    except Exception as e:
        w('| field health | FAILED: %s | - |' % e)

    w()
    w('## Continuation')
    w()
    w('Per the specification, the production chain continues automatically unless there '
      'is a genuine physics, numerics or safety blocker. Review the verdict column above '
      'for any entry in the failure wording; absent one, the next segment proceeds.')
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print('interim_note: wrote %s (%d lines)' % (a.out, len(L)))


if __name__ == '__main__':
    main()
