#!/usr/bin/env python3
"""The `N`-scaling test: is the measured growth a continuum property or finite-`N`?

**Read this before trusting any number it prints.** The first version of this tool
compared each band's *amplification against its own `t = 0` value*, and that was wrong in
a way that inverted the conclusion.

The denominator is the trap. For the centre-of-mass-referenced bands, `A_1` at `t = 0` is
**not** shot noise: it is dominated by the geometric `(2/3)<1/r>|s|` term that a displaced
reference point manufactures on a centrally peaked profile — Session 1's central
correction. Measured on these runs, the core band's `t = 0` value is `4.98x` its own shot
floor at full `N` and `2.05x` at `N/4`, and it is *larger at full `N`* (`9.69e-03`) than at
`N/4` (`8.00e-03`), which a genuine shot seed cannot be. Dividing by it does not normalise
the growth, it divides by an artifact — and because the artifact scales with the CoM offset
rather than with `sqrt(N)`, the two runs' denominators are wrong by different factors.

The consequence is not subtle. On the same particles and the same physics, the core band's
amplification ratio reads `2.94` ("FASTER at low N") when referenced to the CoM and `0.84`
("SLOWER at low N") when referenced to the origin. The verdict flipped with the choice of
reference point, which no physical result may do.

**The reference-free measure is `A_1(t)` divided by that run's OWN `A_shot`.** Both runs'
modes are seeded by shot noise whose amplitude is exactly `(N/2)^{-1/2}`, so dividing by it
removes the `N` dependence of the SEED and leaves the `N` dependence of the GROWTH — which
is the quantity in question. On these runs it gives `1.178 +/- 0.028` across six bands
against the `2.357 +/- 0.056` of the raw amplitudes, i.e. the growth is `N`-independent to
about 18 %, where a relaxation rate `~1/N` would require a factor of order 2-4.

That measure is now primary; the `t = 0`-referenced amplification is printed alongside it
and explicitly labelled unreliable for CoM bands.

Usage
    compare_nscaling.py --full reduced/R6p5_P2 --low reduced/R6p5_N4_P2 \
                        --ref initial_data/reference_values_R6p5.json --at 2.0
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


# A band must have grown by at least this factor in the FULL-N run before its
# low-N/full-N ratio means anything.  Below it the comparison is noise over noise.
MIN_AMP = 3.0


def load(red, lmax=4):
    m = pd.read_csv(os.path.join(red, 'modes.csv')).drop_duplicates(
        ['time', 'band', 'l'])
    s = pd.read_csv(os.path.join(red, 'scalars.csv')).drop_duplicates(['time'])
    return m, s


def at_time(g, tp, col='A_l'):
    """Value nearest a target t/P, with the actual time it came from."""
    if not len(g):
        return np.nan, np.nan
    i = (g.t_over_P - tp).abs().idxmin()
    return float(g.loc[i, col]), float(g.loc[i, 't_over_P'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', required=True, help='reduction dir of the full-N run')
    ap.add_argument('--low', required=True, help='reduction dir of the reduced-N run')
    ap.add_argument('--ref', required=True)
    ap.add_argument('--at', type=float, default=2.0, help='comparison time in P_1/2')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    ref = json.load(open(a.ref))

    mf, sf = load(a.full)
    ml, sl = load(a.low)
    Nf = int(round(sf.N_alive.iloc[0]))
    Nl = int(round(sl.N_alive.iloc[0]))
    print('case %s, (R/M)_eff = %.4g, comparing at t/P_1/2 = %.3f'
          % (ref['label'], ref['RM_eff'], a.at))
    print('  full-N run : N = %d   (A_shot global = %.6e)' % (Nf, (Nf / 2) ** -0.5))
    print('  low-N  run : N = %d   (A_shot global = %.6e)  -> nulls differ by %.3fx'
          % (Nl, (Nl / 2) ** -0.5, ((Nl / 2) ** -0.5) / ((Nf / 2) ** -0.5)))
    if Nf == Nl:
        print('  WARNING: the two reductions have the SAME N; this is not an N-scaling '
              'comparison.')
    print()

    rows = []
    print('PRIMARY (reference-free): A_1 at the comparison time divided by that run\'s')
    print('OWN A_shot.  Removes the N dependence of the SEED, leaves that of the GROWTH.')
    print()
    print('%-6s %13s %13s %10s %11s %10s' % ('band', 'full A/shot', 'low A/shot',
                                             'low/full', 't0-ref amp', 'verdict'))
    # 'all' first: it is the only band whose t = 0 value is pure shot noise in both runs,
    # so it is the one band where the t = 0-referenced number is also trustworthy.
    for band in ['all', 'com', 'm0', 'm1', 'm2', 'm3']:
        gf = mf[(mf.band == band) & (mf.l == 1)].sort_values('time')
        gl = ml[(ml.band == band) & (ml.l == 1)].sort_values('time')
        if not len(gf) or not len(gl):
            continue
        a0f, _ = at_time(gf, 0.0)
        atf, tf = at_time(gf, a.at)
        a0l, _ = at_time(gl, 0.0)
        atl, tl = at_time(gl, a.at)
        ampf, ampl = atf / a0f, atl / a0l
        amp_ratio_t0 = ampl / ampf
        # reference-free: normalise by each run's own finite-N floor
        shot_f = float(mf[(mf.band == band) & (mf.l == 1)].A_shot.iloc[0])
        shot_l = float(ml[(ml.band == band) & (ml.l == 1)].A_shot.iloc[0])
        ratio = (atl / shot_l) / (atf / shot_f)
        # Refuse a verdict when there is no signal to compare.  Before the mode emerges
        # both runs sit at their own shot floors, amplifications wander over ~0.3-2, and
        # their RATIO is noise over noise -- which still renders as a confident
        # "FASTER at low N" unless the threshold is enforced.  Require the full-N band
        # to have actually grown before reading anything into the comparison.
        if atf / shot_f < MIN_AMP:
            v = 'no signal (full-N is %.2fx its own floor)' % (atf / shot_f)
        elif 0.7 <= ratio <= 1.43:
            v = 'N-INDEPENDENT'
        elif ratio > 1.43:
            v = 'FASTER at low N'
        else:
            v = 'SLOWER at low N'
        rows.append((band, atf / shot_f, atl / shot_l, ratio, v, amp_ratio_t0))
        print('%-6s %13.3f %13.3f %10.3f %11.3f  %s'
              % (band, atf / shot_f, atl / shot_l, ratio, amp_ratio_t0, v))
    print()
    print('  (amplification = A_1 at the comparison time divided by the same run\'s '
          't = 0 value)')
    print('  full-N sampled at t/P = %.4f, low-N at t/P = %.4f' % (tf, tl))
    print()

    # supporting scalars: the disruption measures should scale the same way
    print('%-22s %14s %14s %10s' % ('scalar', 'full-N', 'low-N', 'low/full'))
    # These are per-particle scattering measures.  If the scattering is graininess
    # driven they should scale with the noise amplitude, i.e. as sqrt(N_full/N_low) = 2
    # for a fourfold cut -- and that can be true while the MODE is still collective.
    for col, lab in [('dL_rms', '|dL| rms'), ('sigma_r_u', 'sigma_r'),
                     ('Rcom', 'R_CoM')]:
        vf, _ = at_time(sf[['t_over_P', col]].rename(columns={col: 'A_l'}), a.at)
        vl, _ = at_time(sl[['t_over_P', col]].rename(columns={col: 'A_l'}), a.at)
        print('%-22s %14.4g %14.4g %10.3f' % (lab, vf, vl, vl / max(vf, 1e-300)))
    print()

    # summarise the reference-free ratio over every band that has a signal
    sig = [r for r in rows if r[1] >= MIN_AMP]
    if sig:
        v = np.array([r[3] for r in sig])
        print('REFERENCE-FREE ratio over %d bands with signal: %.3f +/- %.3f'
              % (len(sig), v.mean(), v.std()))
        print('  1.00 = N-independent growth (continuum).  A relaxation rate ~1/N would')
        print('  need a factor of order 2-4 here.')
        print()
    core = [r for r in rows if r[0] == 'm0']
    if core:
        _, ampf, ampl, ratio, _, amp_t0 = core[0]
        print('CONCLUSION on the core band:')
        if ampf < MIN_AMP:
            print('  NO VERDICT. The full-N core band is only %.2fx its own floor, '
                  'which is' % ampf)
            print('  within its own shot fluctuation, so the low-N/full-N ratio is noise '
                  'over noise.')
            print('  Compare at a time where the signal exists: the full-N R6p5 core '
                  'reaches 26.8x')
            print('  at 2 P_1/2 and R10 reaches 86x at 5 P_1/2.')
        elif 0.7 <= ratio <= 1.43:
            print('  The amplification is N-INDEPENDENT (%.3f of the full-N value at '
                  'N/%d).' % (ratio, round(Nf / Nl)))
            print('  This is the signature of a CONTINUUM instability: cutting the '
                  'particle number')
            print('  fourfold did not change the growth, so the growth is not set by '
                  'the graininess.')
        elif ratio > 1.43:
            print('  The amplification is %.2fx FASTER at N/%d.' % (ratio, round(Nf / Nl)))
            print('  That is the signature of FINITE-N RELAXATION, whose rate rises as '
                  'N falls.')
            print('  The Session-2 growth must then be reported as a property of the '
                  'realisation,')
            print('  not of the continuum model.')
        else:
            print('  The amplification is %.2fx SLOWER at N/%d, which neither hypothesis '
                  'predicts.' % (ratio, round(Nf / Nl)))
            print('  Treat as inconclusive and inspect both runs before concluding.')
        print()
        print('  One point is one point: it distinguishes the two hypotheses but does not '
              'measure')
        print('  a scaling law. The homogeneous campaign used an 8x range in N.')

    if a.out:
        json.dump({'N_full': Nf, 'N_low': Nl, 'at': a.at,
                   'bands': {r[0]: {'amp_full': r[1], 'amp_low': r[2],
                                    'ratio': r[3], 'verdict': r[4]} for r in rows}},
                  open(a.out, 'w'), indent=2, default=float)
        print('\nwrote %s' % a.out)


if __name__ == '__main__':
    main()
