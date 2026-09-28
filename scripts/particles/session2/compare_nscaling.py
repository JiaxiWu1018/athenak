#!/usr/bin/env python3
"""The `N`-scaling test: is the instability a continuum property or finite-`N` relaxation?

Session 2's two production runs both used `N = 2,113,536`, so nothing in them separates a
physical instability from graininess-driven relaxation of a cluster whose only support is
tangential. The measured per-particle rms angular-momentum change is `130 %` (R10) and
`1930 %` (R6p5), which is large enough that the question is real.

The discriminator is how the growth scales with the particle number at fixed everything
else. A continuum instability has a rate set by the model, so the amplification at a given
`t/P_1/2` is `N`-independent. Two-body / graininess relaxation has a rate that falls with
`N` (roughly as `N/ln N` in the classical estimate), so a four-fold cut in `N` should make
the growth visibly faster, not equal.

Read two reductions at the same physical time and compare. Amplitudes are always quoted
against each run's own `t = 0` value and against each band's own finite-`N` null, because
those nulls differ between the runs by construction: `A^shot = n_uniq^{-1/2}` with
`n_uniq = N/2`, so cutting `N` by four raises every null by a factor two. Comparing raw
amplitudes instead of amplification factors would show a difference that is pure shot
noise.

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
    print('%-6s %14s %14s %10s %10s' % ('band', 'full-N amp', 'low-N amp',
                                        'low/full', 'verdict'))
    for band in ['m0', 'm1', 'm2', 'm3', 'com']:
        gf = mf[(mf.band == band) & (mf.l == 1)].sort_values('time')
        gl = ml[(ml.band == band) & (ml.l == 1)].sort_values('time')
        if not len(gf) or not len(gl):
            continue
        a0f, _ = at_time(gf, 0.0)
        atf, tf = at_time(gf, a.at)
        a0l, _ = at_time(gl, 0.0)
        atl, tl = at_time(gl, a.at)
        ampf, ampl = atf / a0f, atl / a0l
        ratio = ampl / ampf
        # A factor-4 cut in N changes a classical relaxation rate by ~4/ln-corrections,
        # so relaxation should show up as a clearly faster amplification at low N.
        v = ('N-INDEPENDENT' if 0.7 <= ratio <= 1.43
             else ('FASTER at low N' if ratio > 1.43 else 'SLOWER at low N'))
        rows.append((band, ampf, ampl, ratio, v))
        print('%-6s %14.3f %14.3f %10.3f  %s' % (band, ampf, ampl, ratio, v))
    print()
    print('  (amplification = A_1 at the comparison time divided by the same run\'s '
          't = 0 value)')
    print('  full-N sampled at t/P = %.4f, low-N at t/P = %.4f' % (tf, tl))
    print()

    # supporting scalars: the disruption measures should scale the same way
    print('%-22s %14s %14s %10s' % ('scalar', 'full-N', 'low-N', 'low/full'))
    for col, lab in [('dL_rms', '|dL| rms'), ('sigma_r_u', 'sigma_r'),
                     ('Rcom', 'R_CoM')]:
        vf, _ = at_time(sf[['t_over_P', col]].rename(columns={col: 'A_l'}), a.at)
        vl, _ = at_time(sl[['t_over_P', col]].rename(columns={col: 'A_l'}), a.at)
        print('%-22s %14.4g %14.4g %10.3f' % (lab, vf, vl, vl / max(vf, 1e-300)))
    print()

    core = [r for r in rows if r[0] == 'm0']
    if core:
        _, ampf, ampl, ratio, _ = core[0]
        print('CONCLUSION on the core band:')
        if 0.7 <= ratio <= 1.43:
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
