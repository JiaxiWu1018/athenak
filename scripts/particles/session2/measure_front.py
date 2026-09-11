#!/usr/bin/env python3
"""Measure the outgoing constraint front, and where it is at each time.

The finite-`N` shot noise in the deposited density does not solve the Hamiltonian
constraint, so the violation radiates outward from the matter.  As the front crosses an
ADM-momentum extraction sphere it lifts that surface integral by orders of magnitude, so
knowing where it is decides which spheres are measuring the momentum and which are not.

Two independent measurements, from two different products:

1. **From the field ledger.**  `<base>.plummer_fields.csv` records the
   coordinate-volume-weighted `|H|` in 64 log-spaced bins of ISOTROPIC radius at every
   history time.  For each time the front is located as the outermost bin in the vacuum
   whose `|H|` exceeds a multiple of its own quiet-period baseline.  This gives a
   continuous track.

2. **From the momentum ledger.**  `<base>.plummer_admmom.csv` gives `|P|` on each
   extraction sphere.  Each sphere's onset time -- the first time it leaves its own quiet
   floor -- is one arrival time, and a straight-line fit of radius against arrival time
   gives the speed directly, at the radii that actually matter.

Agreement between the two is the check.  A speed near the 1+log gauge speed `sqrt(2)` or
the light speed identifies what is propagating.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd


def dedup(df, keys):
    return df.drop_duplicates(subset=keys, keep='last').sort_values('time')


def from_momentum(rundir, ref, factor=50.0, quiet_frac=0.15):
    """Arrival time per sphere: first time |P| exceeds `factor` x its own quiet floor."""
    g = glob.glob(os.path.join(rundir, 'out', '*.plummer_admmom.csv'))
    if not g:
        return []
    d = dedup(pd.read_csv(g[0], comment='#'), ['time', 'R'])
    tmax = d.time.max()
    out = []
    for R, s in d.groupby('R'):
        s = s.sort_values('time')
        quiet = s[(s.time > 0.02 * tmax) & (s.time < quiet_frac * tmax)]
        if len(quiet) < 5:
            continue
        floor = float(np.median(quiet.absP_adm.abs()))
        hit = s[(s.absP_adm.abs() > factor * floor) & (s.time > quiet_frac * tmax)]
        out.append((float(R), float(hit.time.iloc[0]) if len(hit) else np.nan, floor))
    return sorted(out)


def from_fields(rundir, ref, factor=5.0e-8, quiet_frac=0.15):
    """Front radius versus time from the |H| profile, in the vacuum region only."""
    g = glob.glob(os.path.join(rundir, 'out', '*.plummer_fields.csv'))
    if not g:
        return None
    f = dedup(pd.read_csv(g[0], comment='#'), ['time', 'bin'])
    f = f[f.dV > 0].copy()
    f['R'] = np.sqrt(f.r_lo * f.r_hi)
    f['absH'] = f.absH_dV / f.dV
    vac = f[(f.R > 1.2 * ref['R_t']) & (f.R < 0.5 * ref['L'])]
    if not len(vac):
        return None
    # An ABSOLUTE threshold, scaled to the constraint violation in the matter region.
    # A per-bin baseline does not work here: the vacuum |H| is essentially zero in the
    # analytic initial data, so a bin whose baseline is a few times 1e-30 is "lit" by
    # anything at all and the tracker pins to the outermost such bin from the first
    # sample.  The physically meaningful statement is "this vacuum bin now carries a
    # constraint violation comparable to a fixed small fraction of the matter's".
    mat = f[f.R < ref['R_t']]
    href = float(np.median(mat[mat.time < quiet_frac * mat.time.max()].absH)) \
        if len(mat) else 0.0
    if not (href > 0):
        return None
    # `factor` is the threshold as a FRACTION of the matter-region |H|.  The front is a
    # very weak disturbance in absolute terms -- at t = 1 P_1/2 the lit vacuum bins carry
    # |H| ~ 1e-9 to 1e-6 against 1.9e-3 in the matter, i.e. 5e-7 to 5e-4 of it -- so the
    # threshold has to sit well below a part per million to see the edge at all.  That
    # weakness is itself the point: the ADM-momentum surface integral is a near-perfect
    # cancellation and is therefore hypersensitive to a disturbance that barely moves the
    # constraint field.
    thresh = factor * href
    rows = []
    for t, s in vac.groupby('time'):
        s = s.sort_values('R')
        lit = s.R[s.absH > thresh]
        rows.append((float(t), float(lit.max()) if len(lit) else np.nan))
    out = pd.DataFrame(rows, columns=['time', 'R_front']).sort_values('time')
    out.attrs['thresh'] = thresh
    out.attrs['href'] = href
    return out.sort_values('time')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rundir', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--hfrac', type=float, default=5.0e-8,
                    help='|H| threshold as a fraction of the matter-region median')
    a = ap.parse_args()
    ref = json.load(open(a.ref))
    P, Rt = ref['P_half'], ref['R_t']
    print('case %s: R_t = %.6g M, P_1/2 = %.6g M, box half-width %g M'
          % (ref['label'], Rt, P, ref['L']))

    print('\n--- 1. arrival times from the ADM-momentum spheres ---')
    arr = from_momentum(a.rundir, ref)
    print('%12s %14s %12s %14s' % ('R [M]', 'quiet floor', 'arrival t', 'arrival t/P'))
    good = []
    for R, t, fl in arr:
        print('%12.4f %14.3e %12s %14s'
              % (R, fl, ('%.4f' % t) if np.isfinite(t) else 'not yet',
                 ('%.4f' % (t / P)) if np.isfinite(t) else '-'))
        if np.isfinite(t):
            good.append((R, t))
    speed = None
    if len(good) >= 2:
        Rs = np.array([r for r, _ in good])
        ts = np.array([t for _, t in good])
        if len(good) >= 3:
            c = np.polyfit(ts, Rs, 1)
            speed, R0 = float(c[0]), float(c[1])
            print('  straight-line fit R = %.4f + %.4f t  ->  speed %.4f, launch radius '
                  '%.3f M (R_t = %.3f)' % (R0, speed, speed, R0, Rt))
        else:
            speed = float((Rs[-1] - Rs[0]) / (ts[-1] - ts[0]))
            print('  two-point speed %.4f (from R = %.3f and %.3f)'
                  % (speed, Rs[0], Rs[-1]))
        print('  for comparison: light speed 1, 1+log gauge sqrt(2) = %.4f, '
              'sqrt(2/alpha_min) = %.4f' % (np.sqrt(2), np.sqrt(2 / ref['alpha_0'])))

    print('\n--- 2. front radius from the |H| profile ---')
    ff = from_fields(a.rundir, ref, factor=a.hfrac)
    if ff is not None and ff.R_front.notna().any():
        print('  threshold |H| > %.3e (= %.1e of the matter-region median %.3e)'
              % (ff.attrs['thresh'], ff.attrs['thresh'] / ff.attrs['href'],
                 ff.attrs['href']))
        sub = ff[ff.R_front.notna()]
        print('%12s %12s %12s' % ('t', 't/P', 'R_front [M]'))
        for _, r in sub.iloc[::max(1, len(sub) // 12)].iterrows():
            print('%12.4f %12.4f %12.4f' % (r.time, r.time / P, r.R_front))
        # fit only where the front is actually advancing: a tracker pinned to one bin
        # for the whole window is reporting the bin edge, not a speed
        adv = sub[sub.R_front.diff().fillna(1) > 0]
        if len(adv) >= 4 and adv.R_front.nunique() >= 3:
            c = np.polyfit(adv.time, adv.R_front, 1)
            print('  straight-line fit over the %d advancing samples: '
                  'R_front = %.4f + %.4f t  ->  speed %.4f'
                  % (len(adv), c[1], c[0], c[0]))
        else:
            print('  the tracker advanced through only %d distinct bins in this window, '
                  'so no speed is fitted from the field ledger; the momentum-sphere '
                  'arrivals above are the measurement' % sub.R_front.nunique())
    else:
        print('  no bin in the vacuum region is lit above its quiet baseline yet')

    if speed is not None:
        print('\n--- when will the front reach each sphere? ---')
        for R in sorted(ref['pmom_radii']):
            t = (R - Rt) / speed
            print('  R = %10.4f : t = %9.3f M = %6.3f P_1/2  %s'
                  % (R, t, t / P, '(already crossed)' if t / P <= ff.time.max() / P
                     else ''))
    if a.out:
        json.dump({'arrivals': [[r, t] for r, t, _ in arr], 'speed': speed},
                  open(a.out, 'w'), indent=2, default=float)
        print('\nwrote %s' % a.out)


if __name__ == '__main__':
    main()
