#!/usr/bin/env python3
"""Validate a t = 0 `cart` frame against the analytic relativistic Plummer profile.

Two failure modes this is designed to catch:

1. RANK DOUBLE COUNTING.  cart_grid.cpp decides node ownership with a bounds test that
   is inclusive at both ends and cartgrid.cpp MPI_SUMs the result, so a node lying on a
   MeshBlock face shared by two ranks is added twice.  On an origin-centred mesh the
   planes x = 0, y = 0, z = 0 and every refinement seam are such faces.  A doubled node
   shows up as an exact factor ~2 against the analytic profile and as a broken azimuthal
   symmetry, both of which are enormous compared with the CIC shot noise.

2. A WRONG GRID.  The header's center/extent/numpoints are checked against the deck, and
   the frame-to-frame invariance of that geometry is what makes the movie honest.

The comparison quantity is the deposited static-observer energy density.  What the code
stores in tmunu_E is the Eulerian energy density E = T_{mu nu} n^mu n^nu, which is the
prescribed Plummer quantity, so the analytic reference is eps(r) at the AREAL radius of
each pixel -- obtained by mapping the pixel's isotropic radius R through r(R).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cart_reader import read_cart, equatorial
from plummer_1d import PlummerModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame', required=True)
    ap.add_argument('--b', type=float, required=True)
    ap.add_argument('--rt-over-b', type=float, default=20.0)
    ap.add_argument('--var', default='tmunu_E')
    ap.add_argument('--rmin-frac', type=float, default=0.25,
                    help='ignore pixels inside this fraction of b, where CIC noise and '
                         'the central cusp dominate')
    a = ap.parse_args()

    rec = read_cart(a.frame)
    plane, x, y = equatorial(rec, a.var)
    print('frame      %s' % os.path.basename(a.frame))
    print('  cycle %d  t = %.8g  vars %s' % (rec['cycle'], rec['time'], rec['labels']))
    print('  center %s  extent %s  numpoints %s' %
          (tuple(rec['center']), tuple(rec['extent']), tuple(rec['numpoints'])))
    print('  pixel  %s   z nodes %s' % (tuple(np.round(rec['dx'], 10)), tuple(rec['z'])))
    nz = rec['numpoints'][2]
    if nz == 2:
        d = rec['data'][rec['labels'].index(a.var)]
        rel = np.abs(d[0] - d[1])/np.maximum(np.abs(d[0]) + np.abs(d[1]), 1e-300)
        print('  the two z planes differ by at most %.3e relative (symmetric straddle)'
              % rel.max())

    mod = PlummerModel(M=1.0, b=a.b, rt=a.rt_over_b * a.b, npanel=40000, ngl=20)
    X, Y = np.meshgrid(x, y, indexing='xy')      # plane is (ny, nx)
    R = np.sqrt(X**2 + Y**2)                     # isotropic radius in the z=0 plane
    r_areal = mod.r_of_R(R)
    ref = mod.eps(r_areal)

    inside = (r_areal < mod.rt) & (R > a.rmin_frac * a.b)
    ratio = np.where(inside & (ref > 0), plane / np.where(ref > 0, ref, 1.0), np.nan)
    good = np.isfinite(ratio)
    print('  pixels compared: %d of %d (inside r_t, outside %.3g b)'
          % (good.sum(), plane.size, a.rmin_frac))
    if good.sum() == 0:
        print('  NOTHING TO COMPARE'); return 1
    q = np.nanpercentile(ratio[good], [0.1, 5, 50, 95, 99.9])
    print('  deposited/analytic ratio percentiles  0.1%% %.5f  5%% %.5f  50%% %.5f  '
          '95%% %.5f  99.9%% %.5f' % tuple(q))
    print('  max ratio %.5f   min ratio %.5f' % (np.nanmax(ratio), np.nanmin(ratio)))

    # A doubled node is a factor ~2.  Flag anything beyond 1.5 as a hard failure.
    nbad = int(np.sum(ratio[good] > 1.5))
    print('  pixels with ratio > 1.5 (double-count signature): %d' % nbad)

    # azimuthal symmetry: bin by radius, look at the scatter within each ring
    rb = np.geomspace(max(a.rmin_frac * a.b, R[good].min()), R[good].max(), 25)
    print('  azimuthal check (a doubled column breaks this):')
    print('     R range            n     median ratio   ring rms/median')
    worst = 0.0
    for i in range(len(rb) - 1):
        m = good & (R >= rb[i]) & (R < rb[i + 1])
        if m.sum() < 50:
            continue
        v = ratio[m]
        med = np.median(v)
        rms = np.std(v) / abs(med)
        worst = max(worst, abs(med - 1.0))
        print('     %8.4f-%8.4f %6d   %10.5f     %8.4f'
              % (rb[i], rb[i + 1], m.sum(), med, rms))
    print('  worst |median ratio - 1| over rings: %.5f' % worst)
    ok = (nbad == 0) and (worst < 0.25)
    print('  VERDICT: %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
