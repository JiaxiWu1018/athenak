#!/usr/bin/env python3
"""Verify that every ADM-momentum extraction sphere lies wholly within one refinement level.

The refined regions of this mesh family are CUBES |x_i| <= H_l with H_l = L/2^l.  On the
sphere of coordinate radius R the quantity max_i |n_i| ranges over [1/sqrt(3), 1], so the
sphere intersects the cube boundary exactly when H_l < R <= sqrt(3) H_l.  A sphere that
straddles a seam samples two different cell sizes around one integration path, and the
coarse side carries prolongation error, so the specification asks us to avoid it.

This script measures the actual fraction of solid angle on each level, by evaluating the
level occupancy at the SAME Gauss-Legendre nodes and with the SAME weights the in-code
diagnostic uses (ntheta nodes in cos(theta) x 2 ntheta uniform in phi), and reports the
worst-case mixing.  A pass means each sphere's solid angle is 100 % on one level.
"""
import argparse
import json
import sys

import numpy as np


def gl_nodes(ntheta):
    x, w = np.polynomial.legendre.leggauss(ntheta)
    th = np.arccos(x)
    ph = 2.0 * np.pi / (2 * ntheta) * np.arange(2 * ntheta)
    TH, PH = np.meshgrid(th, ph, indexing='xy')
    W = np.tile(w * np.pi / ntheta, (2 * ntheta, 1))
    return TH.ravel(), PH.ravel(), W.ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True, help='initial_data/reference_values_<case>.json')
    ap.add_argument('--ntheta', type=int, default=32)
    a = ap.parse_args()
    c = json.load(open(a.ref))
    hw = c['halfwidth']
    dx = c['dx']
    N = c['N']
    th, ph, w = gl_nodes(a.ntheta)
    n = np.stack([np.cos(ph) * np.sin(th), np.sin(ph) * np.sin(th), np.cos(th)])
    print('case %s: L = %g, levels %d, R_t = %.6g, sum(w) = %.12g (4 pi = %.12g)'
          % (c['label'], c['L'], N, c['R_t'], w.sum(), 4 * np.pi))
    print('  each sphere must be 100 %% on ONE level; the seam-free band per level is')
    print('  R/H in (sqrt(3)/2, 1] = (0.8660, 1]')
    bad = 0
    for k, R in enumerate(c['pmom_radii']):
        mx = R * np.abs(n).max(axis=0)            # max_i |x_i| at each node
        # deepest level whose cube contains the node
        lev = np.zeros(mx.size, dtype=int)
        for l in range(1, N + 1):
            lev = np.where(mx <= hw[l], l, lev)
        frac = {}
        for l in sorted(set(lev.tolist())):
            frac[l] = float(w[lev == l].sum() / w.sum())
        levs = sorted(frac, key=lambda l: -frac[l])
        mixed = len(frac) > 1
        if mixed or R <= c['R_t']:
            bad += 1
        print('  r%d  R = %10.6f  R/R_t = %5.2f  ->  %s   %s'
              % (k + 1, R, R / c['R_t'],
                 '  '.join('level %d (dx %g): %6.2f %%' % (l, dx[l], 100 * frac[l])
                           for l in levs),
                 'MIXED' if mixed else ('IN MATTER' if R <= c['R_t'] else 'clean')))
    print('  VERDICT: %s (%d of %d spheres are mixed or inside the matter)'
          % ('PASS' if bad == 0 else 'FAIL', bad, len(c['pmom_radii'])))
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
