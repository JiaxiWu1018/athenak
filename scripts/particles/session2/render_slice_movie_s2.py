#!/usr/bin/env python3
"""Movie B: equatorial density and Hamiltonian-constraint slices on a FIXED UNIFORM grid.

The Session-2 specification requires that the density visualisation not be an
AMR-patch-dependent image whose pixel size changes with refinement.  Session 1's
render_slice_movie.py drew each MeshBlock at its own resolution with an outline, so the
apparent texture of the field changed across every seam and could change in time.  This
renderer instead reads AthenaK's `cart` output, which the code itself interpolates onto a
grid of fixed physical extent and fixed pixel size, and asserts frame-to-frame that the
geometry never changed.

Two further properties matter for honesty and are enforced here:

* **The colour scale is fixed across the whole movie**, computed once from a reference
  frame (default: the first), so brightness change on screen is field change, not
  autoscaling.  Same for the axes limits.  This is the "do not let changing plotting
  bounds create fake motion" requirement.
* **Density is shown on a logarithmic scale with an explicit floor**, and the floor is
  drawn in the colour bar.  The deposited density is a CIC estimate from a finite particle
  number, so it carries real cell-level shot noise and the eighth-order interpolation onto
  the visualisation grid can ring slightly negative in the low-density outskirts; clipping
  at a stated floor is the honest presentation, and silently plotting `log` of a
  partly-negative field is not.
* The Hamiltonian constraint is signed, so it uses a **diverging** scale with a neutral
  midpoint at zero and a symmetric range -- never a rainbow.

Usage
    render_slice_movie_s2.py --rundir DIR --ref initial_data/reference_values_X.json
                             --panel core|wide --out movies/X/slices_core.mp4 [--fps 12]
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402
import s2_style as st                                # noqa: E402
from cart_reader import read_cart, equatorial        # noqa: E402


def series(cartdir, file_id):
    f = glob.glob(os.path.join(cartdir, '*.%s.*.bin' % file_id))
    return sorted(f, key=lambda p: int(re.search(r'\.(\d+)\.bin$', p).group(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rundir', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--panel', choices=['core', 'wide'], default='core')
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=int, default=12)
    ap.add_argument('--floor-decades', type=float, default=6.0,
                    help='density colour range, in decades below the reference maximum')
    a = ap.parse_args()
    st.apply()
    ref = st.load_ref(a.ref)
    P = ref['P_half']

    cart = os.path.join(a.rundir, 'out', 'cart')
    dens = series(cart, 'dens_%s' % a.panel)
    ham = series(cart, 'ham_%s' % a.panel)
    if not dens:
        sys.exit('no cart frames for dens_%s under %s' % (a.panel, cart))
    n = min(len(dens), len(ham)) if ham else len(dens)
    print('%d density frames, %d constraint frames -> using %d' % (len(dens), len(ham), n))

    # --- the fixed-grid guarantee, checked rather than assumed ---------------------
    geo = None
    for p in dens[:n]:
        r = read_cart(p)
        k = (tuple(np.round(r['center'], 10)), tuple(np.round(r['extent'], 10)),
             tuple(r['numpoints']))
        if geo is None:
            geo = k
        elif k != geo:
            sys.exit('%s: the cart grid CHANGED between frames (%r -> %r); the movie '
                     'would not be comparable frame to frame' % (p, geo, k))
    print('grid is fixed across all frames: center %s extent %s numpoints %s'
          % geo)

    # --- one reference frame fixes both colour scales for the whole movie ----------
    r0 = read_cart(dens[0])
    d0, x, y = equatorial(r0, 'tmunu_E')
    dmax = float(np.nanmax(d0))
    dfloor = dmax * 10.0 ** (-a.floor_decades)
    hmaxs = []
    for p in (ham[:n] if ham else []):
        hh, _, _ = equatorial(read_cart(p), 'con_H')
        hmaxs.append(float(np.nanpercentile(np.abs(hh), 99.5)))
    hlim = max(hmaxs) if hmaxs else 1.0
    ext = [x[0], x[-1], y[0], y[-1]]
    pix = float(r0['dx'][0])
    print('density colour range [%.3e, %.3e] (%.1f decades, fixed); '
          'constraint range +/-%.3e (fixed); pixel %.6g M = %.3g dx_fine'
          % (dfloor, dmax, a.floor_decades, hlim, pix, pix / ref['dx_fine']))

    tmp = tempfile.mkdtemp(prefix='s2slice_')
    try:
        for i in range(n):
            rd = read_cart(dens[i])
            dd, _, _ = equatorial(rd, 'tmunu_E')
            fig, axes = plt.subplots(1, 2 if ham else 1, figsize=(10.2, 4.6),
                                     squeeze=False)
            ax = axes[0][0]
            im = ax.imshow(np.clip(dd, dfloor, None), origin='lower', extent=ext,
                           cmap='magma', norm=LogNorm(vmin=dfloor, vmax=dmax),
                           interpolation='nearest')
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(r'deposited $E$ (clipped at $10^{-%g}$ of max)'
                         % a.floor_decades, fontsize=8)
            ax.set_title('Deposited energy density', fontsize=9.5, loc='left')
            if ham:
                rh = read_cart(ham[i])
                hh, _, _ = equatorial(rh, 'con_H')
                ax2 = axes[0][1]
                im2 = ax2.imshow(hh, origin='lower', extent=ext, cmap='RdBu_r',
                                 norm=TwoSlopeNorm(vmin=-hlim, vcenter=0.0, vmax=hlim),
                                 interpolation='nearest')
                cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
                cb2.set_label(r'Hamiltonian constraint $H$', fontsize=8)
                ax2.set_title('Hamiltonian constraint', fontsize=9.5, loc='left')
            th = np.linspace(0, 2 * np.pi, 400)
            for axx in axes[0]:
                for rr, lab in ((ref['R_half'], r'$R_{1/2}$'), (ref['R_t'], r'$R_t$')):
                    if rr < 0.98 * ext[1]:
                        axx.plot(rr * np.cos(th), rr * np.sin(th), color='#bfbfbf',
                                 lw=0.9, ls='--')
                        axx.annotate(lab, xy=(0, rr), color='#bfbfbf', fontsize=7.5,
                                     ha='center', va='bottom')
                axx.set_xlabel('$x/M$')
                axx.set_ylabel('$y/M$')
                axx.set_aspect('equal')
                axx.set_xlim(ext[0], ext[1])
                axx.set_ylim(ext[2], ext[3])
                axx.grid(False)
            fig.suptitle(r'%s:  $(R/M)_{\rm eff} = %.4g$,  $t/P_{1/2} = %.4f$   '
                         r'[fixed uniform %d$\times$%d grid, pixel %.5g $M$ '
                         r'= 1 finest cell]'
                         % (ref['label'], ref['RM_eff'], rd['time'] / P,
                            r0['numpoints'][0], r0['numpoints'][1], pix),
                         fontsize=10)
            fig.savefig(os.path.join(tmp, 'f%04d.png' % i), dpi=120,
                        bbox_inches='tight')
            plt.close(fig)
            if i % 25 == 0:
                print('  frame %d/%d  t/P = %.4f' % (i, n, rd['time'] / P), flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(a.fps),
               '-i', os.path.join(tmp, 'f%04d.png'), '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', a.out]
        rc = subprocess.call(cmd)
        if rc == 0 and os.path.exists(a.out):
            print('wrote %s (%.1f MB, %d frames at %d fps)'
                  % (a.out, os.path.getsize(a.out) / 1e6, n, a.fps))
        else:
            print('ffmpeg failed (rc=%d); frames left in %s' % (rc, tmp))
            return 1
    finally:
        if os.path.isdir(tmp) and os.path.exists(a.out):
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
