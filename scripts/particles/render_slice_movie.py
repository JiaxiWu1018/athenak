#!/usr/bin/env python3
"""Equatorial-slice movie for the Plummer cluster: deposited energy density and |H|.

Every MeshBlock intersecting the view is drawn at its own resolution with its outline, so
the fixed refinement hierarchy is visible and any seam artefact is directly attributable.
Two view scales per quantity: the halo out past R_t, and the core.

Colour: |H| is a magnitude, so it uses a SEQUENTIAL single-hue ramp (magma_r), log-scaled.
The deposited energy density E is also a magnitude and uses the same family; where a
signed field is shown it uses a diverging pair with a neutral midpoint, never a rainbow.

Usage: render_slice_movie.py --dir DIR --case NAME --out DIR [--period P]
                             [--var con|tmunu|z4c] [--half 450] [--fps 12]
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binslice import read_bin_slice

STYLE = {"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
         "savefig.facecolor": "#fcfcfb", "font.size": 9,
         "axes.edgecolor": "#8a8880", "axes.labelcolor": "#0b0b0b",
         "text.color": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"}
R_HALF_ISO = 25.209342804
R_T_ISO = 398.999373433


def panel(ax, fd, key, half, norm, cmap):
    nblk = 0
    for b in range(fd["n_mbs"]):
        g = fd["mb_geometry"][b]
        if g[0] > half or g[1] < -half or g[2] > half or g[3] < -half:
            continue
        arr = fd["mb_data"][key][b][0]
        ax.imshow(np.abs(arr), origin="lower", extent=(g[0], g[1], g[2], g[3]),
                  norm=norm, cmap=cmap, interpolation="nearest", zorder=1)
        ax.add_patch(Rectangle((g[0], g[2]), g[1] - g[0], g[3] - g[2], fill=False,
                               edgecolor="#0b0b0b", lw=0.35, alpha=0.42, zorder=3))
        nblk += 1
    th = np.linspace(0, 2*np.pi, 400)
    for rr, col in ((R_HALF_ISO, "#1baf7a"), (R_T_ISO, "#2a78d6")):
        if rr < half*1.05:
            ax.plot(rr*np.cos(th), rr*np.sin(th), color=col, lw=1.0, ls="--", zorder=4)
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal")
    ax.set_xlabel("$x/M$"); ax.set_ylabel("$y/M$")
    return nblk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory holding the .bin slices")
    ap.add_argument("--case", required=True, help="job basename")
    ap.add_argument("--out", required=True)
    ap.add_argument("--period", type=float, default=1192.496781)
    ap.add_argument("--half-halo", type=float, default=450.0)
    ap.add_argument("--half-core", type=float, default=45.0)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--hmin", type=float, default=1e-12)   # vacuum roundoff ~1e-16
    ap.add_argument("--hmax", type=float, default=2e-3)    # measured central rms 1.5e-3
    ap.add_argument("--emin", type=float, default=1e-13)
    ap.add_argument("--emax", type=float, default=3e-5)    # measured central E 3.0e-5
    a = ap.parse_args()
    plt.rcParams.update(STYLE)
    fr = os.path.join(a.out, "frames_slices")
    os.makedirs(fr, exist_ok=True)

    con = sorted(glob.glob(os.path.join(a.dir, "%s.con.*.bin" % a.case)))
    tmu = sorted(glob.glob(os.path.join(a.dir, "%s.tmunu.*.bin" % a.case)))
    n = min(len(con), len(tmu))
    if n == 0:
        sys.exit("no con/tmunu slices for case %s in %s" % (a.case, a.dir))
    print("%d con / %d tmunu slices, using %d" % (len(con), len(tmu), n), flush=True)
    nH = LogNorm(vmin=a.hmin, vmax=a.hmax)
    nE = LogNorm(vmin=a.emin, vmax=a.emax)

    # A restart that lands exactly on tlim evolves nothing but still writes one output set
    # before exiting. Its particle data is the restart's and is correct -- the two endpoint
    # pvtk frames of this campaign are bit-identical -- but the CONSTRAINT arrays are never
    # computed in that process, so the con dump is identically zero. Plotted on a log norm
    # that renders as an empty panel, which reads as "the constraints vanished". Skip any
    # such frame and say so, rather than shipping a movie whose last frame is a lie.
    kept = 0
    for k in range(n):
        fH = read_bin_slice(con[k])
        fT = read_bin_slice(tmu[k])
        t = fH["time"]
        if not any(np.any(np.asarray(mb)) for mb in fH["mb_data"]["con_H"]):
            print("  SKIP frame %d (%s): con_H is identically zero at t=%.6f -- a no-op "
                  "restart's uninitialised dump, not a physical state"
                  % (k, os.path.basename(con[k]), t), flush=True)
            continue
        fig, axes = plt.subplots(2, 2, figsize=(11.4, 10.4))
        bH1 = panel(axes[0, 0], fH, "con_H", a.half_halo, nH, "magma_r")
        bH2 = panel(axes[0, 1], fH, "con_H", a.half_core, nH, "magma_r")
        bE1 = panel(axes[1, 0], fT, "tmunu_E", a.half_halo, nE, "magma_r")
        bE2 = panel(axes[1, 1], fT, "tmunu_E", a.half_core, nE, "magma_r")
        axes[0, 0].set_title(r"$|H|$, halo $\pm%g\,M$  (%d blocks)"
                             % (a.half_halo, bH1), fontsize=9, loc="left")
        axes[0, 1].set_title(r"$|H|$, core $\pm%g\,M$  (%d blocks)"
                             % (a.half_core, bH2), fontsize=9, loc="left")
        axes[1, 0].set_title(r"deposited $E$, halo $\pm%g\,M$  (%d blocks)"
                             % (a.half_halo, bE1), fontsize=9, loc="left")
        axes[1, 1].set_title(r"deposited $E$, core $\pm%g\,M$  (%d blocks)"
                             % (a.half_core, bE2), fontsize=9, loc="left")
        smH = plt.cm.ScalarMappable(norm=nH, cmap="magma_r")
        cbH = fig.colorbar(smH, ax=axes[0, :], fraction=0.030, pad=0.02, extend="both")
        cbH.set_label(r"$|H|$  (Hamiltonian constraint)", fontsize=8)
        cbH.ax.tick_params(labelsize=7)
        smE = plt.cm.ScalarMappable(norm=nE, cmap="magma_r")
        cbE = fig.colorbar(smE, ax=axes[1, :], fraction=0.030, pad=0.02, extend="both")
        cbE.set_label(r"deposited energy density $E$", fontsize=8)
        cbE.ax.tick_params(labelsize=7)
        fig.suptitle("Plummer cluster, equatorial $z=0$ slice, fixed 7-level refinement "
                     "(block outlines drawn)    $t/P_{1/2}$ = %.4f   ($t/M$ = %.2f)"
                     % (t/a.period, t), fontsize=10, x=0.008, ha="left")
        fig.savefig(os.path.join(fr, "f%04d.png" % kept), dpi=120,
                    bbox_inches="tight", facecolor=STYLE["figure.facecolor"])
        plt.close(fig)
        kept += 1
        if k % 20 == 0 or k == n - 1:
            print("  frame %4d  t/P=%.4f" % (k, t/a.period), flush=True)

    mp4 = os.path.join(a.out, "slices_H_and_E_equatorial.mp4")
    rc = os.system("ffmpeg -y -loglevel error -framerate %d -i %s/f%%04d.png "
                   "-c:v libx264 -pix_fmt yuv420p "
                   "-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' %s" % (a.fps, fr, mp4))
    if rc == 0 and os.path.exists(mp4):
        print("wrote %s (%.1f MB)" % (mp4, os.path.getsize(mp4)/1e6))
    else:
        print("ffmpeg failed (rc=%d); frames are in %s" % (rc, fr))


if __name__ == "__main__":
    main()
