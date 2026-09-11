#!/usr/bin/env python3
"""Particle movie for the Plummer cluster: two views per frame, coloured by initial group.

Left panel  = HALO view, the whole matter region out to just beyond R_t.
Right panel = CORE view, the inner few Plummer scale lengths.
Both are equatorial slabs plus a projected 3-D scatter, and both are drawn at a FIXED
scale for every frame so apparent expansion or contraction is physical.

Colour is a SEQUENTIAL single-hue-family map (cividis) because the quantity -- initial
radial group -- is ordered. A rainbow would invent structure. cividis is perceptually
uniform and colour-vision-deficiency safe. Colour identifies the initial group only; the
frame is annotated with t/P so time is never carried by colour.

Usage: render_particle_movie.py --npz F --out DIR [--period 1192.496781] [--fps 12]
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "font.size": 9,
    "axes.edgecolor": "#8a8880", "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "legend.frameon": False,
}
# Session 2: the model and the refinement seams come from the per-case reference JSON,
# never from module constants.  Session 1's values (R_1/2 = 25.209, R_t = 398.999, seams
# at 64/128/256) belong to b = 20 M and would place every annotation in the wrong place
# for a compactness scan; the Session-2 seams are 10..640 M (R10) and 6..384 M (R6p5).
R_HALF_ISO = None
R_T_ISO = None
SEAMS = []


def set_model(ref_path):
    """Load R_1/2, R_t and the refinement seams for this case."""
    global R_HALF_ISO, R_T_ISO, SEAMS
    import json
    r = json.load(open(ref_path))
    R_HALF_ISO = r['R_half']
    R_T_ISO = r['R_t']
    SEAMS = [float(h) for h in r['halfwidth'][1:]]
    return r


def panel(ax, x, y, c, half, cmax, slab_z, z, title):
    m = np.abs(z) < slab_z
    ax.scatter(x[m], y[m], c=c[m], s=0.35, cmap="cividis", vmin=0, vmax=cmax,
               linewidths=0, rasterized=True)
    th = np.linspace(0, 2*np.pi, 400)
    for rr, lab, col in ((R_HALF_ISO, r"$R_{1/2}$", "#e34948"),
                         (R_T_ISO, r"$R_t$", "#2a78d6")):
        if rr < half*1.05:
            ax.plot(rr*np.cos(th), rr*np.sin(th), color=col, lw=1.0, ls="--", alpha=0.85)
            ax.annotate(lab, xy=(0, rr), color=col, fontsize=8, ha="center",
                        va="bottom")
    for s in SEAMS:
        if s < half:
            ax.plot([-s, s, s, -s, -s], [-s, -s, s, s, -s], color="#c9c8c1", lw=0.6,
                    ls=":")
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal")
    ax.set_xlabel("$x/M$"); ax.set_ylabel("$y/M$")
    ax.set_title(title + r"   ($|z| < %g\,M$, %d shown)" % (slab_z, int(m.sum())),
                 fontsize=9, loc="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", required=True,
                    help="initial_data/reference_values_<case>.json")
    ap.add_argument("--period", type=float, default=None,
                    help="P_1/2; defaults to the value in --ref")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--half-halo", type=float, default=450.0)
    ap.add_argument("--half-core", type=float, default=45.0)
    a = ap.parse_args()
    ref = set_model(a.ref)
    if a.period is None:
        a.period = ref['P_half']
    plt.rcParams.update(STYLE)
    fr = os.path.join(a.out, "frames_particles")
    os.makedirs(fr, exist_ok=True)

    d = np.load(a.npz, allow_pickle=True)
    X, Y, Z, C, T = d["x"], d["y"], d["z"], d["cohort"], d["times"]
    ncoh = int(d["ncohort"]) if "ncohort" in d else int(C.max()) + 1
    nf = X.shape[0]
    print("%d frames x %d particles, %d cohorts" % (nf, X.shape[1], ncoh), flush=True)

    for k in range(nf):
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4))
        panel(axes[0], X[k], Y[k], C, a.half_halo, ncoh - 1,
              0.06*a.half_halo, Z[k], "halo")
        panel(axes[1], X[k], Y[k], C, a.half_core, ncoh - 1,
              0.10*a.half_core, Z[k], "core")
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, ncoh - 1), cmap="cividis")
        cb = fig.colorbar(sm, ax=axes, fraction=0.024, pad=0.02)
        cb.set_label("initial radial group (inner $\\to$ outer)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        fig.suptitle("Plummer cluster particles, equatorial slabs, coloured by INITIAL "
                     "radial group    $t/P_{1/2}$ = %.4f   ($t/M$ = %.2f)"
                     % (T[k]/a.period, T[k]), fontsize=10, x=0.008, ha="left")
        fig.savefig(os.path.join(fr, "f%04d.png" % k), dpi=125,
                    bbox_inches="tight", facecolor=STYLE["figure.facecolor"])
        plt.close(fig)
        if k % 20 == 0 or k == nf - 1:
            print("  frame %4d  t/P=%.4f" % (k, T[k]/a.period), flush=True)

    mp4 = os.path.join(a.out, "particles_core_and_halo.mp4")
    rc = os.system("ffmpeg -y -loglevel error -framerate %d -i %s/f%%04d.png "
                   "-c:v libx264 -pix_fmt yuv420p "
                   "-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' %s" % (a.fps, fr, mp4))
    if rc == 0 and os.path.exists(mp4):
        print("wrote %s (%.1f MB)" % (mp4, os.path.getsize(mp4)/1e6))
    else:
        print("ffmpeg failed (rc=%d); frames are in %s" % (rc, fr))


if __name__ == "__main__":
    main()
