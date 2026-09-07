#!/usr/bin/env python3
"""Headline comparison: the dipole per initial-radius band, live vs frozen metric.

usage: fig_live_vs_frozen.py LIVE_REDUCED FROZEN_REDUCED OUTDIR [--period P]

The two runs share the same initial conditions, the same particles, the same grid and the
same analytic metric table. The ONLY difference is that the live run lets the deposited
source feed back into the geometry. Any difference between them in the same band is
therefore attributable to the feedback loop, not to sampling, the pusher, or the metric
discretisation. Each band is normalised by ITS OWN finite-N floor.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plummer_style as ps
from plummer_style import SERIES, REFLINE, MUTED, INK2, label_end
import matplotlib.pyplot as plt

BANDS = [('m0', r'core, $r_0 < 15.6\,M$'),
         ('m1', r'$15.6 < r_0 < 25.2\,M$'),
         ('m2', r'$25.2 < r_0 < 42.2\,M$'),
         ('m3', r'halo, $r_0 > 42.2\,M$')]


def load(path, l=1):
    d = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if int(r['l']) != l:
            continue
        sh = float(r['A_shot'])
        d[r['band']][float(r['t_over_P'])] = float(r['A_l'])/sh if sh > 0 else np.nan
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("live")
    ap.add_argument("frozen")
    ap.add_argument("outdir")
    ap.add_argument("--period", type=float, default=1192.496781)
    a = ap.parse_args()
    ps.apply_style()
    os.makedirs(a.outdir, exist_ok=True)
    L = load(os.path.join(a.live, "modes.csv"))
    F = load(os.path.join(a.frozen, "modes.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4), sharex=True)
    for k, (b, lab) in enumerate(BANDS):
        ax = axes[k//2][k % 2]
        for src, name, col, ls in ((L, "live (feedback on)", SERIES[0], '-'),
                                   (F, "frozen metric", SERIES[1], '-')):
            if b not in src:
                continue
            t = np.array(sorted(src[b]))
            y = np.array([src[b][x] for x in t])
            ax.plot(t, y, ls, color=col, lw=1.9, label=name)
            label_end(ax, t[-1], y[-1], "live" if col == SERIES[0] else "frozen", col)
        ax.axhline(1.0, color=REFLINE, lw=1.3)
        if k == 0:
            ax.annotate("band's own finite-$N$ floor", xy=(0.02, 1.0),
                        xycoords=("axes fraction", "data"), xytext=(0, 5),
                        textcoords="offset points", color=INK2, fontsize=8)
        ax.set_title(lab, fontsize=10, loc="left")
        ax.set_ylim(0, 3.0)
        if k >= 2:
            ax.set_xlabel(r"$t/P_{1/2}$")
        if k % 2 == 0:
            ax.set_ylabel(r"$A_1\,/\,$ band's own $n_{\rm pair}^{-1/2}$")
        if k == 0:
            ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("Dipole by initial-radius band: the only difference between the two runs "
                 "is stress-energy feedback", fontsize=11, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(a.outdir, "live_vs_frozen_bands.png"))
    plt.close(fig)

    # cohort radii: is the cluster in equilibrium?
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), sharey=True)
    for ax, path, name in ((axes[0], a.live, "live (feedback on)"),
                           (axes[1], a.frozen, "frozen metric")):
        d = defaultdict(dict)
        for r in csv.DictReader(open(os.path.join(path, "cohorts.csv"))):
            d[int(r['cohort'])][float(r['t_over_P'])] = (float(r['r_mean']),
                                                         float(r['r0_lo']))
        picks = [0, 8, 16, 24, 31]
        for i, ci in enumerate(picks):
            t = np.array(sorted(d[ci]))
            y = np.array([d[ci][x][0] for x in t])
            if y[0] <= 0:
                continue
            ax.plot(t, y/y[0], lw=1.8, color=SERIES[i % len(SERIES)],
                    label=r"$r_0 \approx %.0f\,M$" % d[ci][t[0]][1])
            label_end(ax, t[-1], y[-1]/y[0], "%.0f" % d[ci][t[0]][1],
                      SERIES[i % len(SERIES)])
        ax.axhline(1.0, color=REFLINE, lw=1.2)
        ax.set_xlabel(r"$t/P_{1/2}$")
        ax.set_title(name, fontsize=10, loc="left")
        ax.set_ylim(0.994, 1.007)
    axes[0].set_ylabel(r"$\langle r\rangle_{\rm cohort}(t)/\langle r\rangle(0)$")
    axes[0].legend(loc="lower left", fontsize=8, ncol=2)
    fig.suptitle("Lagrangian cohort radii: the cluster stays in equilibrium to better "
                 "than 0.5 % in every mass shell", fontsize=11, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(a.outdir, "live_vs_frozen_cohorts.png"))
    plt.close(fig)
    print("wrote", a.outdir)


if __name__ == "__main__":
    main()
