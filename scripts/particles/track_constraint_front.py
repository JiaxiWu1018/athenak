#!/usr/bin/env python3
"""Track the outgoing constraint front in the vacuum, and test the boundary caveat.

usage: track_constraint_front.py FIELDS_CSV [--out DIR] [--period P] [--rt 398.999373433]
                                 [--box 4096] [--factor 10]

The initial data is analytically exact in the vacuum, so at t = 0 the RMS |H| there is at
roundoff. Two distinct things then happen, and they must not be confused:

1. An AMBIENT FLOOR appears everywhere at once, within one output interval. That is the
   finite-difference truncation error of the constraint diagnostic evaluated on evolving
   fields; it is not a propagating signal and it does not respect any light cone.
2. A LOCALISED FRONT leaves the density discontinuity at R_t and moves outward, with an
   amplitude one to two orders of magnitude above the ambient floor.

The front is what matters: its reflection off the outer boundary is how boundary
contamination would announce itself. This script measures the ambient floor from the
outermost bins, finds the outermost vacuum bin exceeding `factor` times that floor, and
fits the front speed and launch radius.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plummer_style as ps
from plummer_style import SERIES, REFLINE, INK2, MUTED, label_end
import matplotlib.pyplot as plt

np.seterr(all='ignore')


def read_ledger(path):
    hdr, rows, order = None, {}, []
    for line in open(path):
        if line.startswith('#'):
            continue
        if hdr is None:
            hdr = line.strip().split(',')
            continue
        f = line.strip().split(',')
        k = (f[0], f[2])
        if k not in rows:
            order.append(k)
        rows[k] = f
    return {n: i for i, n in enumerate(hdr)}, np.array([rows[k] for k in order],
                                                       dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fields")
    ap.add_argument("--out", default=None)
    ap.add_argument("--period", type=float, default=1192.496781)
    ap.add_argument("--rt", type=float, default=398.999373433)
    ap.add_argument("--box", type=float, default=4096.0)
    ap.add_argument("--factor", type=float, default=10.0)
    ap.add_argument("--far", type=float, default=3000.0,
                    help="radius above which bins define the ambient floor")
    a = ap.parse_args()
    c, F = read_ledger(a.fields)
    ts = np.unique(F[:, c['time']])

    def prof(t):
        m = F[:, c['time']] == t
        out = []
        for r in F[m]:
            dv = r[c['dV']]
            if dv <= 0:
                continue
            out.append((np.sqrt(r[c['r_lo']]*r[c['r_hi']]),
                        np.sqrt(max(r[c['H2_dV']]/dv, 0.0))))
        return np.array(out)

    rec = []
    for t in ts:
        p = prof(t)
        far = p[p[:, 0] > a.far]
        if len(far) == 0:
            continue
        floor = float(np.median(far[:, 1]))
        vac = p[p[:, 0] > a.rt]
        hit = vac[vac[:, 1] > a.factor*floor]
        rec.append((t, hit[:, 0].max() if len(hit) else np.nan, floor,
                    hit[:, 1][-1] if len(hit) else np.nan))
    rec = np.array(rec)

    good = (~np.isnan(rec[:, 1])) & (rec[:, 0] > 20.0)
    speed = launch = None
    if good.sum() >= 6:
        A = np.vstack([rec[good, 0], np.ones(int(good.sum()))]).T
        coef, *_ = np.linalg.lstsq(A, rec[good, 1], rcond=None)
        resid = rec[good, 1] - A @ coef
        s2 = float(resid @ resid)/max(int(good.sum()) - 2, 1)
        cov = s2*np.linalg.inv(A.T @ A)
        speed, sperr = coef[0], float(np.sqrt(cov[0, 0]))
        launch = coef[1]

    print("ambient far-field floor: t=0 %.3e -> steady %.3e"
          % (rec[0, 2], np.median(rec[rec[:, 0] > 20, 2])))
    if speed is not None:
        print("front speed  = %.4f +/- %.4f c   (1+log gauge speed sqrt(2) = 1.4142)"
              % (speed, sperr))
        print("launch radius = %.1f M   (R_t = %.1f M)" % (launch, a.rt))
        for v, nm in ((1.0, "light speed"), (np.sqrt(2.0), "sqrt(2) gauge speed")):
            print("  one-way R_t -> boundary at %-20s = %7.0f M = %.3f P_1/2"
                  % (nm, (a.box - a.rt)/v, (a.box - a.rt)/v/a.period))

    if a.out:
        ps.apply_style()
        os.makedirs(a.out, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        # Exclude t = 0: there the ambient floor is still at roundoff (1e-18), so a
        # threshold of 10x it is met by roundoff-level values everywhere and the
        # "front" is meaningless. The fit excludes it for the same reason.
        keep = rec[:, 0] > 20.0
        tp = rec[keep, 0]/a.period
        ax.plot(tp, rec[keep, 1], 'o', color=SERIES[0], ms=4.5,
                label="measured front (outermost bin above $%g\\times$ floor)" % a.factor)
        tt = np.linspace(0, tp.max()*1.05, 100)
        ax.plot(tt, a.rt + 1.0*tt*a.period, color=REFLINE, lw=1.6, ls='--',
                label=r"$R_t + c\,t$")
        ax.plot(tt, a.rt + np.sqrt(2.0)*tt*a.period, color=MUTED, lw=1.6, ls=':',
                label=r"$R_t + \sqrt{2}\,c\,t$  (1+log gauge speed)")
        if speed is not None:
            ax.plot(tt, launch + speed*tt*a.period, color=SERIES[1], lw=1.8,
                    label=r"fit: $%.2f\,c$ from $%.0f\,M$" % (speed, launch))
        ax.axhline(a.box, color=SERIES[3], lw=1.4)
        ax.annotate(r"outer boundary $R = %g\,M$" % a.box, xy=(0.55, a.box),
                    xycoords=("axes fraction", "data"), xytext=(0, -13),
                    textcoords="offset points", color=SERIES[3], fontsize=8)
        ax.axhline(a.rt, color=MUTED, lw=1.0, ls=':')
        ax.annotate(r"$R_t$", xy=(0.02, a.rt), xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points", color=MUTED, fontsize=8)
        ax.set_yscale('log')
        ax.set_ylim(a.rt*0.85, a.box*1.6)
        ax.set_xlabel(r"$t/P_{1/2}$")
        ax.set_ylabel(r"isotropic radius $R\ [M]$")
        ax.set_title("Outgoing constraint front and the boundary-contamination schedule")
        ax.legend(loc="lower right", fontsize=8)
        fig.savefig(os.path.join(a.out, "front_position.png"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        for i, frac in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
            t = ts[min(int(frac*(len(ts) - 1)), len(ts) - 1)]
            p = prof(t)
            ax.plot(p[:, 0], np.maximum(p[:, 1], 1e-22), lw=1.6,
                    color=SERIES[i % len(SERIES)],
                    label=r"$t/P_{1/2} = %.3f$" % (t/a.period))
        ax.axvline(a.rt, color=MUTED, lw=1.0, ls=':')
        ax.annotate(r"$R_t$", xy=(a.rt, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(2, -11), textcoords="offset points", color=MUTED, fontsize=8)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r"isotropic radius $R\ [M]$")
        ax.set_ylabel(r"volume-weighted RMS $|H|$")
        ax.set_title("Constraint profile in time: matter interior, edge, and vacuum")
        ax.legend(loc="lower left", fontsize=8)
        fig.savefig(os.path.join(a.out, "front_profiles.png"))
        plt.close(fig)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
