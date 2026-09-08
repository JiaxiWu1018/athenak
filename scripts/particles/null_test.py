#!/usr/bin/env python3
"""Test a live band series against a frozen-metric control covering the same window.

The two runs share their initial data exactly and differ only in whether the particle
stress-energy feeds back into the geometry, so the frozen run is a same-realisation null
rather than an analytic one.  That matters here because the nominal Poisson floor is NOT a
usable reference for the inner bands: centre-of-mass subtraction on a cusped profile leaves
the core band at ~3.9 times that floor at t = 0, in both runs.

Three statistics, none of which assumes a growth law:

  exceedance   fraction of frozen samples in the window that exceed the live endpoint,
               and the same for block maxima -- a direct, model-free p-value
  block means  live vs frozen mean over each half-period block, so a trend that is
               spread over the window is not hidden by the endpoint's own noise
  divergence   |live - frozen| against the frozen series' own scatter, using the fact
               that the two runs start identical

Effective sample sizes come from the integrated autocorrelation time of each series, since
a 0.01 P output cadence massively oversamples a process with a ~0.5 P correlation time.

Usage:
  null_test.py --live reduced/prod_com --frozen reduced/frozen_full_com
               [--band m0] [--l 1] [--lo 1.0] [--hi 3.0]
"""
import argparse
import csv
import os
import sys

import numpy as np


def series(d, band, l):
    acc = {}
    with open(os.path.join(d, "modes.csv")) as f:
        for r in csv.DictReader(f):
            if r["band"] == band and int(r["l"]) == l:
                acc[float(r["t_over_P"])] = float(r["A_l"])/float(r["A_shot"])
    t = np.array(sorted(acc))
    return t, np.array([acc[x] for x in t])


def tau_int(y):
    """Integrated autocorrelation time in samples (Sokal, truncated at the first zero)."""
    y = y - y.mean()
    c0 = float((y*y).mean())
    if c0 <= 0:
        return 1.0
    t = 0.5
    for k in range(1, len(y)//4):
        c = float((y[:-k]*y[k:]).mean())/c0
        if c <= 0:
            break
        t += c
    return 2.0*t


def block_stats(t, y, width):
    out = []
    lo = t.min()
    while lo < t.max() - 1e-9:
        m = (t >= lo) & (t < lo + width)
        if m.any():
            out.append((lo, y[m].mean(), y[m].max()))
        lo += width
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--band", default="m0")
    ap.add_argument("--l", type=int, default=1)
    ap.add_argument("--lo", type=float, default=0.0)
    ap.add_argument("--hi", type=float, default=3.0)
    ap.add_argument("--block", type=float, default=0.5)
    a = ap.parse_args()

    tl, yl = series(a.live, a.band, a.l)
    tf, yf = series(a.frozen, a.band, a.l)
    ml = (tl >= a.lo) & (tl <= a.hi)
    mf = (tf >= a.lo) & (tf <= a.hi)
    tl, yl, tf, yf = tl[ml], yl[ml], tf[mf], yf[mf]
    if tf.size == 0:
        sys.exit("frozen control does not cover [%g, %g]" % (a.lo, a.hi))

    dt = float(np.median(np.diff(tl)))
    print("band %s, l = %d, window t/P in [%.3f, %.3f]" % (a.band, a.l, a.lo, a.hi))
    print("  live   %4d samples, tau_int %.3f P, n_eff %5.1f" %
          (yl.size, tau_int(yl)*dt, (tl.max()-tl.min())/max(tau_int(yl)*dt, 1e-9)))
    print("  frozen %4d samples, tau_int %.3f P, n_eff %5.1f  (covers %.3f-%.3f P)" %
          (yf.size, tau_int(yf)*dt, (tf.max()-tf.min())/max(tau_int(yf)*dt, 1e-9),
           tf.min(), tf.max()))
    if tf.max() < a.hi - 1e-6:
        print("  WARNING: the control stops at %.3f P, short of the window end %.3f P."
              % (tf.max(), a.hi))

    print("\nlevels")
    print("  live   end %.3f   max %.3f   mean %.3f" % (yl[-1], yl.max(), yl.mean()))
    print("  frozen end %.3f   max %.3f   mean %.3f   rms %.3f"
          % (yf[-1], yf.max(), yf.mean(), yf.std()))

    # 1. model-free exceedance
    n_ef = max((tf.max()-tf.min())/max(tau_int(yf)*dt, 1e-9), 1.0)
    frac = float(np.mean(yf >= yl[-1]))
    print("\nexceedance of the live endpoint (%.3f) by the control" % yl[-1])
    print("  raw sample fraction      %.4f  (%d of %d)"
          % (frac, int((yf >= yl[-1]).sum()), yf.size))
    print("  frozen maximum           %.3f  -> live endpoint is %.2fx it"
          % (yf.max(), yl[-1]/yf.max()))
    print("  independent frozen looks %.1f; with none exceeding, p < %.3f"
          % (n_ef, 1.0/n_ef) if frac == 0 else
          "  independent frozen looks %.1f; exceedance p ~ %.3f" % (n_ef, frac))

    # 2. block comparison
    print("\nhalf-period block means (live | frozen)")
    bl, bf = block_stats(tl, yl, a.block), block_stats(tf, yf, a.block)
    for i in range(max(len(bl), len(bf))):
        s = "  [%.2f,%.2f)  " % (bl[i][0], bl[i][0]+a.block) if i < len(bl) else "  " + " "*13
        s += "%7.3f" % bl[i][1] if i < len(bl) else "      -"
        s += "  |  %7.3f" % bf[i][1] if i < len(bf) else "  |        -"
        print(s)

    # 3. divergence, using the fact that the runs start identical
    n = min(yl.size, yf.size)
    if n > 1 and abs(tl[0]-tf[0]) < 1e-6:
        d = np.abs(yl[:n] - yf[:n])
        print("\ndivergence |live - frozen| (the runs start identical)")
        for q in (0.25, 0.5, 0.75, 1.0):
            k = min(int(q*(n-1)), n-1)
            print("  at t/P %.2f : %.3f   (frozen rms to that point %.3f)"
                  % (tl[k], d[k], yf[:k+1].std()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
