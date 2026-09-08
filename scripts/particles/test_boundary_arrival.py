#!/usr/bin/env python3
"""Test whether the outflow outer boundary contaminates the late-time core result.

An adversarial review of session 1 raised the objection that the 3 P_1/2 endpoint lies
inside the causal past of the outer boundary: a signal launched at |x| = 4096 M and
travelling at the 1+log gauge speed sqrt(2) reaches the core band's mean radius at
t = 2.4225 P_1/2, so roughly half the [1.8, 3.0] fit window -- and all of the terminal
climb -- is formally contaminated.

The objection is causally correct.  Whether it MATTERS is a question about amplitude, and
that is measurable: if a boundary signal reaches the core, the gauge and constraint fields
there must show a feature at the predicted arrival time that is larger than their own
ambient wander.  This script tests exactly that, per radial bin, for the lapse, the
conformal factor, the trace of K and |H|:

  statistic  s(r) = |X(r, t2) - X(r, t1)| over the arrival window [t1, t2] = [2.30, 2.55] P
  null       the same statistic over every other window of the same length in [0.5, 3.0] P
             that does not overlap the arrival window
  reported   the percentile of s(r) in that null, and the arrival-window change expressed
             in units of the null's standard deviation

A boundary signal that mattered would sit far out in the upper tail at small r.  Usage:

  test_boundary_arrival.py --fields <run>.plummer_fields.csv [--period P] [--rmax 400]
"""
import argparse
import csv
import io
import sys

import numpy as np

# Outer boundary and gauge speed are properties of the production deck, not of this script.
XOUT = 4096.0
VGAUGE = np.sqrt(2.0)


def load(path):
    """Return (times, r_mid, {key: array[ntime, nbin] of dV-weighted means})."""
    rows = [l for l in open(path) if not l.startswith("#")]
    D = {}
    for r in csv.DictReader(io.StringIO("".join(rows))):
        D.setdefault(float(r["time"]), {})[int(r["bin"])] = r
    ts = sorted(D)
    bins = sorted(D[ts[0]])
    rlo = np.array([float(D[ts[0]][b]["r_lo"]) for b in bins])
    rhi = np.array([float(D[ts[0]][b]["r_hi"]) for b in bins])
    rm = np.sqrt(rlo*rhi)
    keys = ["alpha_dV", "chi_dV", "Khat_dV", "absH_dV"]
    out = {k: np.full((len(ts), len(bins)), np.nan) for k in keys}
    for i, t in enumerate(ts):
        dv = np.array([float(D[t][b]["dV"]) for b in bins])
        ok = dv > 0
        for k in keys:
            v = np.array([float(D[t][b][k]) for b in bins])
            out[k][i, ok] = v[ok]/dv[ok]
    # A restart landing exactly on tlim writes one output set before evolving anything; its
    # derived field arrays were never computed, so that frame is identically zero.  Drop any
    # such duplicate rather than let it enter a difference.
    keep = [i for i, t in enumerate(ts)
            if not np.all(np.nan_to_num(out["absH_dV"][i]) == 0.0)]
    if len(keep) != len(ts):
        print("  dropped %d all-zero frame(s) (no-op restart dumps)" % (len(ts) - len(keep)))
    ts = np.array(ts)[keep]
    for k in keys:
        out[k] = out[k][keep]
    return ts, rm, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--period", type=float, default=1192.496781)
    ap.add_argument("--rmax", type=float, default=400.0)
    ap.add_argument("--rcore", type=float, default=10.61,
                    help="core-band mean radius, sets the predicted arrival time")
    ap.add_argument("--halfwin", type=float, default=0.125,
                    help="half-length of the arrival window, in periods")
    a = ap.parse_args()

    ts, rm, F = load(a.fields)
    P = a.period
    tp = ts/P
    t_arr = (XOUT - a.rcore)/VGAUGE/P
    print("outer boundary %g M, gauge speed sqrt(2): a signal launched at t=0 reaches"
          % XOUT)
    print("  r = %.2f M at t/P = %.4f;  r = 400 M at t/P = %.4f"
          % (a.rcore, t_arr, (XOUT - 400.0)/VGAUGE/P))
    print("  record covers t/P = %.3f .. %.4f in %d frames" % (tp[0], tp[-1], len(tp)))

    lo, hi = t_arr - a.halfwin, t_arr + a.halfwin
    W = 2*a.halfwin
    print("\narrival window [%.3f, %.3f] P; null = all same-length windows in [0.5, %.3f] P"
          % (lo, hi, tp[-1]))

    def val(k, x):
        return F[k][int(np.argmin(np.abs(tp - x)))]

    # Null windows: same length, inside the record, not overlapping the arrival window.
    starts = [s for s in np.arange(0.5, tp[-1] - W + 1e-9, 0.01)
              if s + W <= lo or s >= hi]
    print("  %d null windows\n" % len(starts))

    for k, lab in (("alpha_dV", "lapse alpha"), ("chi_dV", "conformal chi"),
                   ("Khat_dV", "trace K"), ("absH_dV", "|H|")):
        print("=== %s ===" % lab)
        print("%9s %13s %13s %10s %9s" % ("r_mid/M", "value", "|d| arrival", "null sd",
                                          "pctile"))
        d_arr = np.abs(val(k, hi) - val(k, lo))
        null = np.array([np.abs(val(k, s + W) - val(k, s)) for s in starts])
        base = val(k, lo)
        for i in range(len(rm)):
            if not (1.5 <= rm[i] <= a.rmax) or np.isnan(base[i]):
                continue
            col = null[:, i]
            col = col[np.isfinite(col)]
            if col.size == 0:
                continue
            pct = 100.0*np.mean(col <= d_arr[i])
            print("%9.2f %13.6g %13.3e %10.3e %8.1f%%"
                  % (rm[i], base[i], d_arr[i], col.std(), pct))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
