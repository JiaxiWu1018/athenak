#!/usr/bin/env python3
"""Endpoint multipole summary for the Plummer campaign, live against the frozen null.

Reports each radial band's l = 1..4 amplitude in units of that band's own finite-N floor,
measured BOTH about the whole-cluster centre of mass (rows m0..m3, the quantity the
established A_l diagnostic uses) and about the coordinate origin (rows o0..o3).  The gap
between the two is the geometric (2/3)<1/r>|s| term that an origin-referenced dipole picks
up from any rigid offset of the cluster from the grid origin, and it is reported explicitly
rather than left implicit, because it is largest in the core and falls off like <1/r> --
i.e. it mimics exactly the "growing, radially confined" signature it would be mistaken for.

Where a frozen-metric control is supplied, the null is its own envelope over the SAME time
window as the live comparison, not over a shorter one.

Usage:
  endpoint_summary.py --live reduced/prod_com [--frozen reduced/pf_frozen_com] [--tmax 3.0]
"""
import argparse
import csv
import os
import sys

import numpy as np


def load_modes(d):
    """{(band, l): (t_over_P[], A/A_shot[])} from a reduction directory."""
    acc = {}
    with open(os.path.join(d, "modes.csv")) as f:
        for r in csv.DictReader(f):
            k = (r["band"], int(r["l"]))
            acc.setdefault(k, {})[float(r["t_over_P"])] = (
                float(r["A_l"])/float(r["A_shot"]))
    out = {}
    for k, v in acc.items():
        t = np.array(sorted(v))                 # dict keys dedupe the duplicated final frame
        out[k] = (t, np.array([v[x] for x in t]))
    return out


def load_scalars(d):
    with open(os.path.join(d, "scalars.csv")) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True)
    ap.add_argument("--frozen")
    ap.add_argument("--tmax", type=float, default=3.0)
    a = ap.parse_args()

    L = load_modes(a.live)
    F = load_modes(a.frozen) if a.frozen else {}
    tmax_f = max(t.max() for t, _ in F.values()) if F else 0.0
    if F:
        print("frozen control covers t/P = 0 .. %.3f; live compared over the same window "
              "where a null is quoted" % tmax_f)

    LAB = {"m0": "core   (inner quartile)", "m1": "band 2", "m2": "band 3",
           "m3": "halo   (outer quartile)"}

    print("\n" + "="*104)
    print("A_l / A_shot(band) at the endpoint, and each band's run maximum")
    print("  CoM = referenced to the whole-cluster centre of mass (the physical measure)")
    print("  org = referenced to the coordinate origin (the superseded measure)")
    print("="*104)
    print("%-24s %2s | %8s %8s | %8s %8s | %9s %9s"
          % ("band", "l", "CoM end", "CoM max", "org end", "org max",
             "null(CoM)", "end/null"))
    for b in ("m0", "m1", "m2", "m3"):
        for l in (1, 2, 3, 4):
            if (b, l) not in L:
                continue
            t, y = L[(b, l)]
            m = t <= a.tmax + 1e-9
            o = L.get(("o" + b[1], l))
            fk = F.get((b, l))
            if fk is not None:
                null = fk[1].max()
                s_null, s_rat = "%9.3f" % null, "%9.2f" % (y[m][-1]/null)
            else:
                s_null = s_rat = "%9s" % "-"
            print("%-24s %2d | %8.3f %8.3f | %8s %8s | %s %s"
                  % (LAB[b] if l == 1 else "", l, y[m][-1], y[m].max(),
                     "%8.3f" % o[1][o[0] <= a.tmax + 1e-9][-1] if o else "-",
                     "%8.3f" % o[1][o[0] <= a.tmax + 1e-9].max() if o else "-",
                     s_null, s_rat))
        print()

    for b, l in (("com", 1), ("all", 1)):
        if (b, l) not in L:
            continue
        t, y = L[(b, l)]
        m = t <= a.tmax + 1e-9
        lab = "whole frame, CoM-removed" if b == "com" else "whole frame, raw"
        extra = ""
        if (b, l) in F:
            extra = "   null %.3f" % F[(b, l)][1].max()
        print("%-28s l=1: end %.3f  max %.3f%s" % (lab, y[m][-1], y[m].max(), extra))

    # CoM-referenced inner/outer halves, which were correct in the original reduction
    S = load_scalars(a.live)
    N = S["N_alive"][0]
    sh = (N/4.0)**-0.5
    tp = S["t_over_P"]
    m = tp <= a.tmax + 1e-9
    print("\nCoM-referenced inner/outer halves (A_shot = (N/4)^-1/2 = %.6e):" % sh)
    for k, lab in (("A1_inner", "inner half"), ("A1_outer", "outer half")):
        y = S[k][m]/sh
        line = "  %-11s t=0 %.3f   end %.3f   max %.3f" % (lab, y[0], y[-1], y.max())
        if a.frozen:
            SF = load_scalars(a.frozen)
            yf = SF[k]/((SF["N_alive"][0]/4.0)**-0.5)
            line += "   | frozen end %.3f  max %.3f" % (yf[-1], yf.max())
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
