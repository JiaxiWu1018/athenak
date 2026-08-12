#!/usr/bin/env python3
"""Print the shell-coherence view of the initial realizations."""
import argparse
import csv
import statistics as st
from pathlib import Path

ORDER = ["shell_fibonacci", "radial_random", "angular_random",
         "monte_carlo", "stratified_random", "monte_carlo_antithetic"]
LABEL = {"shell_fibonacci": "A shell_fibonacci", "radial_random": "B radial_random",
         "angular_random": "C angular_random", "monte_carlo": "D monte_carlo",
         "stratified_random": "E stratified_random",
         "monte_carlo_antithetic": "F mc_antithetic"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    args = ap.parse_args()
    rows = list(csv.DictReader(args.csv.open()))
    for q in sorted({r["radius_over_mass"] for r in rows}, reverse=True):
        print(f"\n=== R/M = {q}   (per-shell multipole amplitude, and shell-to-shell")
        print("    coherence ratio: 1 = independent shells, sqrt(128)=11.3 = fully coherent)")
        for s in ORDER:
            g = [r for r in rows if r["sampler"] == s and r["radius_over_mass"] == q]
            if not g:
                continue
            def m(k):
                return st.mean(float(r[k]) for r in g)
            amp = " ".join(f"{m(f'shellP{l}'):8.2e}" for l in range(1, 9))
            coh = " ".join(f"{m(f'coh{l}'):8.2f}" for l in range(1, 9))
            print(f"  {LABEL[s]:22s}")
            print(f"    per-shell amp l=1..8: {amp}")
            print(f"    coherence ratio     : {coh}")
        print(f"\n  {'sampler':22s} {'uniq radii':>11s} {'repeated r':>11s} "
              f"{'peak contr':>11s} {'KS(CDF)':>10s}")
        for s in ORDER:
            g = [r for r in rows if r["sampler"] == s and r["radius_over_mass"] == q]
            if not g:
                continue
            def m(k):
                return st.mean(float(r[k]) for r in g)
            print(f"  {LABEL[s]:22s} {m('unique_radii'):11.0f} "
                  f"{m('repeated_radius_fraction'):11.4f} "
                  f"{m('shell_peak_contrast'):11.1f} {m('cdf_ks_error'):10.2e}")


if __name__ == "__main__":
    main()
