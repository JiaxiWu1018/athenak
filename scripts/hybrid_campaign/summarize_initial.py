#!/usr/bin/env python3
"""Print the sampler-by-sampler initial-realization comparison table."""
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
    print(f"{len(rows)} realizations")
    for q in sorted({r["radius_over_mass"] for r in rows}, reverse=True):
        print(f"\n=== R/M = {q}")
        hdr = (f"{'sampler':22s} {'P1':>9s} {'P2':>9s} {'P3':>9s} {'P4 mean':>9s} "
               f"{'P4 sd':>9s} {'P6':>9s} {'P8':>9s} {'KS(CDF)':>9s} "
               f"{'|J|':>10s} {'|P|':>10s} {'uniq':>6s} {'peakfrac':>8s}")
        print(hdr)
        for s in ORDER:
            g = [r for r in rows if r["sampler"] == s and r["radius_over_mass"] == q]
            if not g:
                continue
            def col(k):
                return [float(r[k]) for r in g]
            print(f"{LABEL[s]:22s} "
                  f"{st.mean(col('P1')):9.2e} {st.mean(col('P2')):9.2e} "
                  f"{st.mean(col('P3')):9.2e} {st.mean(col('P4')):9.2e} "
                  f"{st.pstdev(col('P4')):9.2e} {st.mean(col('P6')):9.2e} "
                  f"{st.mean(col('P8')):9.2e} {st.mean(col('cdf_ks_error')):9.2e} "
                  f"{st.mean(col('J_norm')):10.3e} {st.mean(col('P_norm')):10.3e} "
                  f"{st.mean(col('unique_positions_frac')):6.3f} "
                  f"{st.mean(col('shell_peak_fraction')):8.3f}")
        print("\n  l=4 principal axis per seed (tests whether the mode is "
              "realization-locked):")
        for s in ORDER:
            g = [r for r in rows if r["sampler"] == s and r["radius_over_mass"] == q]
            if not g:
                continue
            axes = " ".join(f"({float(r['l4_axis_x']):+.2f},"
                            f"{float(r['l4_axis_y']):+.2f},"
                            f"{float(r['l4_axis_z']):+.2f})" for r in g)
            print(f"    {LABEL[s]:22s} {axes}")


if __name__ == "__main__":
    main()
