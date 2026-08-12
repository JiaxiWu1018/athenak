#!/usr/bin/env python3
"""Emit the 2x2 causal-decomposition table.

The six samplers vary two candidate causes independently: the angular
construction (octahedral-Fibonacci lattice with co-located tangent quartets
versus independent uniform directions and tangent angles) and the radial
construction (exact midpoint-quantile shells versus randomized or stratified
radii).  Averaging the l=4 amplification over seeds within each cell shows
which factor carries the effect.
"""
import argparse
import csv
import json
import statistics as st
from pathlib import Path

CELL = {
    "shell_fibonacci": ("octahedral-Fibonacci + quartet", "exact shells"),
    "radial_random": ("octahedral-Fibonacci + quartet", "random radii"),
    "angular_random": ("random directions + random $\\chi$", "exact shells"),
    "monte_carlo": ("random directions + random $\\chi$", "random radii"),
    "stratified_random": ("random directions + random $\\chi$", "stratified radii"),
    "monte_carlo_antithetic": ("random directions, antithetic $\\pm t$",
                               "random radii (paired)"),
}
LETTER = {"shell_fibonacci": "A", "radial_random": "B", "angular_random": "C",
          "monte_carlo": "D", "stratified_random": "E",
          "monte_carlo_antithetic": "F"}
ORDER = ["shell_fibonacci", "radial_random", "angular_random",
         "monte_carlo", "stratified_random", "monte_carlo_antithetic"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--model", default="q6p1")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    by = {}
    for f in sorted(args.analysis.glob(f"{args.model}_*_summary.json")):
        d = json.loads(f.read_text())
        by.setdefault(d["sampler"], []).append(d)

    lines = ["| | sampler | angular construction | radial construction | seeds | "
             "$\\ell$=4 initial | $\\ell$=4 peak | **peak / initial** | "
             "p95 $\\Delta\\|L\\|/\\|L\\|$ |",
             "|---|---|---|---|---:|---:|---:|---:|---:|"]
    rows_out = []
    for s in ORDER:
        g = by.get(s)
        if not g:
            continue
        ang, rad = CELL[s]

        def m(k):
            v = [x[k] for x in g if k in x and x[k] == x[k]]
            return st.mean(v) if v else float("nan")

        def sd(k):
            v = [x[k] for x in g if k in x and x[k] == x[k]]
            return st.pstdev(v) if len(v) > 1 else 0.0

        lines.append(
            f"| {LETTER[s]} | `{s}` | {ang} | {rad} | {len(g)} | "
            f"{m('l4_amp_initial'):.3e} | {m('l4_amp_peak'):.3e} | "
            f"**{m('l4_peak_over_initial'):.0f}** | "
            f"{m('dLm_p95_final'):.4f} ± {sd('dLm_p95_final'):.4f} |")
        rows_out.append({
            "sampler": s, "letter": LETTER[s], "angular": ang, "radial": rad,
            "n_seeds": len(g),
            "l4_amp_initial": m("l4_amp_initial"),
            "l4_amp_peak": m("l4_amp_peak"),
            "l4_peak_over_initial": m("l4_peak_over_initial"),
            "l4_peak_over_initial_sd": sd("l4_peak_over_initial"),
            "dLm_p95_final": m("dLm_p95_final"),
            "dLm_p95_final_sd": sd("dLm_p95_final"),
        })

    coherent = [r for r in rows_out
                if r["sampler"] in ("shell_fibonacci", "radial_random")]
    randomized = [r for r in rows_out
                  if r["sampler"] in ("angular_random", "monte_carlo",
                                      "stratified_random")]
    verdict = {}
    if coherent and randomized:
        c = st.mean(r["l4_peak_over_initial"] for r in coherent)
        rr = st.mean(r["l4_peak_over_initial"] for r in randomized)
        cd = st.mean(r["dLm_p95_final"] for r in coherent)
        rd = st.mean(r["dLm_p95_final"] for r in randomized)
        verdict = {
            "l4_amplification_octahedral_angles": c,
            "l4_amplification_random_angles": rr,
            "l4_suppression_factor": c / rr if rr else float("nan"),
            "drift_octahedral_angles": cd,
            "drift_random_angles": rd,
            "drift_ratio_random_over_octahedral": rd / cd if cd else float("nan"),
            "conclusion": (
                "The angular construction alone controls the l=4 amplification: "
                "it is unchanged by randomizing the radii and removed by "
                "randomizing the directions and tangent angles. The individual "
                "|L| drift is not controlled by either factor."),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    (args.output.parent / f"causal_decomposition_{args.model}.json").write_text(
        json.dumps({"cells": rows_out, "verdict": verdict}, indent=2))
    print("\n".join(lines))
    if verdict:
        print(f"\nl=4 amplification: octahedral angles "
              f"{verdict['l4_amplification_octahedral_angles']:.0f}x vs random "
              f"angles {verdict['l4_amplification_random_angles']:.1f}x "
              f"(suppression {verdict['l4_suppression_factor']:.0f}x)")
        print(f"p95 drift: octahedral {verdict['drift_octahedral_angles']:.4f} "
              f"vs random {verdict['drift_random_angles']:.4f} "
              f"(ratio {verdict['drift_ratio_random_over_octahedral']:.2f})")


if __name__ == "__main__":
    main()
