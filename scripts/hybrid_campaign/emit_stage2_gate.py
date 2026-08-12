#!/usr/bin/env python3
"""Stage-2 gate: compare the hybrid sampler against its three references.

Merges the three new stratified_antithetic 1P summaries with the preserved
2026-08-01 campaign summaries (same grid, same seeds where available) for
monte_carlo_antithetic, stratified_random, and shell_fibonacci, and applies
the pre-registered expectations:

    E1  l=4 peak/initial amplification ~ 2      (i.e. < 10, vs 347 for A)
    E2  radius-matched p95 d|L|/|L| ~ 0.065     (within [0.04, 0.09])
    E3  seed scatter comparable to F            (sd < 0.02)
    E4  KS(CDF) <= stratified_random level      (< 1e-5)
    E5  |P| = |J| = 0 at roundoff; N and M0 unchanged; no losses/nonfinite

The gate PASSES only if all five hold across the three seeds.  On failure the
script reports which expectation failed and for which seed; nothing is tuned
or discarded.
"""
import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

PREV = Path("/work1/eliasmost/jiaxiwu/nrpic_sampler_causality_20260801/analysis/live_q6p1")


def load_summaries(d, sampler):
    out = []
    for f in sorted(Path(d).glob(f"q6p1_{sampler}_s*_summary.json")):
        out.append(json.loads(f.read_text()))
    return out


def agg(rows, key):
    vals = [r[key] for r in rows if key in r and r[key] == r[key]]
    if not vals:
        return float("nan"), float("nan")
    return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", type=Path, required=True,
                    help="analysis dir with the stratified_antithetic summaries")
    ap.add_argument("--initial-csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    groups = {
        "G stratified_antithetic (new)": load_summaries(args.new, "stratified_antithetic"),
        "F monte_carlo_antithetic": load_summaries(PREV, "monte_carlo_antithetic"),
        "E stratified_random": load_summaries(PREV, "stratified_random"),
        "A shell_fibonacci": load_summaries(PREV, "shell_fibonacci"),
    }
    init = [r for r in csv.DictReader(args.initial_csv.open())
            if r["radius_over_mass"] == "6.1"]

    lines = ["| sampler | seeds | $\\ell$=4 peak/init | p95 matched $\\Delta\\|L\\|/\\|L\\|$ | seed sd | "
             "$\\ell$=4 peak | $R_{50}$ ratio | $\\|H\\|_2$ final | lost | nonfinite |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    rows_out = {}
    for label, rows in groups.items():
        if not rows:
            continue
        m = {k: agg(rows, k) for k in
             ("l4_peak_over_initial", "dLm_p95_final", "l4_amp_peak",
              "R50_ratio", "ham_final", "n_lost", "n_nonfinite_final")}
        rows_out[label] = {k: v[0] for k, v in m.items()}
        rows_out[label]["dLm_p95_sd"] = m["dLm_p95_final"][1]
        rows_out[label]["n_seeds"] = len(rows)
        lines.append(
            f"| {label} | {len(rows)} | {m['l4_peak_over_initial'][0]:.1f} | "
            f"{m['dLm_p95_final'][0]:.4f} | {m['dLm_p95_final'][1]:.4f} | "
            f"{m['l4_amp_peak'][0]:.3e} | {m['R50_ratio'][0]:.5f} | "
            f"{m['ham_final'][0]:.3e} | {m['n_lost'][0]:.0f} | "
            f"{m['n_nonfinite_final'][0]:.0f} |")
    (args.output / "stage2_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    g = groups["G stratified_antithetic (new)"]
    gi = [r for r in init if r["sampler"] == "stratified_antithetic"]
    checks = {}
    checks["E1_l4_amplification_lt10"] = all(
        r["l4_peak_over_initial"] < 10 for r in g)
    checks["E2_p95_in_band"] = all(
        0.04 <= r["dLm_p95_final"] <= 0.09 for r in g)
    checks["E3_seed_scatter_lt_0p02"] = agg(g, "dLm_p95_final")[1] < 0.02
    checks["E4_ks_below_1e_minus5"] = all(
        float(r["cdf_ks_error"]) < 1e-5 for r in gi)
    checks["E5_conservation"] = (
        all(float(r["P_norm"]) == 0.0 and float(r["J_norm"]) == 0.0 for r in gi)
        and all(r["n_lost"] == 0 and r["n_nonfinite_final"] == 0 for r in g)
        and all(abs(r["M0_rel_change"]) < 1e-12 for r in g))
    verdict = "PASS" if all(checks.values()) else "FAIL"

    detail = {
        "checks": checks, "verdict": verdict,
        "per_seed": [{"seed": r["seed"],
                      "l4_peak_over_initial": r["l4_peak_over_initial"],
                      "dLm_p95_final": r["dLm_p95_final"],
                      "R50_ratio": r["R50_ratio"],
                      "n_lost": r["n_lost"],
                      "n_nonfinite": r["n_nonfinite_final"]} for r in g],
        "table": rows_out,
    }
    (args.output / "stage2_gate.json").write_text(json.dumps(detail, indent=2))
    print(f"\nGATE: {verdict}")
    for k, v in checks.items():
        print(f"  {k}: {'ok' if v else 'FAILED'}")
    raise SystemExit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
