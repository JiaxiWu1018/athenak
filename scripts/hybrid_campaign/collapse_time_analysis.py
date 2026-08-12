#!/usr/bin/env python3
"""Matched-seed collapse-time comparison for the hybrid stability campaign.

Every run ended in gravitational collapse (lapse-triggered particle excision
followed by a terminal nonfinite cascade), so the compactness comparison is a
paired timing analysis: for each seed, the SAME hash-based realization was
evolved at both compactnesses, so the pair difference cancels realization-level
scatter that would otherwise swamp a 3-seed ensemble.

Collapse onset := time of the first lapse-triggered destruction record.
Times are compared in normalized units t/P (each model's own surface period)
and, separately, in absolute M.  A paired t statistic on the per-seed
differences is reported with the usual caveat that n=3 pairs gives 2 dof;
the sign consistency across pairs is the primary evidence.
"""
import argparse
import json
import math
import re
import statistics as st
from pathlib import Path

PERIODS = {"q6p1": 94.661770035504048, "q5p9": 90.044644089281221}
SEEDS = [1985, 424242, 20260801]


def collapse_onset(run_dir, case):
    dl = run_dir / f"{case}.prtcl_destroy.csv"
    if not dl.exists():
        return None
    with dl.open() as fh:
        fh.readline()
        row = fh.readline().split(",")
    return float(row[1]) if len(row) > 2 else None


def nonfinite_onset(run_dir, case):
    hst = run_dir / f"{case}.user.hst"
    if not hst.exists():
        return None
    for line in hst.open():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        try:
            if float(f[19]) > 0:
                return float(f[0])
        except (ValueError, IndexError):
            continue
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in ("q5p9", "q6p1"):
        for seed in SEEDS:
            case = f"long_{model}_stratified_antithetic_s{seed}"
            run = args.root / "runs" / case
            onset = collapse_onset(run, case)
            nf = nonfinite_onset(run, case)
            P = PERIODS[model]
            rows.append({
                "case": case, "model": model, "seed": seed, "period": P,
                "collapse_onset_M": onset,
                "collapse_onset_over_P": None if onset is None else onset / P,
                "nonfinite_onset_M": nf,
                "terminal": (run / "COLLAPSED_TERMINAL").exists(),
            })

    by = {(r["model"], r["seed"]): r for r in rows}
    pairs = []
    for seed in SEEDS:
        a, b = by[("q5p9", seed)], by[("q6p1", seed)]
        if a["collapse_onset_over_P"] is None or b["collapse_onset_over_P"] is None:
            continue
        pairs.append({
            "seed": seed,
            "q5p9_over_P": a["collapse_onset_over_P"],
            "q6p1_over_P": b["collapse_onset_over_P"],
            "lead_q5p9_over_P": b["collapse_onset_over_P"] - a["collapse_onset_over_P"],
            "q5p9_M": a["collapse_onset_M"], "q6p1_M": b["collapse_onset_M"],
            "lead_q5p9_M": b["collapse_onset_M"] - a["collapse_onset_M"],
        })

    diffs = [p["lead_q5p9_over_P"] for p in pairs]
    mean_d = st.mean(diffs)
    sd_d = st.stdev(diffs) if len(diffs) > 1 else float("nan")
    n = len(diffs)
    tstat = mean_d / (sd_d / math.sqrt(n)) if n > 1 and sd_d > 0 else float("nan")
    same_sign = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)

    means = {m: st.mean([r["collapse_onset_over_P"] for r in rows
                         if r["model"] == m and r["collapse_onset_over_P"]])
             for m in ("q5p9", "q6p1")}
    sds = {m: st.stdev([r["collapse_onset_over_P"] for r in rows
                        if r["model"] == m and r["collapse_onset_over_P"]])
           for m in ("q5p9", "q6p1")}

    result = {
        "per_run": rows, "matched_pairs": pairs,
        "q5p9_mean_over_P": means["q5p9"], "q5p9_sd": sds["q5p9"],
        "q6p1_mean_over_P": means["q6p1"], "q6p1_sd": sds["q6p1"],
        "paired_mean_lead_q5p9_over_P": mean_d,
        "paired_sd": sd_d, "paired_t": tstat, "n_pairs": n,
        "sign_consistent": same_sign,
        "note": ("Positive lead = R/M=5.9 collapses earlier in normalized time. "
                 "Unpaired model distributions overlap (seed sd ~ model gap); "
                 "the matched-seed pairing is what resolves the ordering."),
    }
    (args.output / "collapse_times.json").write_text(json.dumps(result, indent=2))

    lines = ["| seed | R/M=5.9 onset (t/P) | R/M=6.1 onset (t/P) | 5.9 lead (P) | 5.9 lead (M) |",
             "|---:|---:|---:|---:|---:|"]
    for p in pairs:
        lines.append(f"| {p['seed']} | {p['q5p9_over_P']:.3f} | "
                     f"{p['q6p1_over_P']:.3f} | {p['lead_q5p9_over_P']:+.3f} | "
                     f"{p['lead_q5p9_M']:+.1f} |")
    lines.append(f"| **mean ± sd** | {means['q5p9']:.3f} ± {sds['q5p9']:.3f} | "
                 f"{means['q6p1']:.3f} ± {sds['q6p1']:.3f} | "
                 f"**{mean_d:+.3f} ± {sd_d:.3f}** | |")
    (args.output / "collapse_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\npaired t = {tstat:.2f} (n={n} pairs, {n-1} dof), "
          f"sign-consistent: {same_sign}")


if __name__ == "__main__":
    main()
