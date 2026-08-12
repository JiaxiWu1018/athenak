#!/usr/bin/env python3
"""Emit the report's data tables as markdown, straight from the analysis output.

Every number in the delivered report comes from here rather than being
transcribed by hand, so the tables cannot drift from the underlying CSV/JSON.
"""
import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

ORDER = ["shell_fibonacci", "radial_random", "angular_random",
         "monte_carlo", "stratified_random", "monte_carlo_antithetic"]
LETTER = {"shell_fibonacci": "A", "radial_random": "B", "angular_random": "C",
          "monte_carlo": "D", "stratified_random": "E",
          "monte_carlo_antithetic": "F"}
DESC = {"shell_fibonacci": "deterministic shells / Fibonacci / quartet (default)",
        "radial_random": "random radius only",
        "angular_random": "random angles and tangents only",
        "monte_carlo": "fully independent Monte Carlo",
        "stratified_random": "randomized stratified",
        "monte_carlo_antithetic": "antithetic Monte Carlo (paired $\\pm t$)"}


def fnum(v, fmt="{:.3e}"):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(f):
        return "—"
    return fmt.format(f)


def load_csv(p):
    if not Path(p).exists():
        return []
    with open(p) as fh:
        return list(csv.DictReader(fh))


def mean_sd(rows, key):
    vals = []
    for r in rows:
        try:
            f = float(r[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(f):
            vals.append(f)
    if not vals:
        return float("nan"), float("nan"), 0
    return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0), len(vals)


def table_runs(root, out):
    rows = load_csv(root / "analysis/provenance/run_table.csv")
    live = [r for r in rows if r["kind"] == "live" and r["completed"] == "True"]
    fixed = [r for r in rows if r["kind"] == "fixed" and r["completed"] == "True"]
    lines = ["| case | model | sampler | seed | N | t_lim / M | t_lim / P | wall (s) | state |",
             "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for r in sorted(live, key=lambda x: (x["model"], ORDER.index(x["sampler"]),
                                         int(x["seed"]))):
        P = float(r["period"])
        lines.append(
            f"| `{r['case']}` | {r['model']} | {LETTER[r['sampler']]} "
            f"{r['sampler']} | {r['seed']} | {int(r['npart']):,} | "
            f"{float(r['tlim']):.3f} | {float(r['tlim'])/P:.3f} | "
            f"{r.get('elapsed_s','—')} | completed |")
    for r in sorted(fixed, key=lambda x: ORDER.index(x["sampler"])):
        lines.append(
            f"| `{r['case']}` | fixed background | {LETTER[r['sampler']]} "
            f"{r['sampler']} | {r['seed']} | {int(r['npart']):,} | "
            f"{float(r['tlim']):.3f} | — | {r.get('elapsed_s','—')} | completed |")
    (out / "table_runs.md").write_text("\n".join(lines) + "\n")
    return len(live), len(fixed)


def table_initial(root, out, q="6.1"):
    rows = [r for r in load_csv(
        root / "analysis/initial_realizations/initial_realization_summary.csv")
        if r["radius_over_mass"] == q]
    lines = ["| sampler | per-shell $\\ell$=1 | $\\ell$=2 | $\\ell$=4 | $\\ell$=8 | "
             "KS(CDF) | $\\|P\\|$ | $\\|J\\|$ | unique positions | unique radii |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in ORDER:
        g = [r for r in rows if r["sampler"] == s]
        if not g:
            continue
        def m(k):
            return mean_sd(g, k)[0]
        lines.append(
            f"| {LETTER[s]} {DESC[s]} | {fnum(m('shellP1'))} | {fnum(m('shellP2'))} | "
            f"{fnum(m('shellP4'))} | {fnum(m('shellP8'))} | "
            f"{fnum(m('cdf_ks_error'))} | {fnum(m('P_norm'))} | {fnum(m('J_norm'))} | "
            f"{m('unique_positions_frac'):.2f}N | {int(m('unique_radii')):,} |")
    (out / "table_initial_realizations.md").write_text("\n".join(lines) + "\n")


def table_constraints(root, out, model="q6p1"):
    rows = load_csv(root / f"analysis/initial_constraints/{model}_initial_constraints.csv")
    lines = ["| sampler | seeds | $\\|H\\|_2$ at $t=0$ | $\\|M\\|_2$ at $t=0$ | "
             "$\\|J_{\\rm part}\\|$ at $t=0$ | $M_0$ |",
             "|---|---:|---:|---:|---:|---:|"]
    for s in ORDER:
        g = [r for r in rows if r["sampler"] == s]
        if not g:
            continue
        h, hsd, n = mean_sd(g, "ham_t0")
        mm, _, _ = mean_sd(g, "mom_t0")
        j, _, _ = mean_sd(g, "J_norm_t0")
        m0, _, _ = mean_sd(g, "M0_t0")
        lines.append(f"| {LETTER[s]} {DESC[s]} | {n} | {h:.4f} ± {hsd:.4f} | "
                     f"{fnum(mm)} | {fnum(j)} | {m0:.9f} |")
    (out / f"table_constraints_{model}.md").write_text("\n".join(lines) + "\n")


def table_ensemble(root, out, model="q6p1"):
    rows = load_csv(root / f"analysis/ensemble_{model}/{model}_ensemble_by_sampler.csv")
    if not rows:
        return
    # All l=4 columns use the peak and time-mean over the period, not the
    # endpoint: the mode oscillates for the coherent samplers, so an endpoint
    # value depends on where 1P falls in that oscillation.
    lines = ["| sampler | seeds | median $\\Delta\\|L\\|/\\|L\\|$ | "
             "p95 (radius-matched) | $R_{50}(1P)/R_{50}(0)$ | "
             "$\\ell$=4 peak | $\\ell$=4 time-mean | peak / initial | "
             "$\\|$align$\\|$ | torque-$\\ell$4 corr | $\\|H\\|_2$ final |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in ORDER:
        g = [r for r in rows if r["sampler"] == s]
        if not g:
            continue
        r = g[0]
        def v(k):
            return fnum(r.get(k))
        lines.append(
            f"| {LETTER[s]} {DESC[s]} | {r['n_seeds']} | "
            f"{v('dL_median_final_mean')} | "
            f"{fnum(r.get('dLm_p95_final_mean'), '{:.4f}')} ± "
            f"{fnum(r.get('dLm_p95_final_sd'), '{:.4f}')} | "
            f"{fnum(r.get('R50_ratio_mean'), '{:.5f}')} ± "
            f"{fnum(r.get('R50_ratio_sd'), '{:.1e}')} | "
            f"{v('l4_amp_peak_mean')} | "
            f"{v('l4_amp_timemean_mean')} | "
            f"{fnum(r.get('l4_peak_over_initial_mean'), '{:.0f}')} | "
            f"{fnum(r.get('l4_align_absmean_mean'), '{:.3f}')} | "
            f"{fnum(r.get('torque_l4_corr_peak_mean'), '{:.3f}')} | "
            f"{v('ham_final_mean')} |")
    (out / f"table_ensemble_{model}.md").write_text("\n".join(lines) + "\n")


def table_health(root, out, model="q6p1"):
    rows = load_csv(root / f"analysis/ensemble_{model}/{model}_ensemble_by_sampler.csv")
    if not rows:
        return
    lines = ["| sampler | particles lost | nonfinite | rest-mass change | "
             "pusher fallbacks | $\\alpha_{\\min}$ | $\\|H\\|_2$ growth | "
             "$\\|M\\|_2$ final | $\\|J\\|$ final |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in ORDER:
        g = [r for r in rows if r["sampler"] == s]
        if not g:
            continue
        r = g[0]
        lines.append(
            f"| {LETTER[s]} {DESC[s]} | {fnum(r.get('n_lost_mean'), '{:.1f}')} | "
            f"{fnum(r.get('n_nonfinite_final_mean'), '{:.1f}')} | "
            f"{fnum(r.get('M0_rel_change_mean'))} | "
            f"{fnum(r.get('geo_fallbacks_final_mean'), '{:.0f}')} | "
            f"{fnum(r.get('alpha_min_final_mean'), '{:.5f}')} | "
            f"{fnum(r.get('ham_growth_mean'), '{:.2f}')} | "
            f"{fnum(r.get('mom_final_mean'))} | "
            f"{fnum(r.get('J_norm_final_mean'))} |")
    (out / f"table_health_{model}.md").write_text("\n".join(lines) + "\n")


def table_l4_axis(root, out, model="q6p1"):
    p = root / f"analysis/ensemble_{model}/{model}_l4_axis_spread.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    lines = ["| sampler | seed pairs | median axis separation | max |",
             "|---|---:|---:|---:|"]
    for s in ORDER:
        if s not in d:
            continue
        v = d[s]
        lines.append(f"| {LETTER[s]} {DESC[s]} | {v['n_pairs']} | "
                     f"{v['median_deg']:.1f}° | {v['max_deg']:.1f}° |")
    (out / f"table_l4_axis_{model}.md").write_text("\n".join(lines) + "\n")


def table_deposited(root, out, model="q6p1"):
    p = root / f"analysis/deposited_source_{model}/deposited_source_comparison.json"
    fm = root / f"analysis/field_multipoles_{model}/field_multipoles.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    f = json.loads(fm.read_text()) if fm.exists() else {}
    lines = ["| sampler | seeds | deposited-source rms error | p95 | "
             "deposited $\\ell$=4 ($t=0$) | deposited $\\ell$=4 ($1P$) | "
             "deposited $\\ell$=8 ($t=0$) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for s in ORDER:
        cases = [v for v in d.values() if v["sampler"] == s]
        if not cases:
            continue
        rms = st.mean(c["dev_rms"] for c in cases)
        p95 = st.mean(c["dev_p95_abs"] for c in cases)
        fc = [v for v in f.values() if v["sampler"] == s]
        d4i = d4f = d8i = float("nan")
        if fc:
            d4i = st.mean(c["deposited_initial"]["rms_over_shells"]["P4"] for c in fc)
            d8i = st.mean(c["deposited_initial"]["rms_over_shells"]["P8"] for c in fc)
            fin = [c for c in fc if "deposited_final" in c]
            if fin:
                d4f = st.mean(c["deposited_final"]["rms_over_shells"]["P4"] for c in fin)
        lines.append(f"| {LETTER[s]} {DESC[s]} | {len(cases)} | {rms:.4f} | "
                     f"{p95:.4f} | {fnum(d4i)} | {fnum(d4f)} | {fnum(d8i)} |")
    (out / f"table_deposited_{model}.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["q6p1"])
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    nlive, nfixed = table_runs(args.root, args.output)
    table_initial(args.root, args.output, "6.1")
    for model in args.models:
        table_constraints(args.root, args.output, model)
        table_ensemble(args.root, args.output, model)
        table_health(args.root, args.output, model)
        table_l4_axis(args.root, args.output, model)
        table_deposited(args.root, args.output, model)
    print(f"tables written to {args.output} ({nlive} live, {nfixed} fixed runs)")
    for p in sorted(args.output.glob("*.md")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
