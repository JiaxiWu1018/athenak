#!/usr/bin/env python3
"""Ensemble comparison across samplers and seeds, plus the campaign figures.

Consumes the per-run summaries and time series written by analyze_live_case.py
and produces:
  * ensemble_summary.csv   one row per run
  * ensemble_by_sampler.csv  mean / scatter across seeds for each sampler
  * the comparison figures
The interpretation criteria are applied in report_conclusions(), which states
explicitly which of the pre-registered outcomes the data support.
"""
import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ORDER = ["shell_fibonacci", "radial_random", "angular_random",
         "monte_carlo", "stratified_random", "monte_carlo_antithetic"]
LABEL = {"shell_fibonacci": "A deterministic shells/Fibonacci/quartet",
         "radial_random": "B random radius only",
         "angular_random": "C random angles only",
         "monte_carlo": "D full Monte Carlo",
         "stratified_random": "E randomized stratified",
         "monte_carlo_antithetic": "F antithetic Monte Carlo"}
SHORT = {"shell_fibonacci": "A", "radial_random": "B", "angular_random": "C",
         "monte_carlo": "D", "stratified_random": "E",
         "monte_carlo_antithetic": "F"}
# Colour-blind-safe qualitative palette; A is deliberately the darkest so the
# deterministic control reads as the reference curve in every panel.
COLOR = {"shell_fibonacci": "#1b1b1b", "radial_random": "#0072B2",
         "angular_random": "#009E73", "monte_carlo": "#D55E00",
         "stratified_random": "#CC79A7", "monte_carlo_antithetic": "#E69F00"}


def load(analysis_dir, model):
    summaries, series = [], {}
    for f in sorted(Path(analysis_dir).glob(f"{model}_*_summary.json")):
        d = json.loads(f.read_text())
        summaries.append(d)
        ts = f.parent / f"{d['name']}_timeseries.csv"
        if ts.exists():
            with ts.open() as fh:
                series[d["name"]] = list(csv.DictReader(fh))
    return summaries, series


def agg(rows, key):
    vals = [r[key] for r in rows
            if key in r and r[key] is not None and np.isfinite(r[key])]
    if not vals:
        return float("nan"), float("nan"), 0
    return (st.mean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0, len(vals))


def by_sampler(summaries):
    out = []
    for s in ORDER:
        rows = [r for r in summaries if r["sampler"] == s]
        if not rows:
            continue
        rec = {"sampler": s, "label": LABEL[s], "n_seeds": len(rows)}
        for key in ("dL_median_final", "dL_p95_final", "dL_max_final",
                    "dLm_median_final", "dLm_p95_final", "dLm_max_final",
                    "dLm_frac_kept",
                    "R10_ratio", "R25_ratio", "R50_ratio", "R90_ratio",
                    "P4_initial", "P4_final", "P4_growth",
                    "l4_amp_initial", "l4_amp_final", "l4_align_final",
                    "l4_amp_peak", "l4_amp_peak_time_over_P", "l4_amp_timemean",
                    "l4_peak_over_initial", "l4_align_absmean",
                    "torque_l4_corr_peak",
                    "torque_l4_corr_final", "torque_rms_final",
                    "ham_final", "ham_growth", "mom_final",
                    "M0_rel_change", "alpha_min_final", "rho_max_final",
                    "vr_inner10_final", "J_norm_final", "geo_fallbacks_final",
                    "n_lost", "n_nonfinite_final", "t_final_over_P"):
            m, sd, n = agg(rows, key)
            rec[f"{key}_mean"] = m
            rec[f"{key}_sd"] = sd
        rec["seeds"] = ",".join(str(r["seed"]) for r in rows)
        out.append(rec)
    return out


def l4_axis_spread(summaries):
    """Angular spread of the final l=4 axis across seeds, per sampler.

    A mode fixed by the grid would give the same axis for every realization; a
    mode carried by the particle realization gives axes that scatter with the
    seed.  Axes are direction-less (l=4 is even), so the angle is folded to
    [0,90] degrees.
    """
    out = {}
    for s in ORDER:
        rows = [r for r in summaries if r["sampler"] == s]
        if len(rows) < 2:
            continue
        ax = np.array([r["l4_axis_final"] for r in rows])
        ax /= np.linalg.norm(ax, axis=1, keepdims=True)
        angles = []
        for i in range(len(ax)):
            for j in range(i + 1, len(ax)):
                c = abs(float(np.dot(ax[i], ax[j])))
                angles.append(math.degrees(math.acos(min(c, 1.0))))
        out[s] = {"n_pairs": len(angles), "mean_deg": float(np.mean(angles)),
                  "median_deg": float(np.median(angles)),
                  "max_deg": float(np.max(angles))}
    return out


def figures(summaries, series, outdir, model, period):
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False})

    def curves(ax, ykey, ylabel, logy=False):
        for s in ORDER:
            names = [r["name"] for r in summaries if r["sampler"] == s]
            first = True
            for nm in names:
                rows = series.get(nm)
                if not rows:
                    continue
                t = [float(r["t_over_P"]) for r in rows]
                y = [float(r[ykey]) for r in rows]
                ax.plot(t, y, color=COLOR[s], lw=1.3, alpha=0.75,
                        label=LABEL[s] if first else None)
                first = False
        ax.set_xlabel("$t/P$")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")

    # fig01 individual |L| drift
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    for ax, key, lab in zip(axes, ("dL_median", "dL_p95", "dL_max"),
                            ("median $\\Delta|L|/|L|$", "p95 $\\Delta|L|/|L|$",
                             "max $\\Delta|L|/|L|$")):
        curves(ax, key, lab, logy=True)
    axes[0].legend(fontsize=6.5, loc="lower right", frameon=False)
    fig.suptitle(f"Individual angular-momentum drift, {model} "
                 f"(every sampler, every seed)", fontsize=10)
    fig.savefig(outdir / "fig01_L_drift.png", bbox_inches="tight")
    plt.close(fig)

    # fig02 l=4 amplitude, alignment with t=0, and torque correlation
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    curves(axes[0], "l4_amp", "$\\ell=4$ density amplitude", logy=True)
    curves(axes[1], "l4_align_t0", "alignment of $\\ell=4$ with its $t=0$ pattern")
    axes[1].axhline(0.0, color="0.4", lw=0.8, ls=":")
    axes[1].set_ylim(-1.05, 1.05)
    curves(axes[2], "torque_l4_corr", "corr($|dL/dt|$, $\\ell=4$ mode)")
    axes[2].axhline(0.0, color="0.4", lw=0.8, ls=":")
    axes[0].legend(fontsize=6.5, loc="best", frameon=False)
    fig.suptitle(f"The $\\ell=4$ density mode and its torque coupling, {model}",
                 fontsize=10)
    fig.savefig(outdir / "fig02_l4_mode.png", bbox_inches="tight")
    plt.close(fig)

    # fig03 radial contraction
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    for ax, key, lab in zip(axes, ("R10", "R50", "R90"),
                            ("$R_{10}/R_{10}(0)$", "$R_{50}/R_{50}(0)$",
                             "$R_{90}/R_{90}(0)$")):
        for s in ORDER:
            first = True
            for r in [x for x in summaries if x["sampler"] == s]:
                rows = series.get(r["name"])
                if not rows:
                    continue
                t = [float(x["t_over_P"]) for x in rows]
                y0 = float(rows[0][key])
                y = [float(x[key]) / y0 for x in rows]
                ax.plot(t, y, color=COLOR[s], lw=1.3, alpha=0.75,
                        label=LABEL[s] if first else None)
                first = False
        ax.set_xlabel("$t/P$")
        ax.set_ylabel(lab)
    axes[0].legend(fontsize=6.5, loc="best", frameon=False)
    fig.suptitle(f"Mass-radius contraction, {model}", fontsize=10)
    fig.savefig(outdir / "fig03_contraction.png", bbox_inches="tight")
    plt.close(fig)

    # fig04 multipole spectrum, initial and final
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    ls = range(1, 9)
    for s in ORDER:
        rows = [r for r in summaries if r["sampler"] == s]
        if not rows:
            continue
        for ax, when in zip(axes, ("first", "last")):
            spec = []
            for l in ls:
                vals = []
                for r in rows:
                    ts = series.get(r["name"])
                    if not ts:
                        continue
                    vals.append(float(ts[0 if when == "first" else -1][f"P{l}"]))
                spec.append(st.mean(vals) if vals else float("nan"))
            ax.plot(list(ls), spec, "o-", color=COLOR[s], ms=3.5, lw=1.3,
                    label=LABEL[s] if when == "first" else None)
    for ax, ttl in zip(axes, ("$t=0$", f"$t=1P$")):
        ax.set_yscale("log")
        ax.set_xlabel("$\\ell$")
        ax.set_ylabel("particle-density multipole amplitude")
        ax.set_title(ttl, fontsize=9)
    axes[0].legend(fontsize=6.5, loc="best", frameon=False)
    fig.suptitle(f"Angular multipole spectrum of the particle density, {model}",
                 fontsize=10)
    fig.savefig(outdir / "fig04_multipole_spectrum.png", bbox_inches="tight")
    plt.close(fig)

    # fig05 ensemble endpoint comparison with seed scatter
    stats = by_sampler(summaries)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    keys = [("dL_p95_final", "p95 $\\Delta|L|/|L|$ at $1P$", True),
            ("R50_ratio", "$R_{50}(1P)/R_{50}(0)$", False),
            ("ham_final", "final Hamiltonian norm", True)]
    for ax, (key, lab, logy) in zip(axes, keys):
        xs, ys, es, cs, ticks = [], [], [], [], []
        for i, rec in enumerate(stats):
            xs.append(i)
            ys.append(rec[f"{key}_mean"])
            es.append(rec[f"{key}_sd"])
            cs.append(COLOR[rec["sampler"]])
            ticks.append(SHORT[rec["sampler"]])
        ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor="0.5", capsize=3, lw=1)
        ax.scatter(xs, ys, c=cs, s=45, zorder=3)
        # individual seeds
        for i, rec in enumerate(stats):
            pts = [r[key] for r in summaries
                   if r["sampler"] == rec["sampler"] and key in r
                   and np.isfinite(r[key])]
            ax.scatter([i] * len(pts), pts, c=COLOR[rec["sampler"]], s=9,
                       alpha=0.5, zorder=2)
        ax.set_xticks(xs)
        ax.set_xticklabels(ticks)
        ax.set_ylabel(lab)
        if logy:
            ax.set_yscale("log")
    fig.suptitle(f"Ensemble endpoints at one surface period, {model} "
                 f"(mean $\\pm$ seed scatter, individual seeds shown)",
                 fontsize=10)
    fig.savefig(outdir / "fig05_ensemble_endpoints.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--period", type=float, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    summaries, series = load(args.analysis, args.model)
    if not summaries:
        raise SystemExit(f"no summaries for {args.model} in {args.analysis}")
    args.output.mkdir(parents=True, exist_ok=True)

    keys = sorted({k for s in summaries for k in s
                   if not isinstance(s[k], (list, dict))})
    with (args.output / f"{args.model}_ensemble_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "sampler", "seed"] +
                           [k for k in keys if k not in ("name", "sampler", "seed")])
        w.writeheader()
        for s in sorted(summaries, key=lambda r: (ORDER.index(r["sampler"]), r["seed"])):
            w.writerow({k: v for k, v in s.items()
                        if not isinstance(v, (list, dict))})

    stats = by_sampler(summaries)
    with (args.output / f"{args.model}_ensemble_by_sampler.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stats[0].keys()))
        w.writeheader()
        w.writerows(stats)

    spread = l4_axis_spread(summaries)
    (args.output / f"{args.model}_l4_axis_spread.json").write_text(
        json.dumps(spread, indent=2))

    figures(summaries, series, args.output / "figures", args.model, args.period)

    print(f"=== {args.model}: {len(summaries)} runs")
    print(f"{'sampler':34s} {'seeds':>5s} {'p95 matched':>12s} {'sd':>9s} "
          f"{'l4 peak':>10s} {'l4 mean':>10s} {'peak/init':>10s} "
          f"{'|align|':>8s} {'tq corr':>8s} {'R50':>8s}")
    for rec in stats:
        print(f"{rec['label'][:34]:34s} {rec['n_seeds']:5d} "
              f"{rec['dLm_p95_final_mean']:12.4e} "
              f"{rec['dLm_p95_final_sd']:9.2e} "
              f"{rec['l4_amp_peak_mean']:10.3e} "
              f"{rec['l4_amp_timemean_mean']:10.3e} "
              f"{rec['l4_peak_over_initial_mean']:10.1f} "
              f"{rec['l4_align_absmean_mean']:8.3f} "
              f"{rec['torque_l4_corr_peak_mean']:8.3f} "
              f"{rec['R50_ratio_mean']:8.5f}")
    print("\nl=4 final-axis pairwise spread across seeds (degrees):")
    for s, v in spread.items():
        print(f"  {LABEL[s]:40s} median={v['median_deg']:6.1f} "
              f"max={v['max_deg']:6.1f}")


if __name__ == "__main__":
    main()
