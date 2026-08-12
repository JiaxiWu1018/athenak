#!/usr/bin/env python3
"""Cross-model ensemble comparison for the long-duration stability campaign.

Consumes the per-run outputs of analyze_stability_case.py for both
compactnesses and all seeds, and produces:

  * the 6.1-vs-5.9 distinguishability time: the first time at which the
    ensemble-mean R50/R50(0) of the two models differ by more than
    2*sqrt(scatter61^2 + scatter59^2), sustained for >= 0.25P — evaluated in
    normalized time t/P so the two models' phases align;
  * comparison figures with every boundary-contact estimate stamped on the
    time axis and event markers overplotted;
  * an ensemble event table (per model: which seeds show secular contraction,
    unbounded oscillation, cohort departures, with onset times);
  * the final verdict block, applying the pre-registered criteria:
    a claimed contrast must begin before the sqrt(2) boundary mark, persist
    >= 0.25P, appear in >= 2 matched seeds, and exceed seed scatter.
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

COLOR = {"q6p1": "#0072B2", "q5p9": "#D55E00"}
LABEL = {"q6p1": "R/M = 6.1", "q5p9": "R/M = 5.9"}


def load_runs(analysis_dir, model):
    out = []
    for f in sorted(Path(analysis_dir).glob(f"long_{model}_*_summary.json")):
        d = json.loads(f.read_text())
        ts = Path(analysis_dir) / f"{d['case']}_stability_timeseries.csv"
        rows = list(csv.DictReader(ts.open())) if ts.exists() else []
        coh = Path(analysis_dir) / f"{d['case']}_cohorts.csv"
        out.append({"summary": d, "timeseries": rows, "cohort_csv": coh})
    return out


def series_on_grid(rows, key, grid_tp):
    tp = np.array([float(r["t_over_P"]) for r in rows])
    y = np.array([float(r[key]) for r in rows])
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.full_like(grid_tp, np.nan)
    return np.interp(grid_tp, tp[ok], y[ok], left=np.nan, right=np.nan)


def distinguishability(runs61, runs59, key="R50", sustain=0.25):
    tmax = min(min(float(r["timeseries"][-1]["t_over_P"]) for r in runs61),
               min(float(r["timeseries"][-1]["t_over_P"]) for r in runs59))
    grid = np.arange(0.0, tmax, 0.02)

    def stack(runs):
        mat = []
        for r in runs:
            y = series_on_grid(r["timeseries"], key, grid)
            y0 = y[np.isfinite(y)][0] if np.isfinite(y).any() else np.nan
            mat.append(y / y0)
        return np.array(mat)

    a, b = stack(runs61), stack(runs59)
    mean_a, mean_b = np.nanmean(a, 0), np.nanmean(b, 0)
    sd_a = np.nanstd(a, 0, ddof=1) if len(a) > 1 else np.zeros_like(mean_a)
    sd_b = np.nanstd(b, 0, ddof=1) if len(b) > 1 else np.zeros_like(mean_b)
    gap = np.abs(mean_a - mean_b)
    thresh = 2.0 * np.sqrt(sd_a ** 2 + sd_b ** 2)
    distinct = gap > np.maximum(thresh, 1e-12)
    onset = None
    run0 = None
    for i in range(len(grid)):
        if distinct[i] and np.isfinite(gap[i]):
            run0 = i if run0 is None else run0
            if grid[i] - grid[run0] >= sustain:
                onset = float(grid[run0])
                break
        else:
            run0 = None
    return {"grid_tp": grid, "mean61": mean_a, "mean59": mean_b,
            "sd61": sd_a, "sd59": sd_b, "gap": gap, "threshold": thresh,
            "onset_t_over_P": onset, "t_compare_max_over_P": float(tmax)}


def mark_boundaries(ax, marks, period_label=True):
    styles = {"sqrt2": ("-", "0.15"), "v1.7": ("--", "0.45"), "light": (":", "0.6")}
    for name, m in marks.items():
        ls, c = styles.get(name, (":", "0.5"))
        ax.axvline(m["t_over_P"], ls=ls, color=c, lw=1.0)
        ax.annotate(f"boundary ({name})", xy=(m["t_over_P"], 0.02),
                    xycoords=("data", "axes fraction"), rotation=90,
                    fontsize=6, color=c, va="bottom", ha="right")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figdir = args.output / "figures"
    figdir.mkdir(exist_ok=True)

    runs = {m: load_runs(args.analysis, m) for m in ("q6p1", "q5p9")}
    if not runs["q6p1"] or not runs["q5p9"]:
        raise SystemExit("need runs for both models")
    marks = runs["q6p1"][0]["summary"]["boundary_marks"]

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False})

    # fig1: mass radii, both models, all seeds
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, key in zip(axes, ("R10", "R50", "R90")):
        for model in ("q6p1", "q5p9"):
            for i, r in enumerate(runs[model]):
                rows = r["timeseries"]
                tp = [float(x["t_over_P"]) for x in rows]
                y = np.array([float(x[key]) for x in rows])
                ax.plot(tp, y / y[0], color=COLOR[model], lw=1.1, alpha=0.8,
                        label=LABEL[model] if i == 0 else None)
        mark_boundaries(ax, marks)
        ax.set_xlabel("$t/P$")
        ax.set_ylabel(f"{key}$(t)/${key}$(0)$")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Mass radii, both compactnesses, all seeds "
                 "(boundary-contact estimates marked)", fontsize=10)
    fig.savefig(figdir / "fig1_mass_radii.png", bbox_inches="tight")
    plt.close(fig)

    # fig2: cohort trajectories (outer 20 cohorts), one panel per model
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for ax, model in zip(axes, ("q6p1", "q5p9")):
        for r in runs[model][:1]:  # representative seed
            rows = list(csv.reader(r["cohort_csv"].open()))
            head, data = rows[0], np.array(rows[1:], dtype=float)
            tp = data[:, 1]
            ncoh = data.shape[1] - 2
            for ci in range(ncoh - 20, ncoh):
                y = data[:, 2 + ci]
                y0 = y[np.isfinite(y)][0]
                cls = r["summary"]["cohorts"][ci]["st_class"] \
                    if ci < len(r["summary"]["cohorts"]) else "stable"
                col = {"stable": "0.6", "near": "#E69F00",
                       "unstable": "#D55E00"}[cls]
                ax.plot(tp, y / y0, color=col, lw=0.9, alpha=0.85)
        mark_boundaries(ax, r["summary"]["boundary_marks"])
        ax.set_xlabel("$t/P$")
        ax.set_ylabel("cohort median $r_{\\rm areal}(t)/r(0)$")
        ax.set_title(f"{LABEL[model]} — outer 20 Lagrangian cohorts "
                     "(orange: near/unstable ST class)", fontsize=9)
    fig.savefig(figdir / "fig2_cohorts.png", bbox_inches="tight")
    plt.close(fig)

    # fig3: distinguishability
    dist = {}
    for key in ("R50", "R90", "R10"):
        dist[key] = distinguishability(runs["q6p1"], runs["q5p9"], key=key)
    d = dist["R50"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    axes[0].plot(d["grid_tp"], d["mean61"], color=COLOR["q6p1"], label=LABEL["q6p1"])
    axes[0].fill_between(d["grid_tp"], d["mean61"] - d["sd61"],
                         d["mean61"] + d["sd61"], color=COLOR["q6p1"], alpha=0.2)
    axes[0].plot(d["grid_tp"], d["mean59"], color=COLOR["q5p9"], label=LABEL["q5p9"])
    axes[0].fill_between(d["grid_tp"], d["mean59"] - d["sd59"],
                         d["mean59"] + d["sd59"], color=COLOR["q5p9"], alpha=0.2)
    axes[0].set_xlabel("$t/P$"); axes[0].set_ylabel("$R_{50}(t)/R_{50}(0)$")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].plot(d["grid_tp"], d["gap"], color="0.2", label="|mean gap|")
    axes[1].plot(d["grid_tp"], d["threshold"], color="#CC79A7", ls="--",
                 label="2$\\sigma$ combined scatter")
    if d["onset_t_over_P"] is not None:
        for ax in axes:
            ax.axvline(d["onset_t_over_P"], color="#009E73", lw=1.4)
        axes[1].annotate("first distinguishable",
                         xy=(d["onset_t_over_P"], 0.9),
                         xycoords=("data", "axes fraction"), rotation=90,
                         fontsize=7, color="#009E73", ha="right")
    for ax in axes:
        mark_boundaries(ax, marks)
    axes[1].set_xlabel("$t/P$"); axes[1].set_ylabel("ensemble $R_{50}$ gap")
    axes[1].legend(fontsize=7, frameon=False)
    fig.suptitle("6.1 vs 5.9 ensemble distinguishability (normalized time)",
                 fontsize=10)
    fig.savefig(figdir / "fig3_distinguishability.png", bbox_inches="tight")
    plt.close(fig)

    # event table + verdict
    table = []
    for model in ("q6p1", "q5p9"):
        for r in runs[model]:
            s, e = r["summary"], r["summary"]["events"]
            dep = e.get("cohort_departures") or []
            pre_boundary_dep = [x for x in dep
                                if x["onset_t_over_P"] < marks["sqrt2"]["t_over_P"]]
            table.append({
                "case": s["case"], "model": model, "seed": s["seed"],
                "t_final_over_P": s["t_final_over_P"],
                "secular_contraction": e.get("secular_contraction"),
                "unbounded_oscillation": e.get("unbounded_oscillation"),
                "n_cohort_departures_pre_boundary": len(pre_boundary_dep),
                "first_departure": min((x["onset_t_over_P"] for x in pre_boundary_dep),
                                       default=None),
                "alpha_min": s.get("alpha_min_min"),
                "ham_max": s.get("ham_max"),
                "gw_dJz_cum_final": s.get("gw_dJz_cum_final"),
            })

    def model_consensus(model, field):
        vals = [row[field] for row in table if row["model"] == model]
        present = [v for v in vals if v]
        return len(present), len(vals)

    onset = dist["R50"]["onset_t_over_P"]
    clean = marks["sqrt2"]["t_over_P"]
    sec61, n61 = model_consensus("q6p1", "secular_contraction")
    sec59, n59 = model_consensus("q5p9", "secular_contraction")
    dep59 = [row["n_cohort_departures_pre_boundary"] for row in table
             if row["model"] == "q5p9"]
    dep61 = [row["n_cohort_departures_pre_boundary"] for row in table
             if row["model"] == "q6p1"]

    if onset is not None and onset < clean and sum(1 for x in dep59 if x > 0) >= 2:
        verdict = ("CONTRAST DETECTED: the ensembles separate at "
                   f"t={onset:.2f}P (inside the causally clean window, "
                   f"boundary at {clean:.2f}P), with reproducible outer-cohort "
                   "departures at R/M=5.9.")
    elif onset is not None and onset < clean:
        verdict = ("PARTIAL: ensembles separate at "
                   f"t={onset:.2f}P inside the clean window, but the cohort-"
                   "level signature is not reproducible across >=2 seeds; "
                   "treat as suggestive, not established.")
    elif onset is None:
        verdict = ("NO CONTRAST detected within the causally clean interval "
                   f"[0, {clean:.2f}P]: the 6.1 and 5.9 ensembles remain "
                   "within each other's seed scatter.")
    else:
        verdict = ("INCONCLUSIVE: separation occurs only after the "
                   f"conservative boundary mark ({clean:.2f}P); the limiting "
                   "factor is the domain size, not seed statistics.")

    result = {
        "distinguishability": {k: {"onset_t_over_P": dist[k]["onset_t_over_P"],
                                   "t_compare_max_over_P": dist[k]["t_compare_max_over_P"]}
                               for k in dist},
        "boundary_marks": marks,
        "event_table": table,
        "secular_contraction_consensus": {"q6p1": f"{sec61}/{n61}",
                                          "q5p9": f"{sec59}/{n59}"},
        "verdict": verdict,
    }
    (args.output / "ensemble_verdict.json").write_text(json.dumps(result, indent=2))
    with (args.output / "event_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        w.writeheader()
        for row in table:
            w.writerow({k: (json.dumps(v) if isinstance(v, dict) else v)
                        for k, v in row.items()})

    print(f"R50 distinguishability onset: {onset} (clean until {clean:.2f}P)")
    for row in table:
        print(f"  {row['case']:44s} t_end={row['t_final_over_P']:.2f}P "
              f"secular={'Y' if row['secular_contraction'] else 'n'} "
              f"departures={row['n_cohort_departures_pre_boundary']}")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
