#!/usr/bin/env python3
"""Render the two concise GI Session-002 closeout figures from reduced tables."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            converted = {}
            for key, value in row.items():
                if value in ("True", "False"):
                    converted[key] = value == "True"
                elif value == "":
                    converted[key] = math.nan
                else:
                    try:
                        converted[key] = float(value)
                    except ValueError:
                        converted[key] = value
            rows.append(converted)
    return rows


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#202124",
        "axes.labelcolor": "#202124",
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfbfb",
        "grid.color": "#d9dde3",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.65,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def fmt_dx(dx: float) -> str:
    reciprocal = round(1 / dx) if dx > 0 else 0
    if reciprocal and math.isclose(dx, 1 / reciprocal, rel_tol=1e-10):
        return f"1/{reciprocal}"
    return f"{dx:g}"


def occupancy_figure(rows: list[dict], events: dict, png: str, pdf: str) -> dict:
    levels = sorted({int(row["level"]) for row in rows})
    if len(levels) != 10:
        raise ValueError(f"expected 10 occupied AMR levels, found {levels}")
    if not all(int(row["true_min_including_empty"]) == 0 for row in rows):
        raise ValueError("true minimum is not zero everywhere; caption convention changed")

    run_event = events["runs"]["g2_L128"]
    vertical = [
        (run_event["first_fastflow_candidate_time"], "first FastFlow candidate",
         "#7b8794", ":"),
        (run_event["persistent_sequence_start_time"], "persistent validated AH",
         "#202124", "--"),
        (run_event["comfortably_resolved_rmin_ge_8dx_time"], r"$r_{\min}\geq 8\,dx$",
         "#c68a00", "-."),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(16.0, 7.8), sharex=True, sharey=True)
    axes = axes.ravel()
    series = [
        ("min_occupied", "minimum, occupied cells", "#7b8794", ":", "o"),
        ("mean_all_active", "mean, all active cells", "#1769aa", "-", "s"),
        ("max", "maximum", "#d97706", "--", "^"),
    ]
    positive_values = []
    for row in rows:
        positive_values.extend(float(row[key]) for key, *_ in series
                               if np.isfinite(float(row[key])) and float(row[key]) > 0)
    ymin = 10 ** math.floor(math.log10(min(positive_values))) / 1.6
    ymax = 10 ** math.ceil(math.log10(max(positive_values))) * 1.2
    for ax, level in zip(axes, levels):
        lev = sorted((row for row in rows if int(row["level"]) == level),
                     key=lambda row: row["time"])
        time = np.array([row["time"] for row in lev])
        for key, _, color, linestyle, marker in series:
            value = np.array([row[key] for row in lev], dtype=float)
            ax.plot(time, value, color=color, linestyle=linestyle, marker=marker,
                    markersize=3.0, linewidth=1.25, markeredgewidth=0.4)
        for x, _, color, linestyle in vertical:
            if x is not None:
                ax.axvline(x, color=color, linestyle=linestyle, linewidth=0.9,
                           alpha=0.85, zorder=0)
        dx = lev[0]["dx"]
        cells = int(lev[0]["active_leaf_cells"])
        blocks = int(lev[0]["active_leaf_blocks"])
        ax.set_title(f"L{level}  ·  dx={fmt_dx(dx)}\n"
                     f"{blocks} blocks / {cells:,} leaf cells", pad=5)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, which="major")
        ax.grid(True, which="minor", alpha=0.25)
        ax.set_xlim(min(time), max(time))

    axes[-1].set_yscale("log")
    for ax in axes[5:]:
        ax.set_xlabel("simulation time")
    axes[0].set_ylabel("particles per active leaf cell")
    axes[5].set_ylabel("particles per active leaf cell")
    legend = [Line2D([0], [0], color=color, linestyle=linestyle, marker=marker,
                     markersize=4, linewidth=1.4, label=label)
              for _, label, color, linestyle, marker in series]
    legend.extend(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.0,
                         label=label) for _, label, color, linestyle in vertical)
    fig.legend(handles=legend, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.985), fontsize=8.5)
    fig.suptitle("Particle occupancy by active AMR leaf level — g2_L128",
                 x=0.5, y=1.02, fontsize=14, fontweight="semibold")
    fig.text(0.5, 0.008,
             "Minimum shown is among occupied cells. The literal minimum including empty "
             "active leaf cells is zero at every level and output time. "
             "Means include empty cells.",
             ha="center", va="bottom", fontsize=9, color="#454b52")
    fig.tight_layout(rect=[0.025, 0.045, 0.995, 0.93], w_pad=1.0, h_pad=1.2)
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {
        "levels": levels,
        "time_count": len(sorted({row["time"] for row in rows})),
        "true_min_including_empty_zero_everywhere": True,
        "main_minimum": "minimum among occupied active leaf cells",
        "mean_denominator": "all active leaf cells at level, including empty cells",
        "events": {label: x for x, label, _, _ in vertical},
    }


def consecutive_unique_centres(rows: list[dict]) -> np.ndarray:
    points = []
    previous = None
    for row in rows:
        value = np.array([row["time"], row["center_x"], row["center_y"], row["center_z"]])
        xyz = tuple(value[1:])
        if previous != xyz:
            points.append(value)
            previous = xyz
    return np.array(points)


def set_trajectory_limits(ax, x: np.ndarray, y: np.ndarray) -> None:
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xpad = max(0.004, 0.12 * max(xmax - xmin, 1e-6))
    ypad = max(0.004, 0.18 * max(ymax - ymin, 1e-6))
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_aspect("equal", adjustable="box")


def trajectory_panel(ax, rows: list[dict], ordinate: str, ylabel: str,
                     norm: Normalize, cmap) -> None:
    all_points = consecutive_unique_centres(rows)
    persistent = [row for row in rows if row["after_persistent_validation"]]
    points = consecutive_unique_centres(persistent)
    ax.plot(all_points[:, 1], all_points[:, 2 if ordinate == "center_y" else 3],
            color="#aeb6bf", linewidth=1.0, alpha=0.8, zorder=1)
    yindex = 2 if ordinate == "center_y" else 3
    ax.plot(points[:, 1], points[:, yindex], color="#39424e", linewidth=1.0,
            alpha=0.8, zorder=2)
    ax.scatter(points[:, 1], points[:, yindex], c=points[:, 0], cmap=cmap, norm=norm,
               s=22, edgecolor="white", linewidth=0.35, zorder=3)
    ax.scatter([-3.0], [0.0], marker="x", s=55, color="#202124", linewidth=1.4,
               label="original clump centre", zorder=5)
    ax.scatter([points[0, 1]], [points[0, yindex]], marker="o", s=70,
               facecolor="white", edgecolor="#1769aa", linewidth=1.5,
               label="persistent validation begins", zorder=6)
    ax.scatter([points[-1, 1]], [points[-1, yindex]], marker="*", s=95,
               facecolor="#d97706", edgecolor="#6f4500", linewidth=0.6,
               label="final", zorder=6)
    if len(points) >= 3:
        start = max(0, len(points) - 3)
        ax.annotate("", xy=(points[-1, 1], points[-1, yindex]),
                    xytext=(points[start, 1], points[start, yindex]),
                    arrowprops=dict(arrowstyle="->", color="#202124", lw=1.1))
    xall = np.r_[all_points[:, 1], -3.0]
    yall = np.r_[all_points[:, yindex], 0.0]
    set_trajectory_limits(ax, xall, yall)
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.grid(True)


def ah_figure(rows128: list[dict], rows64: list[dict], events: dict,
              png: str, pdf: str) -> dict:
    persistent128 = [row for row in rows128 if row["after_persistent_validation"]]
    persistent64 = [row for row in rows64 if row["after_persistent_validation"]]
    if not persistent128:
        raise ValueError("no persistent g2_L128 AH rows")
    final = persistent128[-1]
    if not math.isclose(final["m_irr"], 0.120709, abs_tol=5e-7):
        raise ValueError(f"unexpected final g2_L128 M_irr={final['m_irr']}")

    fig = plt.figure(figsize=(16.4, 5.2))
    grid = fig.add_gridspec(1, 4, width_ratios=(1.0, 1.0, 1.0, 0.035),
                            wspace=0.30)
    axes = np.array([fig.add_subplot(grid[0, index]) for index in range(3)])
    colorbar_axis = fig.add_subplot(grid[0, 3])
    ax = axes[0]
    pre128 = [row for row in rows128 if not row["after_persistent_validation"]]
    ax.plot([row["time"] for row in persistent64],
            [row["m_irr"] for row in persistent64], color="#aeb6bf", linewidth=1.1,
            label=r"$dx=1/64$ (context)", zorder=1)
    ax.scatter([row["time"] for row in pre128], [row["m_irr"] for row in pre128],
               s=12, facecolor="white", edgecolor="#7b8794", linewidth=0.6,
               label="isolated gate-passing surfaces", zorder=2)
    ax.plot([row["time"] for row in persistent128],
            [row["m_irr"] for row in persistent128], color="#1769aa", linewidth=1.7,
            label=r"$dx=1/128$ after persistence gate", zorder=3)
    ax.axhline(0.12, color="#202124", linestyle="--", linewidth=1.0,
               label=r"clump ADM parameter $M_1=0.12$")
    ax.axhline(0.13006, color="#d97706", linestyle=":", linewidth=1.1,
               label="clump rest mass 0.13006")
    ax.scatter([final["time"]], [final["m_irr"]], marker="*", s=75,
               color="#d97706", edgecolor="#6f4500", linewidth=0.5, zorder=5)
    ax.annotate(f"{final['m_irr']:.6f}\nat t={final['time']:.4f}",
                xy=(final["time"], final["m_irr"]), xytext=(-66, -34),
                textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#4c5560", lw=0.8))
    ax.set_xlabel("simulation time")
    ax.set_ylabel(r"irreducible mass $M_{\rm irr}$")
    ax.set_title("Validated horizon mass")
    ax.grid(True)
    ax.legend(frameon=False, fontsize=7.4, loc="upper left")

    tmin = persistent128[0]["time"]
    tmax = persistent128[-1]["time"]
    norm = Normalize(vmin=tmin, vmax=tmax)
    cmap = plt.get_cmap("cividis")
    trajectory_panel(axes[1], rows128, "center_y", "y", norm, cmap)
    axes[1].set_title("AH centre in the x-y plane")
    trajectory_panel(axes[2], rows128, "center_z", "z", norm, cmap)
    axes[2].set_title("AH centre in the x-z plane")
    axes[2].legend(frameon=False, fontsize=7.4, loc="best")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(sm, cax=colorbar_axis)
    colorbar.set_label("simulation time")

    run_event = events["runs"]["g2_L128"]
    fig.suptitle("Apparent-horizon mass and centre trajectory — g2_L128",
                 fontsize=14, fontweight="semibold", y=0.97)
    fig.text(0.5, 0.035,
             "Every plotted point has an actual horizon-shape block and passes the "
             "independent geometry, resolution, expansion, and mass gates. "
             f"Persistent validation begins at t={run_event['persistent_sequence_start_time']:.5f}.",
             ha="center", va="bottom", fontsize=8.8, color="#454b52")
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.20, top=0.82)
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {
        "g2_L128_validated_points": len(rows128),
        "g2_L128_persistent_points": len(persistent128),
        "g2_L64_persistent_points": len(persistent64),
        "final_time": final["time"],
        "final_m_irr": final["m_irr"],
        "final_r_min_cells": final["r_min_cells"],
        "final_non_sphericity": final["non_sphericity"],
        "final_spin": final["spin"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_root")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--qa-json", default=None)
    args = parser.parse_args()
    root = os.path.abspath(args.session_root)
    reduced = os.path.join(root, "reduced_data")
    outdir = os.path.abspath(args.outdir or os.path.join(root, "plots"))
    os.makedirs(outdir, exist_ok=True)
    style()

    occupancy_path = os.path.join(reduced, "particle_occupancy_g2_L128.csv")
    events_path = os.path.join(reduced, "ah_validation_events.json")
    ah128_path = os.path.join(reduced, "ah_evolution_g2_L128.csv")
    ah64_path = os.path.join(reduced, "ah_evolution_g2_L64.csv")
    occupancy = read_csv(occupancy_path)
    rows128 = read_csv(ah128_path)
    rows64 = read_csv(ah64_path)
    events = json.load(open(events_path))

    outputs = {
        "particle_occupancy_png": os.path.join(outdir, "particle_occupancy_by_amr_level.png"),
        "particle_occupancy_pdf": os.path.join(outdir, "particle_occupancy_by_amr_level.pdf"),
        "ah_evolution_png": os.path.join(outdir, "ah_mass_and_center_trajectory.png"),
        "ah_evolution_pdf": os.path.join(outdir, "ah_mass_and_center_trajectory.pdf"),
    }
    occ_qa = occupancy_figure(occupancy, events, outputs["particle_occupancy_png"],
                              outputs["particle_occupancy_pdf"])
    ah_qa = ah_figure(rows128, rows64, events, outputs["ah_evolution_png"],
                      outputs["ah_evolution_pdf"])
    qa = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sources": {path: sha256_file(path)
                    for path in [occupancy_path, events_path, ah128_path, ah64_path]},
        "occupancy_figure": occ_qa,
        "ah_figure": ah_qa,
        "outputs": {},
    }
    for label, path in outputs.items():
        qa["outputs"][label] = {
            "path": path, "size_bytes": os.path.getsize(path), "sha256": sha256_file(path)
        }
        print(f"wrote {path}")
    qa_path = os.path.abspath(args.qa_json or os.path.join(reduced, "figure_qa.json"))
    with open(qa_path, "w") as stream:
        json.dump(qa, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"wrote {qa_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
