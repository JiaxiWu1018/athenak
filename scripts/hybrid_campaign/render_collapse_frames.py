#!/usr/bin/env python3
"""Render matched-frame collapse movies for the hybrid-sampler campaign.

Two sequences (2D x-y projection; 3D view) of the matched seed-1985 pair,
R/M = 6.1 vs 5.9, on a common normalized-time axis t/P (the physically matched
clock: the two models have different surface periods). Every frame renders all
particles with finite coordinates in the nearest particle dump and annotates
stored/finite counts; once a model passes its recorded collapse onset the
panel is flagged, and excision-phase frames state how many particles have been
destroyed. Nothing is interpolated and no dump is altered.

Colors: the campaign's fixed model identities (R/M=6.1 #0072B2, R/M=5.9
#D55E00) — the same CVD-safe pair as every report figure; all text in ink.
"""
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/work1/eliasmost/jiaxiwu/nrpic_hybrid_stability_20260802")
OUT = ROOT / "movies_frames"
CASES = {
    "q6p1": {"case": "long_q6p1_stratified_antithetic_s1985",
             "period": 94.661770035504048, "collapse_over_P": 2.7290,
             "color": "#0072B2", "label": "R/M = 6.1"},
    "q5p9": {"case": "long_q5p9_stratified_antithetic_s1985",
             "period": 90.044644089281221, "collapse_over_P": 2.6200,
             "color": "#D55E00", "label": "R/M = 5.9"},
}
NPART = 528384
NFRAMES = 60
TP_MAX = 2.85           # last matched normalized time (q6p1's final dump)
LIM2D = 12.0
LIM3D = 10.0


def read_positions(path):
    blob = path.read_bytes()
    time = float(re.search(rb"time=\s*([-+0-9.eE]+)", blob).group(1))
    m = re.search(rb"POINTS\s+(\d+)\s+float", blob)
    n = int(m.group(1))
    off = blob.find(m.group(0))
    start = blob.find(b"\n", off + len(m.group(0))) + 1
    pos = np.frombuffer(blob[start:start + 12 * n], dtype=">f4").astype(float)
    pos = pos.reshape(n, 3)
    mo = b"SCALARS ptag float\nLOOKUP_TABLE default"
    off = blob.find(mo)
    start = blob.find(b"\n", off + len(mo)) + 1
    tag = np.frombuffer(blob[start:start + 4 * n], dtype=">f4").astype(np.int64)
    return time, pos, tag


def dump_index(case_dir, case):
    files = sorted(case_dir.glob(f"{case}.prtcl_all.*.part.vtk"))
    out = []
    for f in files:
        head = f.read_bytes()[:2048]
        t = float(re.search(rb"time=\s*([-+0-9.eE]+)", head).group(1))
        out.append((t, f))
    return out


def main():
    (OUT / "2d").mkdir(parents=True, exist_ok=True)
    (OUT / "3d").mkdir(parents=True, exist_ok=True)
    idx = {k: dump_index(ROOT / "runs" / v["case"] / "pvtk", v["case"])
           for k, v in CASES.items()}
    targets = np.linspace(0.0, TP_MAX, NFRAMES)
    manifest = []

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "0.75",
                         "xtick.color": "0.35", "ytick.color": "0.35"})

    for fi, tp in enumerate(targets):
        chosen = {}
        for k, v in CASES.items():
            times = np.array([t for t, _ in idx[k]])
            j = int(np.argmin(np.abs(times / v["period"] - tp)))
            chosen[k] = idx[k][j]

        rec = {"frame": fi, "target_t_over_P": float(tp)}
        # ---------- 2D ----------
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.6))
        for ax, (k, v) in zip(axes, CASES.items()):
            t, pos, tag = read_positions(chosen[k][1])
            finite = np.isfinite(pos).all(axis=1)
            stored = len(tag)
            destroyed = NPART - stored
            p = pos[finite]
            inview = (np.abs(p[:, 0]) < LIM2D) & (np.abs(p[:, 1]) < LIM2D)
            ax.scatter(p[inview, 0], p[inview, 1], s=0.4, c=v["color"],
                       alpha=0.12, linewidths=0, rasterized=True)
            ax.set_xlim(-LIM2D, LIM2D); ax.set_ylim(-LIM2D, LIM2D)
            ax.set_aspect("equal")
            ax.set_xlabel("x / M"); ax.set_ylabel("y / M")
            tp_own = t / v["period"]
            status = ""
            if destroyed > 0:
                status = f"   excision active: {destroyed:,} destroyed"
            elif tp_own >= v["collapse_over_P"]:
                status = "   collapse onset passed"
            ax.set_title(f"{v['label']}   $t/P$={tp_own:.3f}   t={t:.1f}M\n"
                         f"stored {stored:,}, finite {int(finite.sum()):,}"
                         f"{status}", fontsize=10,
                         color="0.15")
            if tp_own >= v["collapse_over_P"]:
                for s in ax.spines.values():
                    s.set_edgecolor(v["color"]); s.set_linewidth(2.0)
            rec[f"{k}_dump"] = chosen[k][1].name
            rec[f"{k}_t"] = t
            rec[f"{k}_stored"] = stored
            rec[f"{k}_finite"] = int(finite.sum())
        fig.suptitle("Hybrid-sampler collapse, matched seed 1985 — "
                     f"matched normalized time $t/P \\approx$ {tp:.3f} "
                     "(collapse onsets: 5.9 at 2.620P, 6.1 at 2.729P)",
                     fontsize=11, color="0.1")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(OUT / "2d" / f"frame_{fi:04d}.png", dpi=120)
        plt.close(fig)

        # ---------- 3D (1-in-8 pair sample, both pair members kept) ----------
        fig = plt.figure(figsize=(12.8, 6.6))
        for pi, (k, v) in enumerate(CASES.items()):
            ax = fig.add_subplot(1, 2, pi + 1, projection="3d")
            t, pos, tag = read_positions(chosen[k][1])
            finite = np.isfinite(pos).all(axis=1)
            sel = finite & ((tag // 2) % 8 == 0)
            p = pos[sel]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=0.5, c=v["color"],
                       alpha=0.18, linewidths=0)
            ax.set_xlim(-LIM3D, LIM3D); ax.set_ylim(-LIM3D, LIM3D)
            ax.set_zlim(-LIM3D, LIM3D)
            ax.set_xlabel("x/M", labelpad=-4); ax.set_ylabel("y/M", labelpad=-4)
            ax.set_zlabel("z/M", labelpad=-4)
            ax.tick_params(pad=-2, labelsize=7)
            tp_own = t / v["period"]
            ax.set_title(f"{v['label']}   $t/P$={tp_own:.3f}   "
                         f"(1-in-8 pair sample, {len(p):,} shown)",
                         fontsize=10, color="0.15")
        fig.suptitle("Hybrid-sampler collapse, matched seed 1985 — 3D view, "
                     f"$t/P \\approx$ {tp:.3f}", fontsize=11, color="0.1")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(OUT / "3d" / f"frame_{fi:04d}.png", dpi=120)
        plt.close(fig)

        manifest.append(rec)
        if fi % 10 == 0:
            print(f"frame {fi}/{NFRAMES}", flush=True)

    (OUT / "movie_frame_manifest.json").write_text(json.dumps({
        "cases": {k: v["case"] for k, v in CASES.items()},
        "collapse_onsets_over_P": {k: v["collapse_over_P"]
                                   for k, v in CASES.items()},
        "n_frames": NFRAMES, "t_over_P_max": TP_MAX,
        "sampling_3d": "pairs with (tag//2) % 8 == 0, both members kept",
        "frames": manifest}, indent=1))
    print("RENDER_DONE")


if __name__ == "__main__":
    main()
