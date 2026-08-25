#!/usr/bin/env python3
"""Render the final two-panel g2_L128 Session-002 movie from retained raw dumps.

Left: log10 of the deposited particle energy density ``tmunu_E`` on z=0.
Right: log10 of the Hamiltonian-constraint magnitude ``|con_H|`` on z=0.

Both panels use one fixed physical window and fixed normalization.  Every active
z=0 MeshBlock boundary is drawn and colored by AMR level.  The walking tracker centre
is marked.  The apparent-horizon contour is reconstructed from actual FastFlow shape
coefficients and appears only after the first run of at least three surfaces passes the
independent offline gates; no lapse contour is used.
"""
from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import datetime as dt
import glob
import hashlib
import json
import math
import os
import re
import shutil
import sys

import imageio.v2 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_ah as AH  # noqa: E402


CONFIG = None


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_one(run_dir: str, pattern: str) -> str | None:
    out = os.path.join(run_dir, "out")
    for folder in [run_dir, out, *glob.glob(os.path.join(out, "*"))]:
        paths = sorted(glob.glob(os.path.join(folder, pattern)))
        if paths:
            return paths[0]
    return None


def evaluate_persistent_surfaces(run_dir: str, dx: float, rmax_gate: float,
                                 mass_adm: float, hmean_rel: float,
                                 persist: int) -> tuple[list[dict], dict]:
    fsum = find_one(run_dir, "*.horizon_summary_0.txt")
    fshape = find_one(run_dir, "*.horizon_shape_0.txt")
    ftracker = find_one(run_dir, "*.co_0.txt")
    if not fsum or not fshape or not ftracker:
        raise FileNotFoundError("missing horizon summary, shape, or tracker file")
    summary = AH.read_summary(fsum)
    shapes = AH.read_shape(fshape)
    tracker = AH.read_tracker(ftracker)
    theta = np.linspace(1e-6, math.pi - 1e-6, 41)
    phi = np.linspace(0.0, 2 * math.pi, 80, endpoint=False)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    all_records = []
    for index, (_, time, coef) in enumerate(shapes):
        lmax = next((level for level in range(33)
                     if AH.shape_ncoef(level) == len(coef)), None)
        if lmax is None:
            raise ValueError(f"cannot infer lmax for shape {index}")
        k = int(np.argmin(np.abs(summary[:, 1] - time)))
        row = summary[k]
        area, hmean = float(row[7]), float(row[9])
        mirr = math.sqrt(area / (16 * math.pi)) if area > 0 else math.nan
        radius = AH.surface_radius(coef, lmax, th, ph)
        rmin, rmax = float(np.nanmin(radius)), float(np.nanmax(radius))
        hrel = abs(hmean) / area if area > 0 else math.inf
        finite = np.isfinite([row[2], area, row[10], row[11], rmin, rmax]).all()
        ok = (rmin > 2 * dx and rmin > 0 and rmax <= rmax_gate
              and hrel < hmean_rel and 0 < mirr <= mass_adm and finite)
        kt = int(np.argmin(np.abs(tracker[:, 0] - time)))
        all_records.append({
            "shape_index": index,
            "time": float(time),
            "m_irr": float(mirr),
            "r_min_surface": rmin,
            "r_max_surface": rmax,
            "r_min_cells": rmin / dx,
            "hrel": hrel,
            "centre": tracker[kt, 1:].tolist(),
            "coef": coef.tolist(),
            "lmax": int(lmax),
            "ok": bool(ok),
        })

    sequences = []
    start = None
    for i, record in enumerate(all_records):
        if record["ok"] and start is None:
            start = i
        if start is not None and (not record["ok"] or i == len(all_records) - 1):
            end = i if record["ok"] and i == len(all_records) - 1 else i - 1
            sequences.append((start, end))
            start = None
    if not sequences:
        raise ValueError("no gate-passing AH sequence")
    qualifying = [(start, end) for start, end in sequences
                  if end - start + 1 >= persist]
    if not qualifying:
        raise ValueError("AH sequence does not satisfy persistence requirement")
    first_persistent_start, first_persistent_end = qualifying[0]
    longest_start, longest_end = max(sequences, key=lambda ab: ab[1] - ab[0] + 1)
    longest_records = all_records[longest_start:longest_end + 1]
    persistent_records = [record for record in all_records[first_persistent_start:]
                          if record["ok"]]
    comfortable = next((record for record in persistent_records
                        if record["r_min_cells"] >= 8.0), None)

    report = os.path.join(run_dir, "analysis", "ah_report.txt")
    report_checks = {}
    if os.path.exists(report):
        text = open(report, errors="replace").read()
        expected = {
            "shape_blocks": (r"GENUINE converged finds \(shape-file blocks\):\s*(\d+)",
                             len(all_records), int),
            "passing": (r"finds passing every gate:\s*(\d+)",
                        sum(r["ok"] for r in all_records), int),
            "longest": (r"longest run of consecutive passing finds:\s*(\d+)",
                        len(longest_records), int),
            "persistent_start": (r"first of a persistent run at t =\s*([-+0-9.eE]+)",
                                 persistent_records[0]["time"], float),
        }
        for name, (pattern, value, caster) in expected.items():
            match = re.search(pattern, text)
            parsed = caster(match.group(1)) if match else None
            report_checks[name] = {
                "parsed": parsed,
                "computed": value,
                "matches": (math.isclose(parsed, value, abs_tol=5e-6)
                            if caster is float and parsed is not None else parsed == value),
            }
        if not all(item["matches"] for item in report_checks.values()):
            raise ValueError(f"movie AH audit disagrees with archived report: {report_checks}")

    event = {
        "first_fastflow_candidate_time": all_records[0]["time"],
        "first_gate_passing_time": next(r["time"] for r in all_records if r["ok"]),
        "persistent_sequence_start_time": persistent_records[0]["time"],
        "first_persistent_sequence_end_time": all_records[first_persistent_end]["time"],
        "first_persistent_sequence_length": first_persistent_end - first_persistent_start + 1,
        "longest_sequence_start_time": longest_records[0]["time"],
        "longest_sequence_end_time": longest_records[-1]["time"],
        "longest_sequence_length": len(longest_records),
        "persistent_sequence_end_time": persistent_records[-1]["time"],
        "persistent_surface_count": len(persistent_records),
        "comfortably_resolved_rmin_ge_8dx_time": comfortable["time"] if comfortable else None,
        "final_m_irr": persistent_records[-1]["m_irr"],
        "final_time": persistent_records[-1]["time"],
        "report_reconciliation": report_checks,
        "source_files": {
            "summary": {"path": fsum, "sha256": sha256_file(fsum)},
            "shape": {"path": fshape, "sha256": sha256_file(fshape)},
            "tracker": {"path": ftracker, "sha256": sha256_file(ftracker)},
        },
        "tracker": tracker.tolist(),
    }
    return persistent_records, event


def init_worker(config: dict) -> None:
    global CONFIG
    CONFIG = config
    sys.path.insert(0, os.path.join(config["athenak"], "vis", "python"))


def projected_slice(path: str, variable: str, level: int,
                    xlim: tuple[float, float], ylim: tuple[float, float]):
    import bin_convert  # noqa: E402
    raw = bin_convert.read_binary(path)
    data = bin_convert.read_binary_as_athdf(
        path, level=level, x1_min=xlim[0], x1_max=xlim[1],
        x2_min=ylim[0], x2_max=ylim[1]
    )
    field = np.asarray(data[variable], dtype=float)
    field = field[field.shape[0] // 2] if field.shape[0] > 1 else field[0]
    xf = np.asarray(data["x1f"], dtype=float)
    yf = np.asarray(data["x2f"], dtype=float)
    i0 = int(np.searchsorted(xf, xlim[0]))
    j0 = int(np.searchsorted(yf, ylim[0]))
    x = np.asarray(data["x1v"], dtype=float)[i0:i0 + field.shape[1]]
    y = np.asarray(data["x2v"], dtype=float)[j0:j0 + field.shape[0]]
    return float(data["Time"]), field, x, y, np.asarray(raw["mb_logical"]), \
        np.asarray(raw["mb_geometry"]), int(raw["nx1_out_mb"])


def block_segments(geometry: np.ndarray, logical: np.ndarray,
                   xlim: tuple[float, float], ylim: tuple[float, float]):
    by_level = {}
    for geom, log in zip(geometry, logical):
        if not (geom[4] <= 0.0 <= geom[5]):
            continue
        if geom[1] <= xlim[0] or geom[0] >= xlim[1] \
                or geom[3] <= ylim[0] or geom[2] >= ylim[1]:
            continue
        x0, x1, y0, y1 = geom[0], geom[1], geom[2], geom[3]
        segments = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                    ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
        by_level.setdefault(int(log[3]), []).extend(segments)
    return by_level


def add_boundaries(ax, segments: dict[int, list], level_min: int, level_max: int) -> None:
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=level_min, vmax=max(level_max, level_min + 1))
    for level in sorted(segments):
        seg = segments[level]
        width = 0.45 + 0.10 * (level - level_min)
        ax.add_collection(LineCollection(seg, colors=[(0, 0, 0, 0.58)],
                                         linewidths=width + 0.75, zorder=4))
        ax.add_collection(LineCollection(seg, colors=[cmap(norm(level))],
                                         linewidths=width, alpha=0.95, zorder=5))


def ah_contour(surface: dict, nphi: int = 361) -> tuple[np.ndarray, np.ndarray]:
    phi = np.linspace(0.0, 2 * math.pi, nphi)
    radius = AH.surface_radius(np.asarray(surface["coef"]), surface["lmax"],
                               np.full_like(phi, math.pi / 2), phi)
    center = np.asarray(surface["centre"])
    return center[0] + radius * np.cos(phi), center[1] + radius * np.sin(phi)


def nearest_tracker(time: float, tracker: np.ndarray) -> np.ndarray:
    index = int(np.argmin(np.abs(tracker[:, 0] - time)))
    return tracker[index, 1:]


def phase_label(time: float, event: dict) -> str:
    if time < 3.0:
        return "initial localized clump"
    if time < event["first_fastflow_candidate_time"]:
        return "gravitational contraction"
    if time < event["persistent_sequence_start_time"]:
        return "steep cusp; candidates not yet validated"
    comfortable = event["comfortably_resolved_rmin_ge_8dx_time"]
    if comfortable is None or time < comfortable:
        return "persistent validated apparent horizon"
    return "horizon growth and settling"


def render_frame(task: tuple[int, str, str]) -> dict:
    index, density_path, h_path = task
    config = CONFIG
    xlim = tuple(config["xlim"])
    ylim = tuple(config["ylim"])
    level = int(config["level"])
    td, density, x, y, logical, geometry, nxb = projected_slice(
        density_path, "tmunu_E", level, xlim, ylim
    )
    th, hfield, xh, yh, logical_h, geometry_h, _ = projected_slice(
        h_path, "con_H", level, xlim, ylim
    )
    if not math.isclose(td, th, abs_tol=2e-7):
        raise ValueError(f"frame {index}: density time {td} != constraint time {th}")
    if density.shape != hfield.shape or not np.allclose(x, xh) or not np.allclose(y, yh):
        raise ValueError(f"frame {index}: projected grids differ")
    if not np.array_equal(logical, logical_h) or not np.allclose(geometry, geometry_h):
        raise ValueError(f"frame {index}: density and constraint MeshBlocks differ")

    segments = block_segments(geometry, logical, xlim, ylim)
    levels = sorted(segments)
    finest = int(np.max(logical[:, 3]))
    dx = float(np.min((geometry[:, 1] - geometry[:, 0]) / nxb))
    tracker = np.asarray(config["event"]["tracker"])
    center = nearest_tracker(td, tracker)
    surfaces = config["surfaces"]
    surface_times = config["surface_times"]
    k = bisect.bisect_right(surface_times, td + 1e-9) - 1
    surface = surfaces[k] if k >= 0 else None

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.8), dpi=120)
    fields = [
        (np.log10(np.maximum(density, 10 ** config["dens_vmin"])),
         config["dens_vmin"], config["dens_vmax"], "inferno",
         r"deposited particle energy density  $\log_{10}(\mathrm{tmunu\_E})$"),
        (np.log10(np.maximum(np.abs(hfield), 10 ** config["h_vmin"])),
         config["h_vmin"], config["h_vmax"], "magma",
         r"Hamiltonian-constraint magnitude  $\log_{10}|H|$"),
    ]
    for ax, (field, vmin, vmax, cmap, label) in zip(axes, fields):
        image = ax.imshow(field, origin="lower", extent=(*xlim, *ylim),
                          cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
                          aspect="equal", rasterized=True)
        add_boundaries(ax, segments, levels[0], levels[-1])
        ax.plot(center[0], center[1], "+", color="black", markersize=12,
                markeredgewidth=3.0, zorder=8)
        ax.plot(center[0], center[1], "+", color="white", markersize=10,
                markeredgewidth=1.5, zorder=9)
        if surface is not None:
            ahx, ahy = ah_contour(surface)
            ax.plot(ahx, ahy, color="black", linewidth=3.2, zorder=8)
            ax.plot(ahx, ahy, color="white", linewidth=1.8, zorder=9)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(label, fontsize=11, pad=8)
        colorbar = fig.colorbar(image, ax=ax, pad=0.018, fraction=0.046)
        colorbar.ax.tick_params(labelsize=8)

    cmap_level = plt.get_cmap("viridis")
    norm_level = Normalize(vmin=levels[0], vmax=max(levels[-1], levels[0] + 1))
    handles = [Line2D([0], [0], color=cmap_level(norm_level(level)), linewidth=2.2,
                      label=f"L{level}") for level in levels]
    handles.append(Line2D([0], [0], marker="+", color="white", markeredgecolor="black",
                          markersize=9, linestyle="none", label="walking tracker centre"))
    if surface is not None:
        handles.append(Line2D([0], [0], color="white", markeredgecolor="black",
                              linewidth=2.2, label="validated AH shape"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.012))
    ah_text = "no validated AH"
    if surface is not None:
        ah_text = f"validated AH: M_irr={surface['m_irr']:.4f}"
    fig.suptitle(
        f"GI single-clump collapse — g2_L128     t={td:6.2f}     "
        f"{phase_label(td, config['event'])}\n"
        f"active leaf MeshBlocks={len(logical)}   finest=L{finest}, "
        f"dx={dx:.7f}=1/{round(1/dx)}   {ah_text}",
        fontsize=13, fontweight="semibold", y=0.985
    )
    fig.tight_layout(rect=[0.012, 0.065, 0.995, 0.92], w_pad=1.3)
    frame_path = os.path.join(config["frames_dir"], f"frame_{index:05d}.png")
    fig.savefig(frame_path, dpi=120, facecolor="white")
    plt.close(fig)
    return {
        "index": index,
        "time": td,
        "frame": frame_path,
        "density_file": density_path,
        "constraint_file": h_path,
        "active_leaf_blocks": int(len(logical)),
        "levels_in_view": levels,
        "finest_level": finest,
        "finest_dx": dx,
        "tracker_center": center.tolist(),
        "ah_shape_time": surface["time"] if surface else None,
        "ah_m_irr": surface["m_irr"] if surface else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--athenak", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-name",
                        default="jeans_s002_g2_L128_density_H_amr_AH.mp4")
    parser.add_argument("--centre", nargs=2, type=float, default=[-3.0, 0.0])
    parser.add_argument("--window", type=float, default=3.0)
    parser.add_argument("--level", type=int, default=10)
    parser.add_argument("--dx-fine", type=float, default=1 / 128)
    parser.add_argument("--rmax-gate", type=float, default=0.40)
    parser.add_argument("--mass-adm", type=float, default=0.73)
    parser.add_argument("--hmean-rel", type=float, default=0.20)
    parser.add_argument("--persist", type=int, default=3)
    parser.add_argument("--dens-vmin", type=float, default=-6.0)
    parser.add_argument("--dens-vmax", type=float, default=0.5)
    parser.add_argument("--h-vmin", type=float, default=-8.0)
    parser.add_argument("--h-vmax", type=float, default=0.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    output_root = os.path.abspath(args.output_root)
    frames_dir = os.path.join(output_root, "frames")
    qa_dir = os.path.join(output_root, "qa_frames")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(qa_dir, exist_ok=True)
    density_files = sorted(glob.glob(os.path.join(run_dir, "out", "bin", "*.tmunu.*.bin")))
    h_files = sorted(glob.glob(os.path.join(run_dir, "out", "bin", "*.con.*.bin")))
    if len(density_files) != len(h_files) or not density_files:
        raise ValueError(f"slice count mismatch density={len(density_files)} H={len(h_files)}")

    surfaces, event = evaluate_persistent_surfaces(
        run_dir, args.dx_fine, args.rmax_gate, args.mass_adm, args.hmean_rel, args.persist
    )
    if not math.isclose(event["final_m_irr"], 0.120709, abs_tol=5e-7):
        raise ValueError(f"unexpected final independently validated mass {event['final_m_irr']}")
    config = {
        "athenak": os.path.abspath(args.athenak),
        "frames_dir": frames_dir,
        "xlim": [args.centre[0] - args.window, args.centre[0] + args.window],
        "ylim": [args.centre[1] - args.window, args.centre[1] + args.window],
        "level": args.level,
        "dens_vmin": args.dens_vmin,
        "dens_vmax": args.dens_vmax,
        "h_vmin": args.h_vmin,
        "h_vmax": args.h_vmax,
        "surfaces": surfaces,
        "surface_times": [surface["time"] for surface in surfaces],
        "event": event,
    }
    tasks = list(zip(range(len(density_files)), density_files, h_files))
    print(f"rendering {len(tasks)} paired frames with {args.processes} processes")
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.processes, initializer=init_worker, initargs=(config,)) as pool:
        metadata = list(pool.map(render_frame, tasks))
    metadata.sort(key=lambda row: row["index"])

    movie_path = os.path.join(output_root, args.output_name)
    with iio.get_writer(movie_path, fps=args.fps, codec="libx264", macro_block_size=None,
                        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                       "-crf", "18"]) as writer:
        for row in metadata:
            writer.append_data(iio.imread(row["frame"]))

    reader = iio.get_reader(movie_path)
    decoded = 0
    first_shape = None
    for frame in reader:
        decoded += 1
        if first_shape is None:
            first_shape = list(frame.shape)
    movie_meta = reader.get_meta_data()
    reader.close()
    if decoded != len(metadata):
        raise ValueError(f"decoded {decoded} frames but rendered {len(metadata)}")

    target_times = {
        "initial": metadata[0]["time"],
        "contracting": 6.0,
        "first_persistent_ah": event["persistent_sequence_start_time"],
        "final": metadata[-1]["time"],
    }
    qa_frames = {}
    for label, target in target_times.items():
        row = min(metadata, key=lambda item: abs(item["time"] - target))
        destination = os.path.join(qa_dir, f"{label}_t{row['time']:.2f}.png")
        shutil.copy2(row["frame"], destination)
        qa_frames[label] = destination

    input_manifest = []
    for density_path, h_path in zip(density_files, h_files):
        input_manifest.append({
            "density": {"path": density_path, "size_bytes": os.path.getsize(density_path),
                        "sha256": sha256_file(density_path)},
            "constraint": {"path": h_path, "size_bytes": os.path.getsize(h_path),
                           "sha256": sha256_file(h_path)},
        })
    qa = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_dir": run_dir,
        "script": {"path": os.path.abspath(__file__), "sha256": sha256_file(__file__)},
        "render_contract": {
            "plane": "z=0",
            "window": {"x": config["xlim"], "y": config["ylim"]},
            "density_field": "tmunu_E (deposited particle energy density)",
            "density_log10_limits": [args.dens_vmin, args.dens_vmax],
            "constraint_field": "con_H (Hamiltonian constraint)",
            "constraint_log10_abs_limits": [args.h_vmin, args.h_vmax],
            "mesh_boundaries": "all active z=0 leaf MeshBlocks in window, colored by level",
            "center_marker": "walking co_0 tracker",
            "ah_overlay": "reconstructed shape coefficients after the first persistence "
                          "gate, with every retained surface independently valid; never "
                          "a lapse contour",
        },
        "ah_validation": {key: value for key, value in event.items() if key != "tracker"},
        "input_manifest": input_manifest,
        "frames": metadata,
        "qa_frames": qa_frames,
        "movie": {
            "path": movie_path,
            "size_bytes": os.path.getsize(movie_path),
            "sha256": sha256_file(movie_path),
            "fps_requested": args.fps,
            "rendered_frames": len(metadata),
            "decoded_frames": decoded,
            "decoded_frame_shape": first_shape,
            "imageio_metadata": {key: value for key, value in movie_meta.items()
                                 if isinstance(value, (str, int, float, bool, list, tuple,
                                                       type(None)))},
        },
    }
    qa_path = os.path.join(output_root, "movie_qa.json")
    with open(qa_path, "w") as stream:
        json.dump(qa, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"wrote {movie_path}: {decoded} decoded frames at {args.fps} fps")
    print(f"wrote {qa_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
