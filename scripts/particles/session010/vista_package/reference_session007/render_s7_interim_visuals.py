#!/usr/bin/env python3
"""Create presentation-ready Session-007 matter and numerical-health visuals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import types
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


COLORS = {
    0: "#f28e2b",
    1: "#21b6c7",
    2: "#7ac943",
    "envelope": "#b7c0cb",
    "total": "#252a34",
    "H": "#355f8d",
    "M": "#d17a22",
    "Z": "#6f4e7c",
}
GROUPS = (
    ("Envelope", 0, 3_000_000, COLORS["envelope"], 40_000),
    ("Clump A", 3_000_000, 4_000_000, COLORS[0], 25_000),
    ("Clump B", 4_000_000, 5_000_000, COLORS[1], 25_000),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def vtk_metadata(path: Path) -> tuple[float, int, int]:
    with path.open("rb") as stream:
        header = stream.read(1024)
    tmatch = re.search(rb"time=\s*([-+0-9.eE]+)", header)
    cmatch = re.search(rb"cycle=\s*(\d+)", header)
    nmatch = re.search(rb"POINTS\s+(\d+)\s+float", header)
    if not (tmatch and cmatch and nmatch):
        raise RuntimeError(f"invalid particle VTK header: {path}")
    return float(tmatch.group(1)), int(cmatch.group(1)), int(nmatch.group(1))


def read_block(raw: bytes, marker: bytes, count: int) -> np.ndarray:
    start = raw.find(marker)
    if start < 0:
        raise RuntimeError(f"missing VTK marker {marker!r}")
    start = raw.find(b"\n", start + len(marker)) + 1
    if start <= 0:
        raise RuntimeError(f"malformed VTK marker {marker!r}")
    return np.frombuffer(raw, dtype=">f4", count=count, offset=start)


def read_particles(path: Path):
    raw = path.read_bytes()
    tmatch = re.search(rb"time=\s*([-+0-9.eE]+)", raw[:1024])
    cmatch = re.search(rb"cycle=\s*(\d+)", raw[:1024])
    nmatch = re.search(rb"POINTS\s+(\d+)\s+float", raw[:1024])
    if not (tmatch and cmatch and nmatch):
        raise RuntimeError(f"invalid particle VTK file: {path}")
    count = int(nmatch.group(1))
    pos = read_block(raw, nmatch.group(0), 3 * count).reshape(count, 3)
    mass = read_block(raw, b"SCALARS prtcl_mass float\nLOOKUP_TABLE default", count)
    tags = read_block(raw, b"SCALARS ptag float\nLOOKUP_TABLE default", count).astype(
        np.int64
    )
    return (
        float(tmatch.group(1)), int(cmatch.group(1)), pos, mass, tags,
        hashlib.sha256(raw).hexdigest(),
    )


def mix64(values: np.ndarray) -> np.ndarray:
    mixed = values.astype(np.uint64).copy()
    mixed ^= mixed >> np.uint64(30)
    mixed *= np.uint64(0xBF58476D1CE4E5B9)
    mixed ^= mixed >> np.uint64(27)
    mixed *= np.uint64(0x94D049BB133111EB)
    mixed ^= mixed >> np.uint64(31)
    return mixed


def choose_tags(tags: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    selected = []
    details = {}
    for label, low, high, _, target in GROUPS:
        available = tags[(tags >= low) & (tags < high)]
        take = min(target, len(available))
        order = np.argpartition(mix64(available), take - 1)[:take]
        values = np.sort(available[order])
        selected.append(values)
        details[label] = {
            "tag_range": [low, high],
            "available": int(len(available)),
            "selected": int(len(values)),
            "selected_tag_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }
    return np.concatenate(selected), details


def selected_mask(tags: np.ndarray, selected: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(selected, tags)
    valid = positions < len(selected)
    result = np.zeros(len(tags), dtype=bool)
    result[valid] = selected[positions[valid]] == tags[valid]
    return result


def deposit_xy(pos: np.ndarray, mass: np.ndarray, limit: float, cells: int) -> np.ndarray:
    dx = 2.0 * limit / cells
    ux = (pos[:, 0].astype(np.float64) + limit) / dx - 0.5
    uy = (pos[:, 1].astype(np.float64) + limit) / dx - 0.5
    ix, iy = np.floor(ux).astype(np.int64), np.floor(uy).astype(np.int64)
    fx, fy = ux - ix, uy - iy
    weights = mass.astype(np.float64, copy=False)
    result = np.zeros(cells * cells, dtype=np.float64)
    for ox in (0, 1):
        gx, wx = ix + ox, fx if ox else 1.0 - fx
        for oy in (0, 1):
            gy, wy = iy + oy, fy if oy else 1.0 - fy
            keep = (gx >= 0) & (gx < cells) & (gy >= 0) & (gy < cells)
            flat = gy[keep] * cells + gx[keep]
            result += np.bincount(
                flat, weights=(weights * wx * wy)[keep], minlength=cells * cells
            )
    return (result.reshape(cells, cells) / dx**2).astype(np.float32)


def deduplicated_files(paths: Iterable[Path], metadata) -> list[Path]:
    by_time: dict[float, Path] = {}
    for path in paths:
        by_time[round(metadata(path)[0], 10)] = path
    return [by_time[key] for key in sorted(by_time)]


def read_trackers(paths: list[Path]) -> list[dict[str, float]]:
    by_time: dict[float, dict[str, float]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.split()
                row = {
                    "cycle": int(fields[0]), "time": float(fields[1]),
                    "x": float(fields[2]), "y": float(fields[3]), "z": float(fields[4]),
                    "lapse_min": float(fields[15]),
                    "horizon_tracking": int(fields[16]),
                }
                by_time[round(row["time"], 10)] = row
    return [by_time[key] for key in sorted(by_time)]


def read_horizons(paths_by_id: dict[int, list[Path]]):
    result: dict[int, list[dict[str, float]]] = {}
    provenance: dict[str, Any] = {}
    for horizon_id, paths in paths_by_id.items():
        by_time: dict[float, dict[str, float]] = {}
        provenance[str(horizon_id)] = []
        for path in paths:
            provenance[str(horizon_id)].append({
                "path": str(path.resolve()), "sha256": sha256_file(path)
            })
            with path.open("r", encoding="utf-8", newline="") as stream:
                for raw in csv.DictReader(stream):
                    try:
                        if int(raw["published_this_candidate"]) != 1:
                            continue
                        if int(raw["quality_persist_ok"]) != 1:
                            continue
                        rmin, rmax = float(raw["rmin"]), float(raw["rmax"])
                        if not (0.0 < rmin <= rmax):
                            continue
                        row = {
                            "cycle": int(raw["cycle"]), "time": float(raw["time"]),
                            "x": float(raw["center_x"]), "y": float(raw["center_y"]),
                            "z": float(raw["center_z"]), "rmin": rmin, "rmax": rmax,
                            "radius": 0.5 * (rmin + rmax), "area": float(raw["area"]),
                            "hrel": float(raw["hrel"]),
                            "rmin_cells": float(raw["rmin_cells"]),
                        }
                    except (KeyError, TypeError, ValueError):
                        continue
                    by_time[round(row["time"], 10)] = row
        rows = [by_time[key] for key in sorted(by_time)]
        result[horizon_id] = rows
        provenance[str(horizon_id)].append({
            "accepted_records": len(rows),
            "first_time": rows[0]["time"] if rows else None,
            "last_time": rows[-1]["time"] if rows else None,
        })
    return result, provenance


def at_time(rows: list[dict[str, float]], time_value: float):
    if not rows or time_value < rows[0]["time"] - 1.0e-8:
        return None
    if time_value > rows[-1]["time"] + 0.13:
        return None
    times = np.fromiter((row["time"] for row in rows), dtype=np.float64)
    index = int(np.searchsorted(times, time_value + 1.0e-8, side="right") - 1)
    return rows[max(index, 0)]


def tracker_at(rows: list[dict[str, float]], time_value: float):
    times = np.fromiter((row["time"] for row in rows), dtype=np.float64)
    index = int(np.searchsorted(times, time_value + 1.0e-8, side="right") - 1)
    return rows[max(index, 0)]


def draw_horizons(ax, horizons, time_value: float, dark: bool) -> list[int]:
    visible = []
    for horizon_id, rows in horizons.items():
        row = at_time(rows, time_value)
        if row is None:
            continue
        color = COLORS[horizon_id]
        linewidth = 2.5 if horizon_id == 2 else 2.0
        ax.add_patch(Circle(
            (row["x"], row["y"]), row["radius"], fill=False, edgecolor=color,
            linewidth=linewidth, linestyle="-" if horizon_id == 2 else "--", zorder=12,
        ))
        ax.scatter([row["x"]], [row["y"]], marker="+", s=70, linewidths=1.8,
                   color=color, zorder=13)
        visible.append(horizon_id)
    return visible


def draw_trackers(ax, trackers, time_value: float, limit: float) -> None:
    for tracker_id, rows in trackers.items():
        history = [row for row in rows if row["time"] <= time_value + 1.0e-8]
        if not history:
            continue
        stride = max(1, len(history) // 500)
        ax.plot([r["x"] for r in history[::stride]], [r["y"] for r in history[::stride]],
                color=COLORS[tracker_id], linewidth=1.5, alpha=0.8, zorder=9)
        current = history[-1]
        ax.scatter([current["x"]], [current["y"]], s=46, facecolors="none",
                   edgecolors=COLORS[tracker_id], linewidths=1.8, zorder=11)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)


def set_dark_axes(ax, limit: float) -> None:
    ax.set_facecolor("#080b10")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x/M$", fontsize=16, color="white")
    ax.set_ylabel(r"$y/M$", fontsize=16, color="white")
    ax.tick_params(colors="#e4e8ed", labelsize=13)
    for spine in ax.spines.values():
        spine.set_color("#89939e")
    ax.grid(color="white", alpha=0.08, linewidth=0.6)


def image_qa(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        return {
            "width": image.width, "height": image.height,
            "stddev": float(pixels.std()), "nonblank": bool(pixels.std() > 5.0),
        }


def encode_movie(frame_pattern: Path, movie: Path, fps: int, frame_count: int):
    movie.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(frame_pattern), "-c:v", "mpeg4", "-q:v", "2",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(movie),
    ], check=True)
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,nb_read_frames,duration",
        "-of", "json", str(movie),
    ], text=True))["streams"][0]
    if int(probe["nb_read_frames"]) != frame_count:
        raise RuntimeError(f"movie has {probe['nb_read_frames']} frames, expected {frame_count}")
    return probe


def prepare_particle_cache(paths: list[Path], cache_dir: Path, central_limit: float,
                           cells: int):
    cache_dir.mkdir(parents=True, exist_ok=True)
    selection_path = cache_dir.parent / "selected_tags.npy"
    selection_json = cache_dir.parent / "selection.json"
    if selection_path.exists() and selection_json.exists():
        selected = np.load(selection_path)
        selection = json.loads(selection_json.read_text(encoding="utf-8"))
    else:
        _, _, _, _, tags, _ = read_particles(paths[0])
        selected, selection = choose_tags(tags)
        np.save(selection_path, selected)
        write_json(selection_json, selection)

    records = []
    scales = []
    for index, source in enumerate(paths):
        cache = cache_dir / f"particle_{index:05d}.npz"
        if not cache.exists():
            time_value, cycle, pos, mass, tags, digest = read_particles(source)
            density = deposit_xy(pos, mass, central_limit, cells)
            use = selected_mask(tags, selected)
            sample_pos, sample_tags = pos[use], tags[use]
            sample_group = np.full(len(sample_tags), 255, dtype=np.uint8)
            group_count, group_mass = [], []
            for group_id, (_, low, high, _, _) in enumerate(GROUPS):
                group = (tags >= low) & (tags < high)
                group_count.append(int(group.sum()))
                group_mass.append(float(mass[group].astype(np.float64).sum()))
                sample_group[(sample_tags >= low) & (sample_tags < high)] = group_id
            np.savez_compressed(
                cache, density=density,
                x=sample_pos[:, 0].astype(np.float32),
                y=sample_pos[:, 1].astype(np.float32),
                group=sample_group, time=np.float64(time_value), cycle=np.int64(cycle),
                total_particles=np.int64(len(tags)),
                group_count=np.asarray(group_count, dtype=np.int64),
                group_mass=np.asarray(group_mass, dtype=np.float64),
                source_sha256=np.asarray(digest),
            )
        with np.load(cache) as data:
            positive = data["density"][data["density"] > 0.0]
            if not len(positive):
                raise RuntimeError(f"empty particle density in {source}")
            scales.append((float(np.quantile(positive, 0.02)),
                           float(np.quantile(positive, 0.998))))
            records.append({
                "frame": index, "time": float(data["time"]),
                "cycle": int(data["cycle"]),
                "particles": int(data["total_particles"]),
                "group_count": [int(v) for v in data["group_count"]],
                "group_mass": [float(v) for v in data["group_mass"]],
                "path": str(source.resolve()), "bytes": source.stat().st_size,
                "sha256": str(data["source_sha256"]),
            })
        print(f"particle cache {index + 1}/{len(paths)}", flush=True)
    vmin = min(row[0] for row in scales)
    vmax = max(row[1] for row in scales)
    if not 0.0 < vmin < vmax:
        raise RuntimeError(f"invalid density scale {vmin}, {vmax}")
    return records, selection, vmin, vmax


def render_matter_movie(paths: list[Path], cache_dir: Path, frame_dir: Path,
                        trackers, horizons, central_limit: float, wide_limit: float,
                        vmin: float, vmax: float, fps: int, movie: Path):
    frame_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#080b10")
    for index, _ in enumerate(paths):
        frame = frame_dir / f"matter_{index:05d}.png"
        if frame.exists():
            print(f"matter frame {index + 1}/{len(paths)} cached", flush=True)
            continue
        with np.load(cache_dir / f"particle_{index:05d}.npz") as data:
            density = data["density"]
            x, y, group = data["x"], data["y"], data["group"]
            time_value = float(data["time"])
            particles = int(data["total_particles"])
        fig = plt.figure(figsize=(16, 7.5), dpi=120, facecolor="#080b10")
        gs = fig.add_gridspec(1, 3, width_ratios=(1, 0.035, 1), wspace=0.15)
        ax0, cax, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
        image = ax0.imshow(
            np.ma.masked_less_equal(density, 0.0), origin="lower",
            extent=(-central_limit, central_limit, -central_limit, central_limit),
            interpolation="bilinear", cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),
            rasterized=True,
        )
        colorbar = fig.colorbar(image, cax=cax)
        colorbar.set_label(r"$\Sigma_\mathrm{p}$", color="white",
                           fontsize=18, labelpad=9)
        colorbar.ax.tick_params(colors="white", labelsize=12)
        colorbar.outline.set_edgecolor("#89939e")
        for group_id, (label, _, _, color, _) in enumerate(GROUPS):
            use = group == group_id
            ax1.scatter(x[use], y[use], s=0.7 if group_id == 0 else 1.4,
                        alpha=0.25 if group_id == 0 else 0.75, color=color,
                        linewidths=0, rasterized=True, label=label)
        draw_trackers(ax0, trackers, time_value, central_limit)
        draw_trackers(ax1, trackers, time_value, wide_limit)
        visible = draw_horizons(ax0, horizons, time_value, True)
        draw_horizons(ax1, horizons, time_value, True)
        set_dark_axes(ax0, central_limit)
        set_dark_axes(ax1, wide_limit)
        ax0.set_title("Central matter", color="white", fontsize=18, pad=10)
        ax1.set_title("Tagged particles", color="white", fontsize=18, pad=10)
        handles, labels = ax1.get_legend_handles_labels()
        for hid in visible:
            handles.append(Line2D([0], [0], color=COLORS[hid], linewidth=2.2,
                                  linestyle="-" if hid == 2 else "--"))
            labels.append("Common AH" if hid == 2 else f"AH {hid}")
        ax1.legend(handles, labels, loc="upper right", frameon=False, fontsize=12,
                   labelcolor="white", markerscale=4)
        fig.suptitle(rf"Matter evolution     $t/M={time_value:.2f}$",
                     color="white", fontsize=22, y=0.985)
        ax0.text(0.025, 0.97, rf"$N_\mathrm{{p}}={particles:,}$",
                 transform=ax0.transAxes, ha="left", va="top", color="white", fontsize=14)
        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.105, top=0.91)
        fig.savefig(frame, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"matter frame {index + 1}/{len(paths)}", flush=True)
    probe = encode_movie(frame_dir / "matter_%05d.png", movie, fps, len(paths))
    qa = {
        "first": image_qa(frame_dir / "matter_00000.png"),
        "middle": image_qa(frame_dir / f"matter_{len(paths)//2:05d}.png"),
        "last": image_qa(frame_dir / f"matter_{len(paths)-1:05d}.png"),
        "ffprobe": probe,
    }
    if not all(qa[key]["nonblank"] for key in ("first", "middle", "last")):
        raise RuntimeError("blank matter movie frame")
    return qa


def read_history(paths: list[Path]):
    by_time = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip() or line.startswith("#"):
                    continue
                fields = [float(v) for v in line.split()]
                if not all(math.isfinite(v) for v in fields):
                    raise RuntimeError(f"nonfinite history row in {path}")
                row = dict(zip(("time", "dt", "C", "H", "M", "Z", "Mx", "My",
                                "Mz", "Theta", "Volume"), fields))
                by_time[round(row["time"], 8)] = row
    return [by_time[key] for key in sorted(by_time)]


def accepted_horizon_physics(horizons, summaries_by_id):
    result = {}
    for horizon_id, paths in summaries_by_id.items():
        accepted = {round(row["time"], 4): row for row in horizons[horizon_id]}
        summary = {}
        for path in paths:
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip() or line.startswith("#"):
                        continue
                    fields = line.split()
                    time_value = float(fields[1])
                    key = round(time_value, 4)
                    if key not in accepted:
                        continue
                    values = [float(v) for v in fields]
                    if not all(math.isfinite(v) for v in values):
                        continue
                    consumer = accepted[key]
                    summary[key] = {
                        "horizon": horizon_id, "time": consumer["time"],
                        "cycle": consumer["cycle"], "M_BH": values[2],
                        "M_irr": values[12], "chi": values[13], "area": values[7],
                        "Sx": values[3], "Sy": values[4], "Sz": values[5],
                        "center_x": consumer["x"], "center_y": consumer["y"],
                        "rmin_cells": consumer["rmin_cells"], "hrel": consumer["hrel"],
                    }
        result[horizon_id] = [summary[key] for key in sorted(summary)]
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_static_figure(outdir: Path, particle_records, trackers, horizon_physics,
                       history, event_times):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 14, "axes.titlesize": 17,
        "axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 11.5, "axes.linewidth": 1.0,
    })
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), constrained_layout=True)
    ax = axes[0, 0]
    tracker_styles = {0: "-", 1: "--"}
    for tid in (0, 1):
        ax.plot([r["x"] for r in trackers[tid]], [r["y"] for r in trackers[tid]],
                color=COLORS[tid], lw=2.0, ls=tracker_styles[tid], label=f"Tracker {tid}")
        ax.scatter(trackers[tid][0]["x"], trackers[tid][0]["y"], s=45,
                   color=COLORS[tid], marker="o")
        ax.scatter(trackers[tid][-1]["x"], trackers[tid][-1]["y"], s=70,
                   color=COLORS[tid], marker="x", linewidths=2)
    ax.set(xlabel=r"$x/M$", ylabel=r"$y/M$", title="Tracker trajectories")
    ax.set_aspect("equal")
    ax.legend(frameon=False)

    paired = []
    right = {round(r["time"], 8): r for r in trackers[1]}
    for left in trackers[0]:
        other = right.get(round(left["time"], 8))
        if other:
            paired.append((left["time"], math.dist(
                (left["x"], left["y"], left["z"]),
                (other["x"], other["y"], other["z"]))))
    ax = axes[0, 1]
    ax.plot([r[0] for r in paired], [r[1] for r in paired], color="#355f8d", lw=2.2)
    event_styles = {0: "--", 1: ":", 2: "-"}
    event_labels = {0: "AH 0", 1: "AH 1", 2: "Common AH"}
    for hid in (0, 1, 2):
        ax.axvline(event_times[hid], color=COLORS[hid], lw=1.4,
                   ls=event_styles[hid], alpha=0.9, label=event_labels[hid])
    ax.set(xlabel=r"$t/M$", ylabel=r"$d/M$", title="Tracker separation")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="lower left")

    ax = axes[0, 2]
    times = np.asarray([r["time"] for r in particle_records])
    masses = np.asarray([r["group_mass"] for r in particle_records])
    group_styles = {0: "-", 1: "--", 2: "-."}
    for gid, (label, _, _, color, _) in enumerate(GROUPS):
        ax.plot(times, masses[:, gid] / masses[0, gid], lw=2.1, color=color,
                ls=group_styles[gid], label=label)
    ax.set(xlabel=r"$t/M$", ylabel=r"$M_\mathrm{p}(t)/M_\mathrm{p}(0)$",
           title="Surviving particle rest mass")
    ax.set_ylim(-0.02, 1.03)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    horizon_styles = {0: "--", 1: ":", 2: "-"}
    for hid, label in ((0, "AH 0"), (1, "AH 1"), (2, "Common AH")):
        rows = horizon_physics[hid]
        ax.plot([r["time"] for r in rows], [r["M_BH"] for r in rows],
                lw=2.1, color=COLORS[hid], ls=horizon_styles[hid], label=label)
    ax.set(xlabel=r"$t/M$", ylabel=r"$M_\mathrm{BH}/M$", title="Horizon mass")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for hid, label in ((0, "AH 0"), (1, "AH 1"), (2, "Common AH")):
        rows = horizon_physics[hid]
        ax.plot([r["time"] for r in rows], [r["chi"] for r in rows],
                lw=2.1, color=COLORS[hid], ls=horizon_styles[hid], label=label)
    ax.set(xlabel=r"$t/M$", ylabel=r"$\chi_\mathrm{BH}$", title="Horizon spin")
    ax.set_ylim(bottom=0)

    ax = axes[1, 2]
    ht = np.asarray([r["time"] for r in history])
    for key, label in (("H", r"$\|\mathcal{H}\|_2$"),
                       ("M", r"$\|\mathcal{M}\|_2$"),
                       ("Z", r"$\|Z\|_2$")):
        values = np.asarray([abs(r[key]) for r in history])
        ax.semilogy(ht[values > 0], values[values > 0], lw=1.9,
                    color=COLORS[key], label=label)
    ax.set(xlabel=r"$t/M$", ylabel="Norm", title="Constraint history")
    ax.legend(frameon=False)

    for ax in axes.flat:
        ax.grid(True, color="#d8dde3", linewidth=0.7, alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(left=0 if ax not in (axes[0, 0],) else None)
    png, pdf = outdir / "s7_interim_diagnostics.png", outdir / "s7_interim_diagnostics.pdf"
    fig.savefig(png, dpi=200, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


def render_constraint_movie(bin_convert_path: Path, con_files: list[Path], frame_dir: Path,
                            trackers, horizons, limit: float, fps: int, movie: Path):
    sys.path.insert(0, str(bin_convert_path))
    # bin_convert imports h5py for optional ATHDF writing.  This workflow only uses
    # read_binary(), so keep that unrelated optional dependency from blocking it.
    if "h5py" not in sys.modules:
        sys.modules["h5py"] = types.ModuleType("h5py")
    import bin_convert  # type: ignore  # noqa: E402

    frame_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    metadata = []
    for index, path in enumerate(con_files):
        data = bin_convert.read_binary(str(path))
        central = []
        for geometry, values in zip(data["mb_geometry"], data["mb_data"]["con_H"]):
            x0, x1, y0, y1 = geometry[:4]
            if x1 > -limit and x0 < limit and y1 > -limit and y0 < limit:
                raw = np.abs(np.asarray(values)[0].ravel())
                central.append(raw)
                samples.append(raw[::32])
        metadata.append({"path": str(path.resolve()), "time": float(data["time"]),
                         "cycle": int(data["cycle"]), "central_blocks": len(central)})
        print(f"constraint scale {index + 1}/{len(con_files)}", flush=True)
    values = np.concatenate(samples)
    values = values[np.isfinite(values) & (values > 0)]
    vmin = max(float(np.quantile(values, 0.01)), 1.0e-12)
    vmax = float(np.quantile(values, 0.999))
    log_norm = Normalize(np.log10(vmin), np.log10(vmax))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#080b10")
    for index, path in enumerate(con_files):
        frame = frame_dir / f"constraint_{index:05d}.png"
        if frame.exists():
            print(f"constraint frame {index + 1}/{len(con_files)} cached", flush=True)
            continue
        data = bin_convert.read_binary(str(path))
        time_value = float(data["time"])
        fig, ax = plt.subplots(figsize=(8.6, 8.0), dpi=160, facecolor="#080b10")
        ax.set_facecolor("#080b10")
        for geometry, raw in zip(data["mb_geometry"], data["mb_data"]["con_H"]):
            x0, x1, y0, y1 = geometry[:4]
            if not (x1 > -limit and x0 < limit and y1 > -limit and y0 < limit):
                continue
            plane = np.log10(np.maximum(np.abs(np.asarray(raw)[0]), vmin))
            ax.imshow(plane, origin="lower", extent=(x0, x1, y0, y1), cmap=cmap,
                      norm=log_norm, interpolation="nearest", rasterized=True)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor="white", linewidth=0.25, alpha=0.33))
        draw_trackers(ax, trackers, time_value, limit)
        visible = draw_horizons(ax, horizons, time_value, True)
        set_dark_axes(ax, limit)
        ax.set_title(rf"Hamiltonian constraint     $t/M={time_value:.2f}$",
                     color="white", fontsize=19, pad=12)
        sm = ScalarMappable(norm=log_norm, cmap=cmap)
        colorbar = fig.colorbar(sm, ax=ax, pad=0.025, fraction=0.047)
        colorbar.set_label(r"$\log_{10}|\mathcal{H}|$", color="white", fontsize=16)
        colorbar.ax.tick_params(colors="white", labelsize=12)
        colorbar.outline.set_edgecolor("#89939e")
        if visible:
            handles = [Line2D([0], [0], color=COLORS[h], lw=2,
                              ls="-" if h == 2 else "--") for h in visible]
            labels = ["Common AH" if h == 2 else f"AH {h}" for h in visible]
            ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=12,
                      labelcolor="white")
        fig.subplots_adjust(left=0.10, right=0.91, bottom=0.10, top=0.91)
        fig.savefig(frame, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"constraint frame {index + 1}/{len(con_files)}", flush=True)
    probe = encode_movie(frame_dir / "constraint_%05d.png", movie, fps, len(con_files))
    qa = {
        "first": image_qa(frame_dir / "constraint_00000.png"),
        "middle": image_qa(frame_dir / f"constraint_{len(con_files)//2:05d}.png"),
        "last": image_qa(frame_dir / f"constraint_{len(con_files)-1:05d}.png"),
        "ffprobe": probe,
    }
    if not all(qa[key]["nonblank"] for key in ("first", "middle", "last")):
        raise RuntimeError("blank constraint movie frame")
    return metadata, {"vmin": vmin, "vmax": vmax}, qa


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-out", required=True, type=Path)
    parser.add_argument("--continuation-out", required=True, type=Path)
    parser.add_argument("--original-sidecars", required=True, type=Path)
    parser.add_argument("--bin-convert-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--cells", type=int, default=512)
    parser.add_argument("--central-limit", type=float, default=8.0)
    parser.add_argument("--wide-limit", type=float, default=32.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    particle_candidates = list((args.original_out / "pvtk").glob("*.part.vtk"))
    particle_candidates += list((args.continuation_out / "pvtk").glob("*.part.vtk"))
    particles = deduplicated_files(particle_candidates, vtk_metadata)
    if len(particles) != 257:
        raise RuntimeError(f"expected 257 particle frames, found {len(particles)}")

    horizon_paths = {
        0: [args.original_sidecars / "s7_two_clump_ccw.horizon_consumer_0.csv",
            args.continuation_out / "s7_two_clump_ccw.horizon_consumer_0.csv"],
        1: [args.original_sidecars / "s7_two_clump_ccw.horizon_consumer_1.csv",
            args.continuation_out / "s7_two_clump_ccw.horizon_consumer_1.csv"],
        2: [args.continuation_out / "s7_two_clump_ccw.horizon_consumer_2.csv"],
    }
    horizons, horizon_provenance = read_horizons(horizon_paths)
    trackers = {
        0: read_trackers([args.original_sidecars / "s7_two_clump_ccw.co_0.txt",
                          args.continuation_out / "s7_two_clump_ccw.co_0.txt"]),
        1: read_trackers([args.original_sidecars / "s7_two_clump_ccw.co_1.txt",
                          args.continuation_out / "s7_two_clump_ccw.co_1.txt"]),
    }
    event_times = {hid: horizons[hid][0]["time"] for hid in (0, 1, 2)}

    cache_dir = args.output_dir / "cache"
    particle_records, selection, vmin, vmax = prepare_particle_cache(
        particles, cache_dir, args.central_limit, args.cells
    )
    matter_movie = args.output_dir / "s7_matter_evolution_t0_t64.mp4"
    matter_qa = render_matter_movie(
        particles, cache_dir, args.output_dir / "matter_frames", trackers, horizons,
        args.central_limit, args.wide_limit, vmin, vmax, args.fps, matter_movie,
    )

    history = read_history([
        args.original_out / "s7_two_clump_ccw.z4c.user.hst",
        args.continuation_out / "s7_two_clump_ccw.z4c.user.hst",
    ])
    summaries = {
        0: [args.original_sidecars / "s7_two_clump_ccw.horizon_summary_0.txt",
            args.continuation_out / "s7_two_clump_ccw.horizon_summary_0.txt"],
        1: [args.original_sidecars / "s7_two_clump_ccw.horizon_summary_1.txt",
            args.continuation_out / "s7_two_clump_ccw.horizon_summary_1.txt"],
        2: [args.continuation_out / "s7_two_clump_ccw.horizon_summary_2.txt"],
    }
    horizon_physics = accepted_horizon_physics(horizons, summaries)
    write_csv(args.output_dir / "particle_evolution.csv", particle_records)
    write_csv(args.output_dir / "tracker_0.csv", trackers[0])
    write_csv(args.output_dir / "tracker_1.csv", trackers[1])
    write_csv(args.output_dir / "constraint_history.csv", history)
    write_csv(args.output_dir / "accepted_horizons.csv",
              [row for hid in (0, 1, 2) for row in horizon_physics[hid]])
    static_png, static_pdf = make_static_figure(
        args.output_dir, particle_records, trackers, horizon_physics, history, event_times
    )

    con_by_index = {}
    for root in (args.original_out / "bin", args.continuation_out / "bin"):
        for path in root.glob("*.con_xy.*.bin"):
            match = re.search(r"\.con_xy\.(\d+)\.bin$", path.name)
            if match:
                con_by_index[int(match.group(1))] = path
    chosen = sorted(set(range(0, 641, 4)) | {150, 151, 640})
    con_files = [con_by_index[index] for index in chosen]
    if len(con_files) != len(chosen):
        raise RuntimeError("missing constraint-movie frames")
    constraint_movie = args.output_dir / "s7_constraint_amr_t0_t64.mp4"
    con_metadata, con_scale, constraint_qa = render_constraint_movie(
        args.bin_convert_dir, con_files, args.output_dir / "constraint_frames",
        trackers, horizons, args.central_limit, args.fps, constraint_movie,
    )

    manifest = {
        "schema_version": 1,
        "particle_frames": len(particles),
        "constraint_frames": len(con_files),
        "time_range": [particle_records[0]["time"], particle_records[-1]["time"]],
        "event_times": event_times,
        "particle_selection": selection,
        "density_scale": {"type": "LogNorm", "vmin": vmin, "vmax": vmax},
        "constraint_scale": {"type": "log10_abs", **con_scale},
        "horizons": horizon_provenance,
        "matter_movie": {"path": str(matter_movie.resolve()),
                         "bytes": matter_movie.stat().st_size,
                         "sha256": sha256_file(matter_movie), "qa": matter_qa},
        "constraint_movie": {"path": str(constraint_movie.resolve()),
                              "bytes": constraint_movie.stat().st_size,
                              "sha256": sha256_file(constraint_movie), "qa": constraint_qa},
        "diagnostic_png": {"path": str(static_png.resolve()),
                           "bytes": static_png.stat().st_size,
                           "sha256": sha256_file(static_png), "qa": image_qa(static_png)},
        "diagnostic_pdf": {"path": str(static_pdf.resolve()),
                           "bytes": static_pdf.stat().st_size,
                           "sha256": sha256_file(static_pdf)},
        "constraint_sources": con_metadata,
    }
    write_json(args.output_dir / "visual_manifest.json", manifest)
    print(json.dumps({
        "matter_movie": manifest["matter_movie"],
        "constraint_movie": manifest["constraint_movie"],
        "diagnostic_png": manifest["diagnostic_png"],
        "diagnostic_pdf": manifest["diagnostic_pdf"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
