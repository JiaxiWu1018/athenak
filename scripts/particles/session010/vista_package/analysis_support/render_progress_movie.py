#!/usr/bin/env python3
"""Render synchronized uniform-grid density and particle xy projections."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


GROUPS = (
    ("Envelope", 0, 3_000_000, "#c8d2dc", 40_000),
    ("Clump A", 3_000_000, 4_000_000, "#ff9f43", 25_000),
    ("Clump B", 4_000_000, 5_000_000, "#4dd7c8", 25_000),
)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def vtk_time(path: pathlib.Path) -> float:
    with path.open("rb") as stream:
        header = stream.read(1024)
    match = re.search(rb"time=\s*([-+0-9.eE]+)", header)
    if match is None:
        raise RuntimeError(f"missing time in {path}")
    return float(match.group(1))


def read_block(raw: bytes, marker: bytes, count: int) -> np.ndarray:
    start = raw.find(marker)
    if start < 0:
        raise RuntimeError(f"missing VTK marker {marker!r}")
    start = raw.find(b"\n", start + len(marker)) + 1
    if start <= 0:
        raise RuntimeError(f"malformed VTK marker {marker!r}")
    return np.frombuffer(raw, dtype=">f4", count=count, offset=start)


def read_particles(path: pathlib.Path) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, str]:
    raw = path.read_bytes()
    time_match = re.search(rb"time=\s*([-+0-9.eE]+)", raw[:1024])
    points_match = re.search(rb"POINTS\s+(\d+)\s+float", raw[:1024])
    if time_match is None or points_match is None:
        raise RuntimeError(f"invalid particle VTK file: {path}")
    count = int(points_match.group(1))
    positions = read_block(raw, points_match.group(0), 3 * count).reshape(count, 3)
    mass = read_block(
        raw, b"SCALARS prtcl_mass float\nLOOKUP_TABLE default", count
    )
    tags = read_block(
        raw, b"SCALARS ptag float\nLOOKUP_TABLE default", count
    ).astype(np.int64)
    return float(time_match.group(1)), positions, mass, tags, sha256_bytes(raw)


def mix64(values: np.ndarray) -> np.ndarray:
    mixed = values.astype(np.uint64).copy()
    mixed ^= mixed >> np.uint64(30)
    mixed *= np.uint64(0xBF58476D1CE4E5B9)
    mixed ^= mixed >> np.uint64(27)
    mixed *= np.uint64(0x94D049BB133111EB)
    mixed ^= mixed >> np.uint64(31)
    return mixed


def choose_tags(tags: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    chosen = []
    details: dict[str, Any] = {}
    for label, low, high, _, target in GROUPS:
        available = tags[(tags >= low) & (tags < high)]
        take = min(target, len(available))
        if take:
            order = np.argpartition(mix64(available), take - 1)[:take]
            values = np.sort(available[order])
        else:
            values = np.empty(0, dtype=np.int64)
        chosen.append(values)
        details[label] = {
            "tag_range": [low, high],
            "available": int(len(available)),
            "selected": int(len(values)),
            "selected_tag_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }
    return np.concatenate(chosen), details


def deposit_xy(positions: np.ndarray, mass: np.ndarray, limit: float,
               cells: int) -> np.ndarray:
    """Two-dimensional CIC deposition of projected particle mass."""
    dx = 2.0 * limit / cells
    ux = (positions[:, 0].astype(np.float64) + limit) / dx - 0.5
    uy = (positions[:, 1].astype(np.float64) + limit) / dx - 0.5
    ix = np.floor(ux).astype(np.int64)
    iy = np.floor(uy).astype(np.int64)
    fx = ux - ix
    fy = uy - iy
    weights = mass.astype(np.float64, copy=False)
    flat_density = np.zeros(cells * cells, dtype=np.float64)
    for ox in (0, 1):
        gx = ix + ox
        wx = fx if ox else 1.0 - fx
        for oy in (0, 1):
            gy = iy + oy
            wy = fy if oy else 1.0 - fy
            keep = (gx >= 0) & (gx < cells) & (gy >= 0) & (gy < cells)
            flat = gy[keep] * cells + gx[keep]
            flat_density += np.bincount(
                flat, weights=(weights * wx * wy)[keep], minlength=cells * cells
            )
    return (flat_density.reshape(cells, cells) / dx**2).astype(np.float32)


def load_horizons(paths: list[pathlib.Path]) -> tuple[dict[int, list[dict[str, float]]], dict]:
    horizons: dict[int, list[dict[str, float]]] = {}
    provenance = {}
    for index, path in enumerate(paths):
        rows = []
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                try:
                    if int(row["published_this_candidate"]) != 1:
                        continue
                    if int(row["quality_persist_ok"]) != 1:
                        continue
                    rmin, rmax = float(row["rmin"]), float(row["rmax"])
                    if not (0.0 < rmin <= rmax):
                        continue
                    rows.append({
                        "time": float(row["time"]),
                        "x": float(row["center_x"]),
                        "y": float(row["center_y"]),
                        "radius": 0.5 * (rmin + rmax),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
        rows.sort(key=lambda item: item["time"])
        horizons[index] = rows
        provenance[str(index)] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "accepted_records": len(rows),
            "first_time": rows[0]["time"] if rows else None,
            "last_time": rows[-1]["time"] if rows else None,
        }
    return horizons, provenance


def horizon_at(rows: list[dict[str, float]], time_value: float):
    eligible = [row for row in rows if row["time"] <= time_value + 1.0e-8]
    return eligible[-1] if eligible else None


def draw_horizons(ax, horizons: dict[int, list[dict[str, float]]],
                  time_value: float) -> list[int]:
    visible = []
    for index, rows in horizons.items():
        row = horizon_at(rows, time_value)
        if row is None:
            continue
        ax.add_patch(Circle(
            (row["x"], row["y"]), row["radius"], fill=False,
            edgecolor="#82ff63", linewidth=1.6, linestyle="--", zorder=12,
        ))
        ax.scatter(
            [row["x"]], [row["y"]], marker="+", s=45, linewidths=1.25,
            color="#82ff63", zorder=13,
        )
        visible.append(index)
    return visible


def render_frame(path: pathlib.Path, cache: pathlib.Path, horizons, limit: float,
                 vmin: float, vmax: float, frame: int, frames: int) -> dict[str, Any]:
    with np.load(cache) as data:
        density = data["density"]
        x = data["x"]
        y = data["y"]
        group = data["group"]
        time_value = float(data["time"])
        total_particles = int(data["total_particles"])

    fig = plt.figure(figsize=(16, 7.5), dpi=120, facecolor="#080a0f")
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 0.032, 1.0), wspace=0.13)
    density_ax = fig.add_subplot(grid[0, 0])
    color_ax = fig.add_subplot(grid[0, 1])
    particle_ax = fig.add_subplot(grid[0, 2])
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#080a0f")
    image = density_ax.imshow(
        np.ma.masked_less_equal(density, 0.0), origin="lower",
        extent=(-limit, limit, -limit, limit), interpolation="bilinear",
        cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), rasterized=True,
    )
    colorbar = fig.colorbar(image, cax=color_ax)
    colorbar.set_label("Projected mass density", color="white", fontsize=10)
    colorbar.ax.tick_params(colors="white", labelsize=8)
    colorbar.outline.set_edgecolor("#9ba8b4")

    for index, (label, _, _, color, _) in enumerate(GROUPS):
        use = group == index
        size = 0.55 if index == 0 else 1.05
        alpha = 0.22 if index == 0 else 0.72
        particle_ax.scatter(
            x[use], y[use], s=size, alpha=alpha, color=color,
            linewidths=0, rasterized=True, label=label,
        )

    density_horizons = draw_horizons(density_ax, horizons, time_value)
    particle_horizons = draw_horizons(particle_ax, horizons, time_value)
    for ax in (density_ax, particle_ax):
        ax.set_facecolor("#080a0f")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.set_xlabel("x / M", color="white")
        ax.set_ylabel("y / M", color="white")
        ax.tick_params(colors="#d8dee6", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#66717d")
        ax.grid(color="white", alpha=0.07, linewidth=0.5)
    density_ax.set_title("Uniform-grid xy projected density", color="white", fontsize=13)
    particle_ax.set_title("Particle positions (fixed tag sample)", color="white", fontsize=13)

    handles, labels = particle_ax.get_legend_handles_labels()
    if particle_horizons:
        handles.append(Line2D(
            [0], [0], color="#82ff63", marker="+", linestyle="--",
            linewidth=1.5, markersize=7,
        ))
        labels.append("Accepted AH: true radius and center")
    particle_ax.legend(
        handles, labels, loc="upper right", fontsize=8, framealpha=0.78,
        facecolor="#151922", edgecolor="#66717d", labelcolor="white",
    )
    fig.suptitle(
        f"Session 007 progress   t = {time_value:.3f} M   "
        f"live particles = {total_particles:,}   frame {frame + 1}/{frames}",
        color="white", fontsize=15, y=0.975,
    )
    fig.text(
        0.5, 0.018,
        "Both panels show the xy projection over -32M to +32M; AH overlays appear only after accepted finds.",
        color="#bdc7d2", fontsize=9, ha="center",
    )
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.085, top=0.91)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {
        "time": time_value,
        "particles": total_particles,
        "accepted_horizons": density_horizons,
    }


def image_qa(path: pathlib.Path) -> dict[str, Any]:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        return {
            "width": image.width,
            "height": image.height,
            "channel_stddev": [float(value) for value in pixels.std(axis=(0, 1))],
            "nonblank": bool(float(pixels.std()) > 5.0),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particle-dir", required=True, type=pathlib.Path)
    parser.add_argument("--horizon", action="append", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--movie", required=True, type=pathlib.Path)
    parser.add_argument("--home-movie", required=True, type=pathlib.Path)
    parser.add_argument("--limit", type=float, default=32.0)
    parser.add_argument("--cells", type=int, default=512)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    if args.limit <= 0.0 or args.cells < 64 or args.cells % 2:
        raise RuntimeError("limit must be positive and cells must be an even integer >= 64")
    paths = sorted(args.particle_dir.glob("*.part.vtk"), key=vtk_time)
    if not paths:
        raise RuntimeError("no particle snapshots found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    frame_dir = args.output_dir / "frames"
    cache_dir.mkdir(exist_ok=True)
    frame_dir.mkdir(exist_ok=True)
    horizons, horizon_provenance = load_horizons(args.horizon)

    selection_path = args.output_dir / "selected_tags.npy"
    selection_details_path = args.output_dir / "selection.json"
    if selection_path.exists() and selection_details_path.exists():
        selected_tags = np.load(selection_path)
        selection_details = json.loads(selection_details_path.read_text(encoding="utf-8"))
    else:
        _, _, _, initial_tags, _ = read_particles(paths[0])
        selected_tags, selection_details = choose_tags(initial_tags)
        np.save(selection_path, selected_tags)
        atomic_json(selection_details_path, selection_details)

    source_records = []
    density_quantiles = []
    for index, source in enumerate(paths):
        cache = cache_dir / f"frame_{index:05d}.npz"
        if cache.exists():
            with np.load(cache) as data:
                time_value = float(data["time"])
                total_particles = int(data["total_particles"])
                source_hash = str(data["source_sha256"])
                positive = data["density"][data["density"] > 0.0]
        else:
            time_value, positions, mass, tags, source_hash = read_particles(source)
            density = deposit_xy(positions, mass, args.limit, args.cells)
            keep = np.isin(tags, selected_tags, assume_unique=False)
            sample_positions = positions[keep]
            sample_tags = tags[keep]
            sample_group = np.full(len(sample_tags), 255, dtype=np.uint8)
            for group_index, (_, low, high, _, _) in enumerate(GROUPS):
                sample_group[(sample_tags >= low) & (sample_tags < high)] = group_index
            selected = sample_group != 255
            np.savez_compressed(
                cache, density=density,
                x=sample_positions[selected, 0].astype(np.float32),
                y=sample_positions[selected, 1].astype(np.float32),
                group=sample_group[selected], time=np.float64(time_value),
                total_particles=np.int64(len(tags)), source_sha256=source_hash,
            )
            total_particles = len(tags)
            positive = density[density > 0.0]
            print(
                f"cached {index + 1}/{len(paths)} t={time_value:.6f} "
                f"particles={total_particles}", flush=True,
            )
        if positive.size == 0:
            raise RuntimeError(f"empty deposited density for {source}")
        density_quantiles.append([
            float(np.quantile(positive, 0.05)),
            float(np.quantile(positive, 0.999)),
        ])
        source_records.append({
            "frame": index,
            "time": time_value,
            "particles": total_particles,
            "path": str(source.resolve()),
            "bytes": source.stat().st_size,
            "sha256": source_hash,
        })

    vmin = min(value[0] for value in density_quantiles)
    vmax = max(value[1] for value in density_quantiles)
    if not (0.0 < vmin < vmax):
        raise RuntimeError(f"invalid density scale {vmin}, {vmax}")
    rendered = []
    for index in range(len(paths)):
        frame = frame_dir / f"progress_{index:05d}.png"
        details = render_frame(
            frame, cache_dir / f"frame_{index:05d}.npz", horizons,
            args.limit, vmin, vmax, index, len(paths),
        )
        rendered.append({"frame": index, "path": str(frame.resolve()), **details})
        print(f"rendered {index + 1}/{len(paths)}", flush=True)

    args.movie.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
        "-i", str(frame_dir / "progress_%05d.png"),
        "-c:v", "mpeg4", "-q:v", "2",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.movie),
    ], check=True)
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,nb_read_frames,duration",
        "-of", "json", str(args.movie),
    ], text=True))
    stream = probe["streams"][0]
    if int(stream["nb_read_frames"]) != len(paths):
        raise RuntimeError("movie frame count does not match input snapshots")
    qa = {
        "first_frame": image_qa(frame_dir / "progress_00000.png"),
        "last_frame": image_qa(frame_dir / f"progress_{len(paths) - 1:05d}.png"),
        "ffprobe": stream,
    }
    if not qa["first_frame"]["nonblank"] or not qa["last_frame"]["nonblank"]:
        raise RuntimeError("blank rendered frame detected")

    movie_hash = sha256_file(args.movie)
    args.home_movie.parent.mkdir(parents=True, exist_ok=True)
    if args.home_movie.exists():
        if sha256_file(args.home_movie) != movie_hash:
            raise RuntimeError(f"refusing to overwrite different home movie {args.home_movie}")
    else:
        temporary = args.home_movie.with_name(args.home_movie.name + ".partial")
        shutil.copy2(args.movie, temporary)
        os.replace(temporary, args.home_movie)
    args.home_movie.with_suffix(args.home_movie.suffix + ".sha256").write_text(
        f"{movie_hash}  {args.home_movie.name}\n", encoding="ascii"
    )
    manifest = {
        "schema_version": 1,
        "method": "2-D CIC deposition of particle mass onto a fixed xy grid; right panel is a fixed deterministic tag sample",
        "display": [-args.limit, args.limit, -args.limit, args.limit],
        "grid_cells": [args.cells, args.cells],
        "grid_dx": 2.0 * args.limit / args.cells,
        "density_field": "projected particle mass per coordinate area",
        "density_scale": {"type": "LogNorm", "vmin": vmin, "vmax": vmax},
        "selection": selection_details,
        "horizons": horizon_provenance,
        "sources": source_records,
        "frames": rendered,
        "movie": {
            "scratch_path": str(args.movie.resolve()),
            "home_path": str(args.home_movie.resolve()),
            "bytes": args.movie.stat().st_size,
            "sha256": movie_hash,
            "fps": args.fps,
        },
        "qa": qa,
    }
    atomic_json(args.output_dir / "movie_manifest.json", manifest)
    print(json.dumps(manifest["movie"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
