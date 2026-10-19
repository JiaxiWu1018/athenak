#!/usr/bin/env python3
"""Render a clean, fixed-camera 3-D particle movie for the Jeans collapse."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BACKGROUND = "#071A2B"
GROUP_COLORS = {
    0: "#8E5AB5",  # envelope: purple
    1: "#FF9F1C",  # clump A: orange
    2: "#31B7E5",  # clump B: blue
}
GROUP_STYLES = {
    0: {"size": 1.45, "alpha": 0.34},
    1: {"size": 2.10, "alpha": 0.82},
    2: {"size": 2.10, "alpha": 0.82},
}
VIEW_LIMIT = 24.0
CAMERA = {"elev": 22.0, "azim": -58.0, "roll": 0.0, "focal_length": 1.25}


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("gi_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_sphere(center: tuple[float, float, float], radius: float):
    u = np.linspace(0.0, 2.0 * np.pi, 49)
    v = np.linspace(0.0, np.pi, 25)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def parse_indices(value: str | None, count: int) -> list[int]:
    if value is None:
        return list(range(count))
    indices = sorted({int(item) for item in value.split(",") if item.strip()})
    if not indices or indices[0] < 0 or indices[-1] >= count:
        raise ValueError(f"indices must lie in [0,{count - 1}]")
    return indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-script", type=Path, required=True)
    parser.add_argument("--production-out", type=Path, required=True)
    parser.add_argument("--continuation-out", type=Path, required=True)
    parser.add_argument("--selected-tags", type=Path, required=True)
    parser.add_argument("--original-sidecars", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--movie", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--indices", type=str)
    parser.add_argument("--no-encode", action="store_true")
    args = parser.parse_args()

    base = load_base(args.base_script)
    particle_candidates = list((args.production_out / "pvtk").glob("*.part.vtk"))
    particle_candidates += list((args.continuation_out / "pvtk").glob("*.part.vtk"))
    sources = base.deduplicated_files(particle_candidates, base.vtk_metadata)
    if len(sources) != 257:
        raise RuntimeError(f"expected 257 particle dumps, found {len(sources)}")
    indices = parse_indices(args.indices, len(sources))

    selected = np.load(args.selected_tags)
    if len(selected) != 90_000 or np.any(selected[1:] <= selected[:-1]):
        raise RuntimeError("expected 90,000 unique, sorted selected particle tags")

    horizon_paths = {
        0: [
            args.original_sidecars / "s7_two_clump_ccw.horizon_consumer_0.csv",
            args.continuation_out / "s7_two_clump_ccw.horizon_consumer_0.csv",
        ],
        1: [
            args.original_sidecars / "s7_two_clump_ccw.horizon_consumer_1.csv",
            args.continuation_out / "s7_two_clump_ccw.horizon_consumer_1.csv",
        ],
        2: [args.continuation_out / "s7_two_clump_ccw.horizon_consumer_2.csv"],
    }
    horizons, horizon_provenance = base.read_horizons(horizon_paths)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.frame_dir.mkdir(parents=True, exist_ok=True)
    args.movie.parent.mkdir(parents=True, exist_ok=True)
    records = []

    for output_index, source_index in enumerate(indices):
        source = sources[source_index]
        cache = args.cache_dir / f"particle3d_{source_index:05d}.npz"
        if not cache.exists():
            time_value, cycle, pos, _, tags, source_digest = base.read_particles(source)
            use = base.selected_mask(tags, selected)
            sample_pos = pos[use].astype(np.float32)
            sample_tags = tags[use]
            group = np.full(len(sample_tags), 255, dtype=np.uint8)
            for group_id, (_, low, high, _, _) in enumerate(base.GROUPS):
                group[(sample_tags >= low) & (sample_tags < high)] = group_id
            if np.any(group == 255):
                raise RuntimeError(f"unclassified selected particle in {source}")
            np.savez_compressed(
                cache,
                x=sample_pos[:, 0],
                y=sample_pos[:, 1],
                z=sample_pos[:, 2],
                group=group,
                time=np.float64(time_value),
                cycle=np.int64(cycle),
                source_sha256=np.asarray(source_digest),
            )

        with np.load(cache) as data:
            x = data["x"]
            y = data["y"]
            z = data["z"]
            group = data["group"]
            time_value = float(data["time"])
            cycle = int(data["cycle"])
            source_digest = str(data["source_sha256"])

        visible_horizons = []
        buried = np.zeros(len(x), dtype=bool)
        for horizon_id, rows in horizons.items():
            row = base.at_time(rows, time_value)
            if row is None:
                continue
            visible_horizons.append((horizon_id, row))
            buried |= (
                (x - row["x"]) ** 2
                + (y - row["y"]) ** 2
                + (z - row["z"]) ** 2
                < (1.002 * row["radius"]) ** 2
            )

        in_view = (
            (np.abs(x) <= VIEW_LIMIT)
            & (np.abs(y) <= VIEW_LIMIT)
            & (np.abs(z) <= VIEW_LIMIT)
        )
        keep = (~buried) & in_view

        fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BACKGROUND)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d", facecolor=BACKGROUND)
        ax.set_proj_type("persp", focal_length=CAMERA["focal_length"])
        ax.view_init(elev=CAMERA["elev"], azim=CAMERA["azim"], roll=CAMERA["roll"])
        ax.set_xlim(-VIEW_LIMIT, VIEW_LIMIT)
        ax.set_ylim(-VIEW_LIMIT, VIEW_LIMIT)
        ax.set_zlim(-VIEW_LIMIT, VIEW_LIMIT)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.set_axis_off()

        for group_id in (0, 1, 2):
            use = keep & (group == group_id)
            style = GROUP_STYLES[group_id]
            ax.scatter(
                x[use], y[use], z[use],
                s=style["size"],
                alpha=style["alpha"],
                color=GROUP_COLORS[group_id],
                edgecolors="none",
                linewidths=0,
                depthshade=False,
                rasterized=True,
            )

        for _, row in visible_horizons:
            sx, sy, sz = make_sphere(
                (row["x"], row["y"], row["z"]), row["radius"]
            )
            ax.plot_surface(
                sx, sy, sz,
                color="#000000",
                edgecolor="none",
                linewidth=0,
                antialiased=True,
                shade=False,
                alpha=1.0,
            )

        frame = args.frame_dir / f"frame_{source_index:05d}.png"
        fig.savefig(frame, facecolor=BACKGROUND, edgecolor=BACKGROUND, pad_inches=0)
        plt.close(fig)

        visible_by_group = {
            base.GROUPS[group_id][0]: int(np.count_nonzero(keep & (group == group_id)))
            for group_id in range(len(base.GROUPS))
        }
        buried_by_group = {
            base.GROUPS[group_id][0]: int(np.count_nonzero(buried & (group == group_id)))
            for group_id in range(len(base.GROUPS))
        }
        records.append({
            "frame": source_index,
            "time_M": time_value,
            "cycle": cycle,
            "source": str(source.resolve()),
            "source_sha256": source_digest,
            "horizons": [horizon_id for horizon_id, _ in visible_horizons],
            "visible_by_group": visible_by_group,
            "buried_by_group": buried_by_group,
            "visible_markers": int(np.count_nonzero(keep)),
            "buried_markers": int(np.count_nonzero(buried)),
        })
        print(
            f"3D frame {output_index + 1}/{len(indices)} "
            f"source={source_index} t={time_value:.2f}",
            flush=True,
        )

    metadata = {
        "movie": str(args.movie),
        "frames": len(sources) if args.indices is None else len(indices),
        "fps": args.fps,
        "resolution": [1920, 1080],
        "time_range_M": [records[0]["time_M"], records[-1]["time_M"]],
        "particle_selection": "surviving members of a fixed deterministic 90,000-particle t=0 cohort: 40,000 envelope, 25,000 per clump; accreted/excised tags are not replaced",
        "geometry": "true x-y-z particle positions; not a projection or slab",
        "view_M": [-VIEW_LIMIT, VIEW_LIMIT],
        "camera": CAMERA,
        "background": BACKGROUND,
        "group_colors": {
            base.GROUPS[group_id][0]: GROUP_COLORS[group_id]
            for group_id in range(len(base.GROUPS))
        },
        "axes": False,
        "text": False,
        "horizon_render": "accepted apparent horizons as opaque black spheres; sampled particles geometrically inside each sphere are removed",
        "selected_tags_sha256": sha256_file(args.selected_tags),
        "horizon_provenance": horizon_provenance,
        "records": records,
    }
    metadata_path = args.movie.with_suffix(".json")
    if args.indices is not None:
        metadata_path = args.frame_dir / "qa_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if args.no_encode:
        return
    if indices != list(range(len(sources))):
        raise RuntimeError("encoding requires the complete contiguous frame sequence")
    ffmpeg = os.environ.get("FFMPEG", "ffmpeg")
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "error", "-framerate", str(args.fps),
        "-i", str(args.frame_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.movie),
    ], check=True)


if __name__ == "__main__":
    main()
