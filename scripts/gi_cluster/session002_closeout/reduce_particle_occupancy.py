#!/usr/bin/env python3
"""Reduce particle occupancy by active AMR leaf level for GI Session 002.

The particle VTK files contain every surviving particle at nine output times.  The
full-volume ``E3d`` binary dump supplies the active leaf MeshBlock geometry.  The
``g2_L128`` production hierarchy created/deleted zero blocks, so one verified leaf
geometry applies to all nine particle dumps.

Main assignment uses the MeshBlock logical-location octree.  A deliberately separate
spot check uses direct half-open geometric containment against every leaf block.  The
two routes must give identical per-level cell counts for the selected dump.

The true minimum includes all active leaf cells and is therefore normally zero.  The
CSV also records the minimum among occupied cells, which is the informative minimum
used by the summary figure.  Means are over *all* active leaf cells at a level.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass

import numpy as np


KEY_BITS = 21
KEY_MASK = (1 << KEY_BITS) - 1


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_particle_positions(path: str) -> tuple[float, np.ndarray]:
    """Read time and big-endian float32 POINTS from an AthenaK particle VTK."""
    with open(path, "rb") as stream:
        data = stream.read()
    mt = re.search(rb"time=\s*([-+0-9.eE]+)", data)
    mp = re.search(rb"POINTS\s+(\d+)\s+float", data)
    if mt is None or mp is None:
        raise ValueError(f"{path}: missing AthenaK time or POINTS header")
    time = float(mt.group(1))
    n = int(mp.group(1))
    start = data.find(b"\n", mp.end()) + 1
    pos = np.frombuffer(data, dtype=">f4", count=3 * n, offset=start)
    if pos.size != 3 * n:
        raise ValueError(f"{path}: truncated POINTS block")
    return time, pos.astype(np.float64).reshape(n, 3)


def pack_keys(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
    for a in (ix, iy, iz):
        if np.any(a < 0) or np.any(a > KEY_MASK):
            raise ValueError("logical coordinate outside packed-key range")
    return (ix.astype(np.int64)
            | (iy.astype(np.int64) << KEY_BITS)
            | (iz.astype(np.int64) << (2 * KEY_BITS)))


@dataclass
class LeafMesh:
    logical: np.ndarray
    geometry: np.ndarray
    nxyz: tuple[int, int, int]
    domain_min: np.ndarray
    domain_max: np.ndarray
    levels: np.ndarray
    base_block_width: np.ndarray

    @property
    def cells_per_block(self) -> int:
        return int(np.prod(self.nxyz))


def load_leaf_mesh(path: str, athenak: str) -> tuple[LeafMesh, dict]:
    sys.path.insert(0, os.path.join(athenak, "vis", "python"))
    import bin_convert  # noqa: E402

    raw = bin_convert.read_binary(path)
    logical = np.asarray(raw["mb_logical"], dtype=np.int64)
    geometry = np.asarray(raw["mb_geometry"], dtype=np.float64)
    nxyz = tuple(int(raw[f"nx{i}_out_mb"]) for i in (1, 2, 3))
    if min(nxyz) <= 1:
        raise ValueError(
            f"{path}: expected a full 3-D leaf dump, got cells/block={nxyz}"
        )
    levels = logical[:, 3].astype(int)
    domain_min = geometry[:, [0, 2, 4]].min(axis=0)
    domain_max = geometry[:, [1, 3, 5]].max(axis=0)
    widths = geometry[:, [1, 3, 5]] - geometry[:, [0, 2, 4]]
    base_samples = widths * np.exp2(levels)[:, None]
    base = np.median(base_samples, axis=0)
    rel = np.max(np.abs(base_samples - base) / np.maximum(np.abs(base), 1e-300))
    if rel > 2e-11:
        raise ValueError(f"inconsistent octree block widths: max relative error {rel}")

    # Geometry and logical coordinates must encode the same octree.
    for level in sorted(set(levels)):
        m = levels == level
        bw = base / (2 ** level)
        derived = np.rint((geometry[m][:, [0, 2, 4]] - domain_min) / bw).astype(int)
        if not np.array_equal(derived, logical[m, :3]):
            raise ValueError(f"logical/geometry mismatch at level {level}")

    volumes = np.prod(widths, axis=1)
    leaf_volume = float(volumes.sum())
    domain_volume = float(np.prod(domain_max - domain_min))
    volume_relerr = abs(leaf_volume / domain_volume - 1.0)
    if volume_relerr > 5e-12:
        raise ValueError(
            f"leaf blocks do not tile domain: relative volume error {volume_relerr}"
        )

    mesh = LeafMesh(logical, geometry, nxyz, domain_min, domain_max, levels, base)
    qa = {
        "mesh_file": os.path.abspath(path),
        "mesh_sha256": sha256_file(path),
        "mesh_time": float(raw.get("time", raw.get("Time", math.nan))),
        "active_leaf_blocks": int(len(logical)),
        "cells_per_block": mesh.cells_per_block,
        "active_leaf_cells": int(len(logical) * mesh.cells_per_block),
        "levels": {str(int(level)): int(np.sum(levels == level))
                   for level in sorted(set(levels))},
        "domain_min": domain_min.tolist(),
        "domain_max": domain_max.tolist(),
        "leaf_volume_relative_error": volume_relerr,
        "logical_geometry_consistent": True,
    }
    return mesh, qa


def logical_assignment(pos: np.ndarray, mesh: LeafMesh) -> tuple[np.ndarray, np.ndarray]:
    """Assign particles to leaf blocks from logical-location keys, finest first."""
    n = len(pos)
    block = np.full(n, -1, dtype=np.int32)
    unresolved = np.arange(n, dtype=np.int64)
    span = mesh.domain_max - mesh.domain_min
    root_blocks = np.rint(span / mesh.base_block_width).astype(np.int64)

    for level in sorted(set(mesh.levels), reverse=True):
        if unresolved.size == 0:
            break
        ids = np.flatnonzero(mesh.levels == level)
        ll = mesh.logical[ids, :3]
        block_keys = pack_keys(ll[:, 0], ll[:, 1], ll[:, 2])
        order = np.argsort(block_keys)
        block_keys = block_keys[order]
        block_ids = ids[order]

        bw = mesh.base_block_width / (2 ** level)
        q = np.floor((pos[unresolved] - mesh.domain_min) / bw).astype(np.int64)
        nblock_axis = root_blocks * (2 ** level)
        q = np.maximum(q, 0)
        q = np.minimum(q, nblock_axis - 1)
        pkeys = pack_keys(q[:, 0], q[:, 1], q[:, 2])
        loc = np.searchsorted(block_keys, pkeys)
        inside = loc < len(block_keys)
        hit = np.zeros(len(unresolved), dtype=bool)
        hit[inside] = block_keys[loc[inside]] == pkeys[inside]
        if np.any(hit):
            block[unresolved[hit]] = block_ids[loc[hit]]
        unresolved = unresolved[~hit]

    return block, unresolved


def local_cell_ids(pos: np.ndarray, block: np.ndarray, mesh: LeafMesh) -> np.ndarray:
    if np.any(block < 0):
        raise ValueError("cannot compute local cells for unassigned particles")
    geom = mesh.geometry[block]
    gmin = geom[:, [0, 2, 4]]
    gmax = geom[:, [1, 3, 5]]
    dxyz = (gmax - gmin) / np.asarray(mesh.nxyz)
    q = np.floor((pos - gmin) / dxyz).astype(np.int64)
    q = np.maximum(q, 0)
    q = np.minimum(q, np.asarray(mesh.nxyz) - 1)
    nx, ny, nz = mesh.nxyz
    local = q[:, 0] + nx * (q[:, 1] + ny * q[:, 2])
    return block.astype(np.int64) * mesh.cells_per_block + local


def summarize_assignment(time: float, dump_index: int, pos: np.ndarray,
                         block: np.ndarray, mesh: LeafMesh) -> list[dict]:
    cell = local_cell_ids(pos, block, mesh)
    particle_level = mesh.levels[block]
    rows = []
    for level in sorted(set(mesh.levels)):
        mb = int(np.sum(mesh.levels == level))
        active = mb * mesh.cells_per_block
        m = particle_level == level
        npt = int(np.sum(m))
        if npt:
            _, counts = np.unique(cell[m], return_counts=True)
            occupied = int(len(counts))
            occupied_min = int(counts.min())
            maximum = int(counts.max())
            mean_occupied = float(npt / occupied)
        else:
            occupied = 0
            occupied_min = math.nan
            maximum = 0
            mean_occupied = math.nan
        true_min = occupied_min if occupied == active else 0
        dx = float(mesh.base_block_width[0] / (2 ** level) / mesh.nxyz[0])
        rows.append({
            "time": time,
            "dump_index": dump_index,
            "particle_count_total": int(len(pos)),
            "level": int(level),
            "dx": dx,
            "active_leaf_blocks": mb,
            "active_leaf_cells": active,
            "particles_in_level": npt,
            "occupied_leaf_cells": occupied,
            "occupied_cell_fraction": float(occupied / active),
            "true_min_including_empty": true_min,
            "min_occupied": occupied_min,
            "mean_all_active": float(npt / active),
            "mean_occupied": mean_occupied,
            "max": maximum,
        })
    return rows


def direct_geometry_summary(time: float, dump_index: int, pos: np.ndarray,
                            mesh: LeafMesh) -> tuple[list[dict], dict]:
    """Independent spot check by direct block containment, not logical keys."""
    order = np.argsort(pos[:, 0], kind="mergesort")
    xs = pos[order, 0]
    seen = np.zeros(len(pos), dtype=np.uint8)
    counts_by_level: dict[int, list[np.ndarray]] = {
        int(level): [] for level in sorted(set(mesh.levels))
    }
    npt_by_level = {int(level): 0 for level in sorted(set(mesh.levels))}
    nx, ny, nz = mesh.nxyz
    for ib, g in enumerate(mesh.geometry):
        left = int(np.searchsorted(xs, g[0], side="left"))
        xside = "right" if math.isclose(g[1], mesh.domain_max[0]) else "left"
        right = int(np.searchsorted(xs, g[1], side=xside))
        cand = order[left:right]
        if cand.size == 0:
            counts_by_level[int(mesh.levels[ib])].append(np.zeros(0, dtype=np.int64))
            continue
        upper_y = pos[cand, 1] <= g[3] if math.isclose(g[3], mesh.domain_max[1]) \
            else pos[cand, 1] < g[3]
        upper_z = pos[cand, 2] <= g[5] if math.isclose(g[5], mesh.domain_max[2]) \
            else pos[cand, 2] < g[5]
        inside = ((pos[cand, 1] >= g[2]) & upper_y
                  & (pos[cand, 2] >= g[4]) & upper_z)
        ids = cand[inside]
        if ids.size:
            seen[ids] += 1
            dxyz = ((g[[1, 3, 5]] - g[[0, 2, 4]]) / np.asarray(mesh.nxyz))
            q = np.floor((pos[ids] - g[[0, 2, 4]]) / dxyz).astype(np.int64)
            q = np.maximum(q, 0)
            q = np.minimum(q, np.asarray(mesh.nxyz) - 1)
            local = q[:, 0] + nx * (q[:, 1] + ny * q[:, 2])
            _, cell_counts = np.unique(local, return_counts=True)
        else:
            cell_counts = np.zeros(0, dtype=np.int64)
        level = int(mesh.levels[ib])
        npt_by_level[level] += int(ids.size)
        counts_by_level[level].append(cell_counts)

    if np.any(seen != 1):
        raise ValueError(
            f"direct containment failed: unassigned={int(np.sum(seen == 0))} "
            f"multiply_assigned={int(np.sum(seen > 1))}"
        )

    rows = []
    for level in sorted(counts_by_level):
        nonempty = [a for a in counts_by_level[level] if a.size]
        counts = np.concatenate(nonempty) if nonempty else np.zeros(0, dtype=np.int64)
        mb = int(np.sum(mesh.levels == level))
        active = mb * mesh.cells_per_block
        npt = npt_by_level[level]
        occupied = int(len(counts))
        occupied_min = int(counts.min()) if occupied else math.nan
        rows.append({
            "time": time,
            "dump_index": dump_index,
            "particle_count_total": int(len(pos)),
            "level": level,
            "dx": float(mesh.base_block_width[0] / (2 ** level) / nx),
            "active_leaf_blocks": mb,
            "active_leaf_cells": active,
            "particles_in_level": npt,
            "occupied_leaf_cells": occupied,
            "occupied_cell_fraction": float(occupied / active),
            "true_min_including_empty": occupied_min if occupied == active else 0,
            "min_occupied": occupied_min,
            "mean_all_active": float(npt / active),
            "mean_occupied": float(npt / occupied) if occupied else math.nan,
            "max": int(counts.max()) if occupied else 0,
        })
    qa = {
        "method": "direct half-open containment in every active leaf MeshBlock",
        "particles_seen_once": int(np.sum(seen == 1)),
        "unassigned": int(np.sum(seen == 0)),
        "multiply_assigned": int(np.sum(seen > 1)),
    }
    return rows, qa


def compare_rows(main_rows: list[dict], direct_rows: list[dict]) -> dict:
    by_level = {int(r["level"]): r for r in direct_rows}
    fields = [
        "active_leaf_blocks", "active_leaf_cells", "particles_in_level",
        "occupied_leaf_cells", "true_min_including_empty", "min_occupied", "max",
    ]
    mismatches = []
    for row in main_rows:
        other = by_level[int(row["level"])]
        for field in fields:
            a, b = row[field], other[field]
            if (isinstance(a, float) and math.isnan(a)
                    and isinstance(b, float) and math.isnan(b)):
                continue
            if a != b:
                mismatches.append({"level": int(row["level"]), "field": field,
                                   "logical": a, "direct": b})
    return {"exact_match": not mismatches, "mismatches": mismatches}


CSV_FIELDS = [
    "time", "dump_index", "particle_count_total", "level", "dx",
    "active_leaf_blocks", "active_leaf_cells", "particles_in_level",
    "occupied_leaf_cells", "occupied_cell_fraction", "true_min_including_empty",
    "min_occupied", "mean_all_active", "mean_occupied", "max",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--athenak", required=True)
    parser.add_argument("--mesh-file", default=None,
                        help="full-volume bin dump; default first *.E3d.*.bin")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-qa", required=True)
    parser.add_argument("--spot-index", type=int, default=4)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    mesh_file = args.mesh_file
    if mesh_file is None:
        candidates = sorted(glob.glob(os.path.join(run_dir, "out", "bin", "*.E3d.*.bin")))
        if not candidates:
            raise FileNotFoundError("no full-volume *.E3d.*.bin mesh dump")
        mesh_file = candidates[0]
    vtk_files = sorted(glob.glob(os.path.join(run_dir, "out", "pvtk", "*.part.vtk")))
    if not vtk_files:
        raise FileNotFoundError("no particle VTK dumps")
    if not (0 <= args.spot_index < len(vtk_files)):
        raise ValueError("spot-index outside particle-dump range")

    mesh, mesh_qa = load_leaf_mesh(mesh_file, args.athenak)
    all_rows: list[dict] = []
    dumps = []
    spot_main = spot_pos = spot_time = None
    for index, path in enumerate(vtk_files):
        time, pos = read_particle_positions(path)
        block, unresolved = logical_assignment(pos, mesh)
        if unresolved.size:
            raise ValueError(f"{path}: {len(unresolved)} particles not in a leaf block")
        rows = summarize_assignment(time, index, pos, block, mesh)
        all_rows.extend(rows)
        dumps.append({
            "index": index,
            "time": time,
            "path": os.path.abspath(path),
            "size_bytes": os.path.getsize(path),
            "sha256": sha256_file(path),
            "particle_count": int(len(pos)),
            "assigned_particle_count": int(np.sum(block >= 0)),
            "particles_by_level": {
                str(int(r["level"])): int(r["particles_in_level"]) for r in rows
            },
        })
        print(f"dump {index:02d} t={time:7.3f}: N={len(pos):,} assigned={len(pos):,}")
        if index == args.spot_index:
            spot_main, spot_pos, spot_time = rows, pos.copy(), time

    direct_rows, direct_qa = direct_geometry_summary(
        float(spot_time), args.spot_index, spot_pos, mesh
    )
    comparison = compare_rows(spot_main, direct_rows)
    if not comparison["exact_match"]:
        raise ValueError(f"independent spot check disagrees: {comparison['mismatches'][:5]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    qa = {
        "analysis": "particle count per active 3-D AMR leaf cell",
        "run_dir": run_dir,
        "run_name": os.path.basename(run_dir),
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "main_method": "logical-location octree, finest active leaf first",
        "minimum_convention": {
            "true_min_including_empty": "minimum over every active leaf cell at level",
            "min_occupied": "minimum over active leaf cells containing >=1 particle",
            "mean_all_active": "particles_in_level / all active leaf cells at level",
        },
        "mesh": mesh_qa,
        "particle_dumps": dumps,
        "independent_spot_check": {
            "dump_index": args.spot_index,
            "time": float(spot_time),
            **direct_qa,
            **comparison,
        },
        "all_true_minima_zero": all(
            int(r["true_min_including_empty"]) == 0 for r in all_rows
        ),
        "output_csv": os.path.abspath(args.out_csv),
    }
    with open(args.out_qa, "w") as stream:
        json.dump(qa, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"wrote {args.out_csv} ({len(all_rows)} rows)")
    print(f"wrote {args.out_qa}; independent spot check exact={comparison['exact_match']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
