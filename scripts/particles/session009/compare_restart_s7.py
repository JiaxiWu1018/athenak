#!/usr/bin/env python3
"""Compare Session 007 uninterrupted and rank-changed checkpoint continuations."""
from __future__ import annotations
import argparse
import json
import math
import re
import sys
from pathlib import Path
import numpy as np


def latest(root: Path, pattern: str) -> Path:
    paths = sorted(root.rglob(pattern))
    if not paths:
        raise FileNotFoundError(f"{pattern} under {root}")
    return paths[-1]


def vtk_block(data: bytes, marker: bytes, count: int) -> np.ndarray:
    start = data.find(marker)
    if start < 0:
        raise ValueError(f"missing VTK marker {marker!r}")
    start = data.find(b"\n", start + len(marker)) + 1
    if start <= 0:
        raise ValueError(f"unterminated VTK marker {marker!r}")
    return np.frombuffer(data, dtype=">f4", count=count, offset=start)


def read_particles(path: Path):
    data = path.read_bytes()
    point = re.search(rb"POINTS\s+(\d+)\s+float", data)
    tm = re.search(rb"time=\s*([-+0-9.eE]+)", data)
    if point is None or tm is None:
        raise ValueError(f"incomplete particle output {path}")
    count = int(point.group(1))
    pos = vtk_block(data, point.group(0), 3*count).reshape(count, 3)
    mom = vtk_block(data, b"VECTORS prtcl_vel float", 3*count).reshape(count, 3)
    tag = vtk_block(data, b"SCALARS ptag float\nLOOKUP_TABLE default", count).astype(np.int64)
    mass = vtk_block(data, b"SCALARS prtcl_mass float\nLOOKUP_TABLE default", count)
    return float(tm.group(1)), pos, mom, tag, mass


def field_comparison(left: dict, right: dict, rtol: float, atol: float) -> dict:
    if int(left["cycle"]) != int(right["cycle"]) or float(left["time"]) != float(right["time"]):
        raise AssertionError("field times/cycles differ")
    if set(left["var_names"]) != set(right["var_names"]):
        raise AssertionError("field variable sets differ")
    li = {tuple(int(x) for x in row): i for i, row in enumerate(left["mb_logical"])}
    ri = {tuple(int(x) for x in row): i for i, row in enumerate(right["mb_logical"])}
    if li.keys() != ri.keys():
        raise AssertionError("leaf-mesh logical locations differ")
    result = {}
    for var in left["var_names"]:
        max_abs = max_rel = 0.0
        for key in li:
            a = np.asarray(left["mb_data"][var][li[key]], dtype=np.float64)
            b = np.asarray(right["mb_data"][var][ri[key]], dtype=np.float64)
            delta = np.abs(a-b)
            max_abs = max(max_abs, float(np.max(delta)))
            scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), atol)
            max_rel = max(max_rel, float(np.max(delta/scale)))
            if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False):
                raise AssertionError(f"{var} differs: max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
        result[var] = {"max_abs": max_abs, "max_rel": max_rel}
    return result


def last_tracker(root: Path, index: int):
    path = latest(root, f"*.co_{index}.txt")
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            rows.append([float(x) for x in line.split()])
    if not rows:
        raise ValueError(f"no tracker rows in {path}")
    return rows[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--restarted", required=True, type=Path)
    parser.add_argument("--athenak", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    # Athena binary/VTK validation outputs are float32.  These bounds are tight enough
    # to expose decomposition-dependent evolution changes while allowing a few ulps of
    # MPI reduction/reordering noise in two otherwise equivalent steps.
    parser.add_argument("--field-rtol", type=float, default=1e-5)
    parser.add_argument("--field-atol", type=float, default=1e-8)
    parser.add_argument("--particle-atol", type=float, default=5e-7)
    args = parser.parse_args()
    sys.path.insert(0, str(args.athenak.resolve()/"vis"/"python"))
    import bin_convert

    report = {"schema_version": 1, "fields": {}, "particles": {}, "trackers": {}}
    for kind in ("z4c", "con", "tmunu"):
        a = bin_convert.read_binary(str(latest(args.reference, f"*.{kind}_xy.*.bin")))
        b = bin_convert.read_binary(str(latest(args.restarted, f"*.{kind}_xy.*.bin")))
        report["fields"][kind] = field_comparison(a, b, args.field_rtol, args.field_atol)
    ta, xa, ua, taga, ma = read_particles(latest(args.reference, "*.part.vtk"))
    tb, xb, ub, tagb, mb = read_particles(latest(args.restarted, "*.part.vtk"))
    oa, ob = np.argsort(taga), np.argsort(tagb)
    if ta != tb or not np.array_equal(taga[oa], tagb[ob]):
        raise AssertionError("particle times or persistent tag sets differ")
    if taga.size != 5_000_000 or len(np.unique(taga)) != 5_000_000:
        raise AssertionError("particle tag census is not exactly 0..4,999,999 unique")
    if not np.array_equal(taga[oa], np.arange(5_000_000)):
        raise AssertionError("particle tags do not cover the exact deterministic range")
    errors = {
        "position": float(np.max(np.abs(xa[oa]-xb[ob]))),
        "covariant_momentum": float(np.max(np.abs(ua[oa]-ub[ob]))),
        "rest_mass": float(np.max(np.abs(ma[oa]-mb[ob]))),
    }
    if any(value > args.particle_atol for value in errors.values()):
        raise AssertionError(f"particle restart mismatch {errors}")
    report["particles"] = {"time": ta, "count": int(taga.size), "max_abs": errors,
                           "tag_ranges": {"envelope": [0, 2_999_999],
                                          "left": [3_000_000, 3_999_999],
                                          "right": [4_000_000, 4_999_999]}}
    for index in (0, 1):
        a, b = last_tracker(args.reference, index), last_tracker(args.restarted, index)
        if int(a[0]) != int(b[0]) or abs(a[1]-b[1]) > 1e-12:
            raise AssertionError(f"tracker {index} time/cycle differs")
        # Persistent identity/motion state is columns 3--13 plus the irreversible
        # horizon-tracking flag (17).  Core count/mass and lapse minimum are live
        # observations, not checkpoint state; with tracker_walk_every > 1 the first
        # post-restart row is deliberately an unavailable reacquisition gap.
        identity_indices = list(range(2, 13)) + [16]
        if not all(math.isfinite(a[i]) and math.isfinite(b[i])
                   for i in identity_indices):
            raise AssertionError(f"tracker {index} persistent identity is non-finite")
        delta = max(abs(a[i]-b[i]) for i in identity_indices)
        if delta > args.particle_atol or int(a[12]) != int(b[12]) \
                or int(a[16]) != int(b[16]):
            raise AssertionError(f"tracker {index} persistent identity differs by {delta}")
        live_a = {"core_count": a[13], "core_rest_mass": a[14],
                  "lapse_min": a[15] if math.isfinite(a[15]) else None}
        live_b = {"core_count": b[13], "core_rest_mass": b[14],
                  "lapse_min": b[15] if math.isfinite(b[15]) else None}
        valid_a = a[13] > 0 and a[14] > 0 and math.isfinite(a[15])
        valid_b = b[13] > 0 and b[14] > 0 and math.isfinite(b[15])
        if not valid_a:
            raise AssertionError(f"tracker {index} uninterrupted live diagnostic invalid")
        report["trackers"][str(index)] = {
            "cycle": int(a[0]), "time": a[1], "persistent_identity_max_abs": delta,
            "source": int(a[12]), "horizon_tracking": bool(int(a[16])),
            "uninterrupted_live": live_a, "restarted_live": live_b,
            "restarted_live_valid": valid_b,
            "reacquisition_gap": not valid_b,
            "excluded_from_physical_claims_until_live_reacquisition": not valid_b,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"restart_comparison": "pass", "particle_max_abs": errors},
                     sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
