#!/usr/bin/env python3
"""Compare the deposited T_munu source with the analytic continuum source.

At t=0 the analytic model has a uniform normal-frame energy density inside the
surface,

    rho(r_s) = 3 M / (4 pi R^3)   for r_s <= R,   0 outside,

while the code's matter source is the CIC deposition of the finite-particle
realization (undensitized ADM E, the discrete counterpart of rho).  The
mismatch between the two is one of the quantities the sampler comparison needs,
because the metric is the exact continuum mean-field solution and is never
re-solved against the realized particle source.

This reads the z=0 tmunu slice written by the solver and reports the radial
profile of deposited E, its deviation from the analytic rho, and the angular
multipoles of that deviation inside the cluster.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/work1/eliasmost/jiaxiwu/athenak_sampler_20260801/vis/python")
import bin_convert  # noqa: E402


def load_slice(path, varname="tmunu_E"):
    """Flatten an AthenaK MeshBlock-structured .bin slice into cell lists.

    mb_geometry holds (x1min, x1max, x2min, x2max, x3min, x3max) per block and
    mb_data[var] is (nblocks, nz, ny, nx).  For a z=0 slice output nz is 1 and
    the block's full x3 extent is retained in the header, so the single cell is
    placed at the requested slice coordinate rather than the block centre.
    Blocks live on different refinement levels, so the per-cell spacing is
    returned alongside the coordinates.
    """
    data = bin_convert.read_binary(str(path))
    names = [n.decode() if isinstance(n, bytes) else n for n in data["var_names"]]
    if varname not in names:
        raise SystemExit(f"{path}: variable {varname!r} not found; have {names}")
    arr = np.asarray(data["mb_data"][varname])
    geo = np.asarray(data["mb_geometry"])
    logical = np.asarray(data["mb_logical"])
    level = logical[:, 3] if logical.ndim == 2 and logical.shape[1] >= 4 else None
    nblk, nz, ny, nx = arr.shape
    xs, ys, zs, vals, dxs, levs = [], [], [], [], [], []
    for b in range(nblk):
        x1min, x1max, x2min, x2max, x3min, x3max = geo[b][:6]
        dx1 = (x1max - x1min) / nx
        dx2 = (x2max - x2min) / ny
        x = x1min + (np.arange(nx) + 0.5) * dx1
        y = x2min + (np.arange(ny) + 0.5) * dx2
        z = (np.zeros(1) if nz == 1
             else x3min + (np.arange(nz) + 0.5) * (x3max - x3min) / nz)
        Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
        xs.append(X.ravel()); ys.append(Y.ravel()); zs.append(Z.ravel())
        vals.append(arr[b].ravel())
        dxs.append(np.full(X.size, dx1))
        levs.append(np.full(X.size, level[b] if level is not None else 0))
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(zs),
            np.concatenate(vals), np.concatenate(dxs), np.concatenate(levs),
            float(data["time"]))


def analyze(path, mass, q, nbin=48):
    x, y, z, E, dx, lev, time = load_slice(path)
    # Refinement levels overlap in the slice output, so restrict to the finest
    # level.  For this mesh level 5 spans |x|<=5.4M and the cluster surface is
    # at r_iso=5.05M, so the finest level covers the whole matter region and no
    # cell is counted twice.
    finest = lev == lev.max()
    x, y, z, E, dx = x[finest], y[finest], z[finest], E[finest], dx[finest]
    radius = q * mass
    sq = math.sqrt(1.0 - 2.0 / q)
    r0 = 0.5 * radius * (1.0 - 1.0 / q + sq)
    cnum = (1.0 + sq) * r0 * radius * radius
    riso = np.sqrt(x * x + y * y + z * z)
    # isotropic -> areal radius using the interior conformal factor
    A = np.where(riso <= r0, cnum / (2.0 * r0 ** 3 + mass * riso ** 2),
                 (1.0 + 0.5 * mass / np.maximum(riso, 1e-30)) ** 2)
    rs = A * riso
    rho_c = 3.0 * mass / (4.0 * math.pi * radius ** 3)
    rho_exact = np.where(rs <= radius, rho_c, 0.0)

    inside = rs <= 0.9 * radius          # avoid the surface discontinuity
    dev = (E[inside] - rho_exact[inside]) / rho_c
    edges = np.linspace(0.0, 1.0, nbin + 1)
    u = rs / radius
    prof_u, prof_E, prof_n = [], [], []
    for i in range(nbin):
        sel = (u >= edges[i]) & (u < edges[i + 1])
        if sel.sum() == 0:
            continue
        prof_u.append(float(0.5 * (edges[i] + edges[i + 1])))
        prof_E.append(float(E[sel].mean() / rho_c))
        prof_n.append(int(sel.sum()))

    # angular structure of the deposition error on the z=0 slice: expand the
    # in-cluster deviation in cos(m phi)/sin(m phi) up to m=8
    phi = np.arctan2(y[inside], x[inside])
    modes = {}
    for m in range(1, 9):
        c = float(np.mean(dev * np.cos(m * phi)))
        s = float(np.mean(dev * np.sin(m * phi)))
        modes[f"m{m}"] = math.hypot(c, s)
    return {
        "file": str(path), "time": time, "n_cells_inside": int(inside.sum()),
        "finest_level_cells": int(finest.sum()), "rho_c": rho_c,
        "dev_mean": float(dev.mean()), "dev_rms": float(np.sqrt(np.mean(dev ** 2))),
        "dev_max_abs": float(np.abs(dev).max()),
        "dev_p95_abs": float(np.percentile(np.abs(dev), 95)),
        "azimuthal_modes": modes,
        "radial_profile": {"u": prof_u, "E_over_rho_c": prof_E, "ncell": prof_n},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mass", type=float, default=1.0)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.root / "inputs" / "input_manifest.json").read_text())
    by_name = {c["name"]: c for c in manifest}

    out = {}
    for case in args.cases:
        meta = by_name[case]
        files = sorted((args.root / "runs" / case / "bin").glob("*tmunu*.bin"))
        if not files:
            print(f"{case}: no tmunu slice")
            continue
        try:
            res = analyze(files[0], args.mass, meta["q"])
        except Exception as exc:            # noqa: BLE001 - report and continue
            print(f"{case}: FAILED ({exc})")
            continue
        res.update({"case": case, "sampler": meta["sampler"], "seed": meta["seed"],
                    "q": meta["q"]})
        out[case] = res
        m4 = res["azimuthal_modes"]["m4"]
        print(f"{case:44s} t={res['time']:.2f} dev_rms={res['dev_rms']:.4e} "
              f"dev_p95={res['dev_p95_abs']:.4e} m4={m4:.3e}")
    (args.output / "deposited_source_comparison.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nwrote {len(out)} cases")


if __name__ == "__main__":
    main()
