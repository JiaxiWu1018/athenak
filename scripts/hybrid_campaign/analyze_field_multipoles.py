#!/usr/bin/env python3
"""Deposited-density and metric multipoles through l=8, in three dimensions.

Two distinct fields are decomposed:

  * the deposited matter density, reconstructed by repeating the code's CIC
    deposition of the particle dump onto a uniform grid at the finest mesh
    spacing.  The solver only writes a z=0 tmunu slice, so a full 3D spherical
    decomposition of the deposited source is obtained by redoing the deposition
    from the particle data rather than from a slice.

  * the evolved metric, taken from the 3D coarsened-ADM output, using the
    conformal factor psi^4 = A^2, whose deviation from its spherical average is
    the geometric response to the matter multipoles.

For each field the multipole amplitude at radius shell u is

    P_l(u) = sqrt( sum_m |a_lm(u)|^2 ) / a_00(u),

with a_lm obtained by least-squares projection onto real spherical harmonics
sampled at the cell directions in that shell.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/work1/eliasmost/jiaxiwu/athenak_sampler_20260801/vis/python")
from initial_realization_diagnostics import real_sph_harm_matrix  # noqa: E402


def read_vtk_positions(path):
    blob = path.read_bytes()
    time = float(re.search(rb"time=\s*([-+0-9.eE]+)", blob).group(1))
    m = re.search(rb"POINTS\s+(\d+)\s+float", blob)
    n = int(m.group(1))
    off = blob.find(m.group(0))
    start = blob.find(b"\n", off + len(m.group(0))) + 1
    pos = np.frombuffer(blob[start:start + 12*n], dtype=">f4").astype(float)
    pos = pos.reshape(n, 3)
    mo = b"SCALARS prtcl_mass float\nLOOKUP_TABLE default"
    off = blob.find(mo)
    start = blob.find(b"\n", off + len(mo)) + 1
    mass = np.frombuffer(blob[start:start + 4*n], dtype=">f4").astype(float)
    return time, pos, mass


def cic_deposit(pos, weight, half, ncell):
    """Cloud-in-cell deposition onto a uniform ncell^3 grid spanning +-half."""
    dx = 2.0 * half / ncell
    g = (pos + half) / dx - 0.5
    i0 = np.floor(g).astype(np.int64)
    f = g - i0
    grid = np.zeros((ncell, ncell, ncell))
    for dx_i in (0, 1):
        wx = f[:, 0] if dx_i else 1.0 - f[:, 0]
        ix = i0[:, 0] + dx_i
        for dy_i in (0, 1):
            wy = f[:, 1] if dy_i else 1.0 - f[:, 1]
            iy = i0[:, 1] + dy_i
            for dz_i in (0, 1):
                wz = f[:, 2] if dz_i else 1.0 - f[:, 2]
                iz = i0[:, 2] + dz_i
                ok = ((ix >= 0) & (ix < ncell) & (iy >= 0) & (iy < ncell)
                      & (iz >= 0) & (iz < ncell))
                np.add.at(grid, (ix[ok], iy[ok], iz[ok]),
                          (weight * wx * wy * wz)[ok])
    return grid / dx ** 3


def shell_multipoles(values, x, y, z, edges, lmax):
    """Project a cell-sampled field onto real Y_lm within each radial shell."""
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(np.clip(np.divide(z, np.maximum(r, 1e-30)), -1.0, 1.0))
    phi = np.arctan2(y, x)
    names, Y = real_sph_harm_matrix(theta, phi, lmax)
    rows = []
    for i in range(len(edges) - 1):
        sel = (r >= edges[i]) & (r < edges[i + 1])
        if sel.sum() < 4 * (lmax + 1) ** 2:
            continue
        v = values[sel]
        Ys = Y[:, sel]
        # least squares: solve for a_lm minimizing |Y^T a - v|
        a, *_ = np.linalg.lstsq(Ys.T, v, rcond=None)
        a00 = a[0]
        if not np.isfinite(a00) or a00 == 0:
            continue
        rec = {"r_lo": float(edges[i]), "r_hi": float(edges[i + 1]),
               "ncell": int(sel.sum()), "a00": float(a00)}
        for l in range(1, lmax + 1):
            idx = [k for k, (ll, _) in enumerate(names) if ll == l]
            rec[f"P{l}"] = float(np.sqrt(np.sum(a[idx] ** 2)) / abs(a00))
        rows.append(rec)
    return rows


def deposited_multipoles(vtk_path, r_surface_iso, lmax=8, ncell=96):
    time, pos, mass = read_vtk_positions(vtk_path)
    finite = np.isfinite(pos).all(axis=1)
    pos, mass = pos[finite], mass[finite]
    half = 1.25 * r_surface_iso
    grid = cic_deposit(pos, mass, half, ncell)
    dx = 2.0 * half / ncell
    c = -half + (np.arange(ncell) + 0.5) * dx
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    edges = np.linspace(0.10 * r_surface_iso, 0.95 * r_surface_iso, 9)
    rows = shell_multipoles(grid.ravel(), X.ravel(), Y.ravel(), Z.ravel(),
                            edges, lmax)
    agg = {}
    for l in range(1, lmax + 1):
        vals = [r[f"P{l}"] for r in rows if f"P{l}" in r]
        agg[f"P{l}"] = float(np.sqrt(np.mean(np.square(vals)))) if vals else float("nan")
    return {"time": time, "n_particles": int(finite.sum()),
            "grid_ncell": ncell, "grid_dx": dx,
            "shells": rows, "rms_over_shells": agg}


def metric_multipoles(cbin_path, r_surface_iso, lmax=8):
    import bin_convert
    data = bin_convert.read_coarsened_binary(str(cbin_path))
    names = [n.decode() if isinstance(n, bytes) else n for n in data["var_names"]]
    var = "adm_psi4" if "adm_psi4" in names else names[0]
    arr = np.asarray(data["mb_data"][var])
    geo = np.asarray(data["mb_geometry"])
    logical = np.asarray(data["mb_logical"])
    level = logical[:, 3] if logical.ndim == 2 and logical.shape[1] >= 4 else None
    nblk, nz, ny, nx = arr.shape
    xs, ys, zs, vs, lv = [], [], [], [], []
    for b in range(nblk):
        x1min, x1max, x2min, x2max, x3min, x3max = geo[b][:6]
        x = x1min + (np.arange(nx) + 0.5) * (x1max - x1min) / nx
        y = x2min + (np.arange(ny) + 0.5) * (x2max - x2min) / ny
        z = x3min + (np.arange(nz) + 0.5) * (x3max - x3min) / nz
        Z, Yg, Xg = np.meshgrid(z, y, x, indexing="ij")
        xs.append(Xg.ravel()); ys.append(Yg.ravel()); zs.append(Z.ravel())
        vs.append(arr[b].ravel())
        lv.append(np.full(Xg.size, level[b] if level is not None else 0))
    x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    v, lev = np.concatenate(vs), np.concatenate(lv)
    keep = lev == lev.max()
    x, y, z, v = x[keep], y[keep], z[keep], v[keep]
    edges = np.linspace(0.10 * r_surface_iso, 0.95 * r_surface_iso, 9)
    rows = shell_multipoles(v, x, y, z, edges, lmax)
    agg = {}
    for l in range(1, lmax + 1):
        vals = [r[f"P{l}"] for r in rows if f"P{l}" in r]
        agg[f"P{l}"] = float(np.sqrt(np.mean(np.square(vals)))) if vals else float("nan")
    return {"time": float(data["time"]), "variable": var, "shells": rows,
            "rms_over_shells": agg}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--lmax", type=int, default=8)
    ap.add_argument("--skip-metric", action="store_true")
    args = ap.parse_args()

    manifest = json.loads((args.root / "inputs" / "input_manifest.json").read_text())
    by_name = {c["name"]: c for c in manifest}
    args.output.mkdir(parents=True, exist_ok=True)
    out = {}
    for case in args.cases:
        meta = by_name[case]
        q, mass = meta["q"], 1.0
        sq = math.sqrt(1.0 - 2.0 / q)
        r0 = 0.5 * q * mass * (1.0 - 1.0 / q + sq)
        vtks = sorted((args.root / "runs" / case / "pvtk").glob("*.part.vtk"))
        if not vtks:
            print(f"{case}: no particle dumps")
            continue
        rec = {"case": case, "sampler": meta["sampler"], "seed": meta["seed"],
               "model": meta["model"], "q": q, "r_surface_iso": r0}
        rec["deposited_initial"] = deposited_multipoles(vtks[0], r0, args.lmax)
        if len(vtks) > 1:
            rec["deposited_final"] = deposited_multipoles(vtks[-1], r0, args.lmax)
        if not args.skip_metric:
            cbins = sorted((args.root / "runs" / case / "cbin_adm_1")
                           .glob("*.cbin"))
            try:
                if cbins:
                    rec["metric_initial"] = metric_multipoles(cbins[0], r0, args.lmax)
                if len(cbins) > 1:
                    rec["metric_final"] = metric_multipoles(cbins[-1], r0, args.lmax)
            except Exception as exc:          # noqa: BLE001
                rec["metric_error"] = str(exc)
        out[case] = rec
        d0 = rec["deposited_initial"]["rms_over_shells"]
        dn = rec.get("deposited_final", rec["deposited_initial"])["rms_over_shells"]
        print(f"{case:44s} deposited P4 {d0['P4']:.4e} -> {dn['P4']:.4e}  "
              f"P8 {d0['P8']:.4e} -> {dn['P8']:.4e}")
    (args.output / "field_multipoles.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {len(out)} cases")


if __name__ == "__main__":
    main()
