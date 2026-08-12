#!/usr/bin/env python3
"""Fixed-background smoke analysis for every sampler.

On the frozen analytic metric each particle is an exact circular geodesic, so
|L| and the Killing energy are conserved for every realization regardless of
how it was sampled.  Any drift here therefore indicates a pusher or
coordinate-conversion error in the new sampling path rather than live-field
feedback.  The same quantities under a live field are the physics measurement.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


def read_vtk(path):
    blob = path.read_bytes()
    time = float(re.search(rb"time=\s*([-+0-9.eE]+)", blob).group(1))
    m = re.search(rb"POINTS\s+(\d+)\s+float", blob)
    n = int(m.group(1))

    def block(marker, count, required=True):
        off = blob.find(marker)
        if off < 0:
            if required:
                raise ValueError(f"{path}: missing {marker!r}")
            return None
        start = blob.find(b"\n", off + len(marker)) + 1
        return np.frombuffer(blob[start:start + 4*count], dtype=">f4").astype(float)

    pos = block(m.group(0), 3*n).reshape(n, 3)
    vel = block(b"VECTORS prtcl_vel float", 3*n).reshape(n, 3)
    tag = block(b"SCALARS ptag float\nLOOKUP_TABLE default", n).astype(np.int64)
    energy = block(b"SCALARS prtcl_energy float\nLOOKUP_TABLE default", n)
    mass = block(b"SCALARS prtcl_mass float\nLOOKUP_TABLE default", n)
    order = np.argsort(tag)
    return dict(time=time, pos=pos[order], vel=vel[order], tag=tag[order],
                energy=energy[order], mass=mass[order])


def analyze(run_dir):
    files = sorted(Path(run_dir).glob("pvtk/*.part.vtk"))
    if not files:
        return None
    dumps = [read_vtk(f) for f in files]
    dumps.sort(key=lambda d: d["time"])
    d0, dn = dumps[0], dumps[-1]
    if not np.array_equal(d0["tag"], dn["tag"]):
        common = np.intersect1d(d0["tag"], dn["tag"])
        i0 = np.isin(d0["tag"], common)
        i1 = np.isin(dn["tag"], common)
    else:
        i0 = i1 = np.ones(len(d0["tag"]), bool)

    def Labs(d, sel):
        L = np.cross(d["pos"][sel], d["vel"][sel])
        return np.linalg.norm(L, axis=1)

    L0, L1 = Labs(d0, i0), Labs(dn, i1)
    good = L0 > 0
    dL = np.abs(L1[good] - L0[good]) / L0[good]
    e0, e1 = d0["energy"][i0], dn["energy"][i1]
    egood = np.abs(e0) > 0
    dE = np.abs(e1[egood] - e0[egood]) / np.abs(e0[egood])
    r0 = np.linalg.norm(d0["pos"][i0], axis=1)
    r1 = np.linalg.norm(dn["pos"][i1], axis=1)
    return {
        "run": Path(run_dir).name,
        "n_dumps": len(dumps),
        "t_initial": d0["time"], "t_final": dn["time"],
        "n_initial": int(len(d0["tag"])), "n_final": int(len(dn["tag"])),
        "n_nonfinite_final": int(np.sum(~np.isfinite(dn["pos"]).all(axis=1))),
        "dL_median": float(np.median(dL)), "dL_p95": float(np.percentile(dL, 95)),
        "dL_max": float(dL.max()),
        "dE_median": float(np.median(dE)), "dE_p95": float(np.percentile(dE, 95)),
        "dE_max": float(dE.max()),
        "dr_median": float(np.median(np.abs(r1 - r0) / np.maximum(r0, 1e-30))),
        "dr_max": float(np.max(np.abs(r1 - r0) / np.maximum(r0, 1e-30))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--gate", type=float, default=2.0e-2,
                    help="max acceptable p95 individual |L| drift on a frozen metric")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in args.runs:
        res = analyze(r)
        if res is None:
            print(f"{r.name}: no particle dumps")
            continue
        res["pass"] = bool(res["dL_p95"] < args.gate
                           and res["n_final"] == res["n_initial"]
                           and res["n_nonfinite_final"] == 0)
        rows.append(res)
        print(f"{res['run']:40s} t={res['t_final']:6.2f} "
              f"dL p95={res['dL_p95']:.3e} max={res['dL_max']:.3e} "
              f"dE p95={res['dE_p95']:.3e} N={res['n_final']} "
              f"{'PASS' if res['pass'] else 'FAIL'}")
    with (args.output / "fixed_smoke_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    verdict = {"gate_p95_dL": args.gate,
               "all_pass": all(r["pass"] for r in rows), "runs": rows}
    (args.output / "fixed_smoke_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(f"\nall_pass = {verdict['all_pass']}")


if __name__ == "__main__":
    main()
