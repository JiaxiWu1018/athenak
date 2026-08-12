#!/usr/bin/env python3
"""Validate the Python reference sampler against an AthenaK t=0 particle dump.

The pre-evolution multipole/CDF sweep is computed by
``cluster_sampler_reference.py``.  That is only trustworthy if it reproduces
what the solver actually placed, so this script compares the two particle sets
tag-by-tag.  The .part.vtk dump stores float32, so agreement is checked at
single-precision tolerance on positions and covariant velocities.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

import cluster_sampler_reference as ref


def read_particle_vtk(path):
    blob = path.read_bytes()
    time = float(re.search(rb"time=\s*([-+0-9.eE]+)", blob).group(1))
    match = re.search(rb"POINTS\s+(\d+)\s+float", blob)
    n = int(match.group(1))

    def block(marker, count):
        offset = blob.find(marker)
        if offset < 0:
            raise ValueError(f"{path}: missing block {marker!r}")
        start = blob.find(b"\n", offset + len(marker)) + 1
        return np.frombuffer(blob[start:start + 4*count], dtype=">f4").astype(float)

    pos = block(match.group(0), 3*n).reshape(n, 3)
    vel = block(b"VECTORS prtcl_vel float", 3*n).reshape(n, 3)
    tag = block(b"SCALARS ptag float\nLOOKUP_TABLE default", n).astype(np.int64)
    mass = block(b"SCALARS prtcl_mass float\nLOOKUP_TABLE default", n)
    order = np.argsort(tag)
    return time, pos[order], vel[order], tag[order], mass[order]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("vtk", type=Path)
    ap.add_argument("--sampler", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--radius-over-mass", type=float, required=True)
    ap.add_argument("--nradial", type=int, required=True)
    ap.add_argument("--nangular", type=int, required=True)
    ap.add_argument("--xi", type=float, default=1.0)
    ap.add_argument("--mass", type=float, default=1.0)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    time, pos, vel, tag, mass = read_particle_vtk(args.vtk)
    r = ref.realize(args.sampler, args.seed, mass=args.mass,
                    radius_over_mass=args.radius_over_mass, xi=args.xi,
                    nradial=args.nradial, nangular=args.nangular, octahedral=True)
    if len(tag) != r["npart"]:
        raise SystemExit(f"particle count {len(tag)} != reference {r['npart']}")
    if not np.array_equal(tag, r["tag"]):
        raise SystemExit("tag sets differ between dump and reference")

    scale = max(np.abs(pos).max(), 1.0)
    vscale = max(np.abs(vel).max(), 1e-30)
    dpos = np.abs(pos - r["pos"]).max() / scale
    dvel = np.abs(vel - r["vel"]).max() / vscale
    dmass = abs(mass.mean() - r["particle_mass"]) / r["particle_mass"]
    # float32 dumps: 2^-23 ~ 1.2e-7 relative, allow a few ulp of accumulation
    tol = 5.0e-6
    ok = bool(dpos < tol and dvel < tol and dmass < tol)
    result = {
        "vtk": str(args.vtk), "time": time, "sampler": args.sampler,
        "seed": args.seed, "npart": int(len(tag)),
        "max_rel_position_error": float(dpos),
        "max_rel_velocity_error": float(dvel),
        "rel_particle_mass_error": float(dmass),
        "tolerance": tol, "agrees": ok,
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
