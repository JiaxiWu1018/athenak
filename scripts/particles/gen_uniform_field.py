#!/usr/bin/env python3
"""Generate a uniform lattice of particles filling a box (strictly interior).

Particles are placed at cell centres of an (n1 x n2 x n3) lattice spanning the box, which
keeps them strictly inside [min, max) -- respecting the reader's half-open MeshBlock
ownership convention so none are dropped/double-counted on a boundary. A constant bulk
4-velocity (covariant u_i) may be set. Useful for exercising the reader's MeshBlock
assignment and multi-rank tags (Stage 3).

Example:
  python gen_uniform_field.py --out field.h5 --n1 16 --n2 16 --n3 16 \
      --x1min -0.5 --x1max 0.5 --x2min -0.5 --x2max 0.5 --x3min -0.5 --x3max 0.5
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _prtcl_io import write_particle_table  # noqa: E402


def centers(lo, hi, n):
    """n cell-centre coordinates strictly inside [lo, hi)."""
    edges = np.linspace(lo, hi, n + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="uniform_field.h5", help="output HDF5 file")
    ap.add_argument("--x1min", type=float, default=-0.5)
    ap.add_argument("--x1max", type=float, default=0.5)
    ap.add_argument("--x2min", type=float, default=-0.5)
    ap.add_argument("--x2max", type=float, default=0.5)
    ap.add_argument("--x3min", type=float, default=-0.5)
    ap.add_argument("--x3max", type=float, default=0.5)
    ap.add_argument("--n1", type=int, default=8, help="particles along x1")
    ap.add_argument("--n2", type=int, default=8, help="particles along x2")
    ap.add_argument("--n3", type=int, default=8, help="particles along x3")
    ap.add_argument("--ux", type=float, default=0.0, help="bulk covariant u_x")
    ap.add_argument("--uy", type=float, default=0.0, help="bulk covariant u_y")
    ap.add_argument("--uz", type=float, default=0.0, help="bulk covariant u_z")
    ap.add_argument("--mass", type=float, default=None,
                    help="uniform per-particle rest mass (omit to use <particles> mass)")
    args = ap.parse_args()

    cx = centers(args.x1min, args.x1max, args.n1)
    cy = centers(args.x2min, args.x2max, args.n2)
    cz = centers(args.x3min, args.x3max, args.n3)
    # deterministic ordering (k slowest, i fastest) -> reproducible global file-index tags
    zz, yy, xx = np.meshgrid(cz, cy, cx, indexing="ij")
    x = xx.ravel()
    y = yy.ravel()
    z = zz.ravel()
    npart = x.size
    ux = np.full(npart, args.ux)
    uy = np.full(npart, args.uy)
    uz = np.full(npart, args.uz)
    mass = None if args.mass is None else np.full(npart, args.mass)
    write_particle_table(args.out, x, y, z, ux, uy, uz, mass=mass)


if __name__ == "__main__":
    main()
