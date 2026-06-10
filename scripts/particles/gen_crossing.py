#!/usr/bin/env python3
"""Generate the HDF5 twin of part_crossing's LATTICE mode (NRPIC Stage 3a(b)).

Emits the same npx x npy x npz lattice with velocities cycling all 26 neighbor
directions (plus a rest particle every 27th, by lattice-index tag) that
src/pgen/part_crossing.cpp mode=lattice creates internally. Used to cross-check the
<particles> init=file path against init=pgen: the two runs must produce identical
per-tag trajectories (file rows are tagged by row index = the same lattice index).

Note: the reader stores ux,uy,uz into the covariant-velocity slots, which the drift
pusher reads as coordinate velocities -- identical to what the pgen does directly.

Usage:
  python3 gen_crossing.py out.h5 [--np 8 8 8] [--vmax 1.0]
          [--xmin -0.5 -0.5 -0.5] [--xmax 0.5 0.5 0.5]
"""
import argparse

import numpy as np

from _prtcl_io import write_particle_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--np", nargs=3, type=int, default=[8, 8, 8])
    ap.add_argument("--vmax", type=float, default=1.0)
    ap.add_argument("--xmin", nargs=3, type=float, default=[-0.5, -0.5, -0.5])
    ap.add_argument("--xmax", nargs=3, type=float, default=[0.5, 0.5, 0.5])
    args = ap.parse_args()

    npx, npy, npz = args.np
    # direction table: all (a,b,c) in {-1,0,1}^3 minus rest, lexicographic order
    # (c outermost), matching part_crossing.cpp mode=lattice
    dirs = [(a, b, c) for c in (-1, 0, 1) for b in (-1, 0, 1) for a in (-1, 0, 1)
            if (a, b, c) != (0, 0, 0)]
    ncyc = len(dirs) + 1   # +1 = rest particle

    xs, ys, zs, vx, vy, vz = [], [], [], [], [], []
    for k in range(npz):
        for j in range(npy):
            for i in range(npx):
                xs.append(args.xmin[0] + (i + 0.5) * (args.xmax[0] - args.xmin[0]) / npx)
                ys.append(args.xmin[1] + (j + 0.5) * (args.xmax[1] - args.xmin[1]) / npy)
                zs.append(args.xmin[2] + (k + 0.5) * (args.xmax[2] - args.xmin[2]) / npz)
                tag = i + npx * (j + npy * k)
                idir = tag % ncyc
                if idir < len(dirs):
                    a, b, c = dirs[idir]
                    f = args.vmax / np.sqrt(abs(a) + abs(b) + abs(c))
                    vx.append(f * a), vy.append(f * b), vz.append(f * c)
                else:
                    vx.append(0.0), vy.append(0.0), vz.append(0.0)

    write_particle_table(args.out, xs, ys, zs, vx, vy, vz)
    print(f"{args.out}: {len(xs)} particles ({npx}x{npy}x{npz} lattice, "
          f"vmax={args.vmax}, {ncyc}-cycle directions)")


if __name__ == "__main__":
    main()
