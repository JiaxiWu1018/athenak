#!/usr/bin/env python3
"""Generate a single-particle NRPIC initial-condition HDF5 file.

Useful for the Stage-1 energy check: place one particle at a known position with zero
velocity (u_i = 0) and confirm the code recovers IPEN = lapse alpha at that point
(=1 in flat space, =1/sqrt(1+2M/r) in Schwarzschild-Kerr-Schild).

Example:
  python gen_single_particle.py --out one.h5 --x 6.0 --y 0 --z 0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _prtcl_io import write_particle_table  # noqa: E402


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--out", default="single_particle.h5", help="output HDF5 file")
  ap.add_argument("--x", type=float, default=0.0, help="x position")
  ap.add_argument("--y", type=float, default=0.0, help="y position")
  ap.add_argument("--z", type=float, default=0.0, help="z position")
  ap.add_argument("--ux", type=float, default=0.0, help="covariant u_x")
  ap.add_argument("--uy", type=float, default=0.0, help="covariant u_y")
  ap.add_argument("--uz", type=float, default=0.0, help="covariant u_z")
  ap.add_argument("--mass", type=float, default=None,
                  help="per-particle rest mass (omit to use <particles> mass)")
  args = ap.parse_args()

  mass = None if args.mass is None else [args.mass]
  write_particle_table(args.out, [args.x], [args.y], [args.z],
                       [args.ux], [args.uy], [args.uz], mass=mass)


if __name__ == "__main__":
  main()
