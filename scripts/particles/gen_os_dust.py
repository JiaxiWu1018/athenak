#!/usr/bin/env python3
"""Oppenheimer-Snyder pressureless-dust collapse initial conditions -- STAGE 4 (not yet
implemented).

The headline NRPIC validation is OS collapse of a uniform-density dust ball to a black
hole, compared against the exact GR solution. That requires the Stage-4 feedback machinery
(particle stress-energy -> Tmunu -> Z4c) plus matching interior+exterior initial data, so it
is intentionally left as a stub here. When implemented it will emit the same HDF5 contract as
the other generators (see _prtcl_io.write_particle_table): uniform-density sphere of N
particles, each of mass M/N, with the OS initial 4-velocity field.
"""
import sys


def main():
  sys.exit("gen_os_dust.py is a Stage-4 placeholder and is not yet implemented.")


if __name__ == "__main__":
  main()
