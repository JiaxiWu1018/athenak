#!/usr/bin/env python3
"""Oppenheimer-Snyder pressureless-dust collapse initial conditions (NRPIC Stage 4b).

Emits the standard NRPIC particle HDF5 contract (see _prtcl_io.write_particle_table):
positions x,y,z, covariant 4-velocity u_i = ux,uy,uz, and a PER-PARTICLE rest mass.
The companion pgen ``src/pgen/particles/nr_pic_os.cpp`` sets the matching conformally-flat,
time-symmetric metric from the same (M, R0) -- keep ``<problem> os_mass`` and
``os_radius_over_mass`` IN SYNC with --mass / --radius here; a mismatch invalidates
the t=0 Hamiltonian-constraint comparison.

Physics (geometric units G=c=1; M = ADM mass, R0 = areal surface radius):
  conformally-flat, time-symmetric (K_ij = 0) data, isotropic Cartesian coordinates,
  3-metric gamma_ij = psi^4 delta_ij  =>  sqrt(gamma) = psi^6.
  isotropic surface radius   r0 = (R0/2)(1 - M/R0 + sqrt(1 - 2M/R0))
  interior (r <= r0)  psi^2 = (1 + sqrt(1-2M/R0)) r0 R0^2 / (2 r0^3 + M r^2)
  exterior (r >  r0)  psi   = 1 + M/(2r)              [vacuum Schwarzschild-isotropic]
  The Hamiltonian constraint for time-symmetric data is the flat-space Poisson
  equation  lap psi = -2 pi psi^5 rho.  Plugging the interior psi gives a
  SPATIALLY-UNIFORM normal-observer energy density (this is exactly the homogeneous
  OS dust ball):
      rho0 = 3 M r0 / (pi (1 + sqrt(1-2M/R0))^2 R0^4)
  (-> 3M/(4 pi R0^3), the Newtonian uniform density, as M/R0 -> 0).  Time-symmetric
  => the dust is momentarily at rest => u_i = 0, W = 1, so the deposited source must
  reproduce  E(x) = rho0  inside the ball.

The deposit kernel computes  E(x) = sum_p m_p / (sqrt(gamma) dV)  (W=1 here), so the
required per-particle masses are (this FIXES the legacy "m_p = M/N" bug, which ignores
both the binding energy and the sqrt(gamma) sampling measure):

  scheme "lattice" (deterministic, low noise): particles on a Cartesian lattice of
     spacing h covering the ball; m_p = rho0 * psi^6(x_p) * h^3   (= the proper rest
     mass of one lattice cell).  Sum_p m_p = proper rest mass M_p.

  scheme "mc" (equal mass): N positions drawn with coordinate number density
     proportional to psi^6 (rho0 const => proper-volume measure), each m_p = M_p / N.

Either way  Sum_p m_p = M_p = rho0 * Integral_ball psi^6 d^3x = the PROPER REST MASS,
which EXCEEDS the ADM mass M by the gravitational binding energy (~10% at R0=5-10M).
M (geometry) and M_p (sum of particle masses) are physically distinct -- do not
conflate them.

Examples:
  python gen_os_dust.py --out os.h5 --mass 1.0 --radius 5.0 --scheme lattice --n 64
  python gen_os_dust.py --out os.h5 --mass 1.0 --radius 5.0 --scheme mc --npart 200000
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _prtcl_io import write_particle_table  # noqa: E402


def os_params(M, R0_over_M):
    """Return (R0, r0, C, rho0) for the conformally-flat time-symmetric OS ball."""
    if R0_over_M <= 2.0:
        sys.exit(f"radius R0/M = {R0_over_M} must exceed 2 (dust must start outside 2M)")
    R0 = R0_over_M * M
    om2 = 1.0 - 2.0 / R0_over_M                 # 1 - 2M/R0
    sq = math.sqrt(om2)
    r0 = 0.5 * R0 * (1.0 - 1.0 / R0_over_M + sq)  # isotropic surface radius
    C = (1.0 + sq) * r0 * R0 * R0                 # interior psi^2 numerator
    rho0 = 3.0 * M * r0 / (math.pi * (1.0 + sq) ** 2 * R0 ** 4)
    return R0, r0, C, rho0


def psi_interior(r, M, r0, C):
    """Conformal factor psi(r) for r <= r0 (vectorized)."""
    psi2 = C / (2.0 * r0 ** 3 + M * r * r)
    return np.sqrt(psi2)


def proper_mass(M, r0, C, rho0):
    """M_p = rho0 * Integral_{r<=r0} psi^6 d^3x  via a fine radial quadrature."""
    rr = np.linspace(0.0, r0, 200001)
    psi6 = psi_interior(rr, M, r0, C) ** 6
    integrand = psi6 * 4.0 * math.pi * rr * rr
    # trapezoidal rule (version-independent; np.trapz removed in NumPy 2)
    integral = float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(rr)))
    return rho0 * integral


def gen_lattice(M, r0, C, rho0, n):
    """Cartesian lattice of cell centres in [-r0, r0]^3, kept where r <= r0.

    m_p = rho0 * psi^6(x_p) * h^3 (proper rest mass of one lattice cell)."""
    h = 2.0 * r0 / n
    edges = np.linspace(-r0, r0, n + 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    # deterministic ordering (k slowest, i fastest) -> reproducible file-row tags
    zz, yy, xx = np.meshgrid(c, c, c, indexing="ij")
    x, y, z = xx.ravel(), yy.ravel(), zz.ravel()
    r = np.sqrt(x * x + y * y + z * z)
    inside = r <= r0
    x, y, z, r = x[inside], y[inside], z[inside], r[inside]
    mass = rho0 * psi_interior(r, M, r0, C) ** 6 * h ** 3
    return x, y, z, mass


def gen_mc(M, r0, C, rho0, npart, seed):
    """Equal-mass particles, coordinate number density proportional to psi^6.

    Rejection sampling: propose uniform in the ball, accept with psi^6(r)/psi^6(0)."""
    rng = np.random.default_rng(seed)
    psi6_max = psi_interior(np.array([0.0]), M, r0, C)[0] ** 6
    xs, ys, zs = [], [], []
    got = 0
    while got < npart:
        batch = max(npart - got, 4096)
        p = rng.uniform(-r0, r0, size=(batch, 3))
        r = np.sqrt(np.sum(p * p, axis=1))
        in_ball = r <= r0
        p, r = p[in_ball], r[in_ball]
        acc = rng.uniform(0.0, 1.0, size=r.size) < (
            psi_interior(r, M, r0, C) ** 6 / psi6_max)
        p = p[acc]
        take = min(npart - got, p.shape[0])
        xs.append(p[:take, 0])
        ys.append(p[:take, 1])
        zs.append(p[:take, 2])
        got += take
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    Mp = proper_mass(M, r0, C, rho0)
    mass = np.full(npart, Mp / npart)
    return x, y, z, mass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="os_dust.h5", help="output HDF5 file")
    ap.add_argument("--mass", type=float, default=1.0, help="ADM mass M")
    ap.add_argument("--radius", type=float, default=5.0,
                    help="areal surface radius in units of M (R0/M)")
    ap.add_argument("--scheme", choices=["lattice", "mc"], default="lattice")
    ap.add_argument("--n", type=int, default=64,
                    help="lattice: cell centres per dimension across the diameter 2 r0")
    ap.add_argument("--npart", type=int, default=200000, help="mc: number of particles")
    ap.add_argument("--seed", type=int, default=12345, help="mc: RNG seed")
    args = ap.parse_args()

    M = args.mass
    R0, r0, C, rho0 = os_params(M, args.radius)
    if args.scheme == "lattice":
        x, y, z, mass = gen_lattice(M, r0, C, rho0, args.n)
    else:
        x, y, z, mass = gen_mc(M, r0, C, rho0, args.npart, args.seed)
    npart = x.size
    ux = np.zeros(npart)   # time-symmetric: u_i = 0 (momentarily at rest)
    uy = np.zeros(npart)
    uz = np.zeros(npart)

    write_particle_table(args.out, x, y, z, ux, uy, uz, mass=mass)

    # provenance + the M vs M_p distinction (and consistency-check against the pgen)
    Mp = float(np.sum(mass))
    import h5py
    with h5py.File(args.out, "a") as f:
        f.attrs["os_mass"] = M
        f.attrs["os_radius_over_mass"] = args.radius
        f.attrs["os_scheme"] = args.scheme
        f.attrs["os_r0_isotropic"] = r0
        f.attrs["os_rho0"] = rho0
        f.attrs["os_proper_mass"] = Mp
    print(f"OS dust: M_ADM={M:.6g}, R0={R0:.6g} (={args.radius:g} M), "
          f"r0_iso={r0:.6g}, rho0={rho0:.6g}")
    print(f"  scheme={args.scheme}, N={npart}, sum m_p (proper rest mass) M_p={Mp:.6g} "
          f"= {Mp / M:.4f} x M_ADM (excess = binding energy)")


if __name__ == "__main__":
    main()
