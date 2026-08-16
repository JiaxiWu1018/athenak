#!/usr/bin/env python3
"""Generate a single-particle geodesic in a Cartesian Kerr-Schild background.

Produces the NRPIC HDF5 IC contract: position (x, y, z) and covariant spatial
four-velocity (ux, uy, uz). Two modes are supported:

  rest      : particle momentarily at rest (u_i = 0) at areal radius r0 on the
              x-axis. In a stationary spacetime -u_t is conserved, so IPEN should
              hold at alpha(r0); the particle then falls radially inward.
  circular  : equatorial circular geodesic at areal radius r0. Constant-r
              equatorial motion is a circle of Cartesian radius
              x0 = sqrt(r0^2 + a^2) in the x-y plane, with
              dphi/dt = Omega = sign*sqrt(M)/(r0^1.5 + sign*a*sqrt(M)). We build
              the coordinate 4-velocity direction (1,0,x0*Omega,0), normalize it
              with the local KS 4-metric, and lower indices to get u_i. The radius,
              -u_t (energy), and u_phi (angular momentum) should remain constant.

The Cartesian Kerr-Schild metric uses g = eta + 2H ell ell, matching AthenaK's
Cartesian Kerr-Schild coordinates. M=1 is the AthenaK convention.

Example:
  python gen_geodesic_orbit.py --mode circular --a 0.0 --r0 10 --out schw_circ_r10.h5
  python gen_geodesic_orbit.py --mode rest     --a 0.0 --r0 10 --out schw_rest_r10.h5
  python gen_geodesic_orbit.py --mode circular --a 0.5 --r0 8  --out kerr_circ_r8.h5
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _prtcl_io import write_particle_table  # noqa: E402


def ks_metric(x, y, z, M, a):
    """Return the Cartesian Kerr-Schild metric and radius at ``(x, y, z)``."""
    rho2 = x * x + y * y + z * z
    r2 = 0.5 * ((rho2 - a * a) + np.sqrt((rho2 - a * a) ** 2 + 4.0 * a * a * z * z))
    r = np.sqrt(r2)
    H = M * r2 * r / (r2 * r2 + a * a * z * z)  # M r^3 / (r^4 + a^2 z^2)
    ell = np.array(
        [1.0, (r * x + a * y) / (r2 + a * a), (r * y - a * x) / (r2 + a * a), z / r]
    )
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    g = eta + 2.0 * H * np.outer(ell, ell)
    return g, r


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=["rest", "circular"], default="circular")
    ap.add_argument("--M", type=float, default=1.0, help="BH mass (AthenaK fixes M=1)")
    ap.add_argument("--a", type=float, default=0.0, help="BH spin parameter")
    ap.add_argument(
        "--r0", type=float, default=10.0, help="areal radius of the orbit/start"
    )
    ap.add_argument(
        "--sign", type=float, default=1.0, help="+1 prograde, -1 retrograde (circular)"
    )
    ap.add_argument("--mass", type=float, default=None, help="per-particle rest mass")
    ap.add_argument("--out", default="geodesic_ic.h5")
    args = ap.parse_args()
    M, a, r0 = args.M, args.a, args.r0

    # equatorial start point on the x-axis: Cartesian radius x0 = sqrt(r0^2 + a^2)
    x0 = np.sqrt(r0 * r0 + a * a)
    pos = np.array([x0, 0.0, 0.0])
    g, _ = ks_metric(pos[0], pos[1], pos[2], M, a)
    alpha = 1.0 / np.sqrt(-np.linalg.inv(g)[0, 0])  # lapse = 1/sqrt(-g^{tt})

    if args.mode == "rest":
        u_dn = np.array([0.0, 0.0, 0.0])
        E = alpha  # with u_i=0, -u_t = alpha
        L = 0.0
        print(
            f"[rest] r0={r0}  x0={x0:.6f}  alpha(r0)={alpha:.6f}"
            f"  -> IPEN should hold at {alpha:.6f}"
        )
    else:
        Omega = args.sign * np.sqrt(M) / (r0**1.5 + args.sign * a * np.sqrt(M))
        udir = np.array([1.0, 0.0, x0 * Omega, 0.0])  # (u^t,u^x,u^y,u^z) direction
        norm = -(udir @ g @ udir)
        if norm <= 0:
            raise SystemExit(
                f"no timelike circular orbit at r0={r0}, a={a} (norm={norm:.3e}); "
                f"try larger r0 (must exceed the photon/ISCO radius)"
            )
        ut = 1.0 / np.sqrt(norm)
        u_up = ut * udir
        u_dn4 = g @ u_up  # lower all indices
        u_dn = u_dn4[1:4]
        E = -u_dn4[0]
        L = pos[0] * u_dn[1] - pos[1] * u_dn[0]  # u_phi = x u_y - y u_x
        print(f"[circular] r0={r0} a={a} Omega={Omega:.6f}  x0={x0:.6f}")
        print(
            f"           u_i=({u_dn[0]:.6e},{u_dn[1]:.6e},{u_dn[2]:.6e})"
        )
        print(
            f"           conserved   E=-u_t={E:.6f}   L=u_phi={L:.6f}"
            f"   gamma_t=u^t={ut:.6f}"
        )

    write_particle_table(
        args.out,
        [pos[0]],
        [pos[1]],
        [pos[2]],
        [u_dn[0]],
        [u_dn[1]],
        [u_dn[2]],
        mass=None if args.mass is None else [args.mass],
    )


if __name__ == "__main__":
    main()
