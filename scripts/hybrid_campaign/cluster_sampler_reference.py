#!/usr/bin/env python3
"""Standalone reference implementation of the AthenaK homogeneous-cluster samplers.

This mirrors ``src/pgen/nr_pic_homogeneous_cluster.cpp`` bit-for-bit in the
construction of positions and covariant velocities, so that the cheap
pre-evolution diagnostics (radial CDF error, multipoles, |L| distribution,
net P and J) can be swept over many samplers and seeds without launching the
solver.  It is validated against the executable's own t=0 particle dump by
``validate_reference_sampler.py``; do not trust its numbers unless that
validation has been run for the configuration in question.

Continuum measure realized by every sampler:

    p(u) du  ~  u^2 du / [ W(u) sqrt(1 - 2 C u^2) ],      u = r_s/R,  C = M/R
    v(u)     =  xi sqrt(x/(1-2x)),   x = C u^2,   W = 1/sqrt(1-v^2)
    u_i      =  A W v t_i                     (stored covariant 3-velocity)
"""
import argparse
import json
import math
import numpy as np

UINT64 = np.uint64
TWO53 = float(1 << 53)

# uint64 wraparound is the intended modular arithmetic of SplitMix64, not an error.
np.seterr(over="ignore")


def splitmix64(x):
    """Vectorized SplitMix64, identical to the C++ helper of the same name."""
    x = np.asarray(x, dtype=UINT64)
    x = x + UINT64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> UINT64(30))) * UINT64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> UINT64(27))) * UINT64(0x94D049BB133111EB)
    return x ^ (x >> UINT64(31))


def hash_unit_id(seed, ids, stream):
    """Counter-based uniform on [0,1) keyed by (seed, id, stream)."""
    ids = np.asarray(ids, dtype=UINT64)
    key = splitmix64(UINT64(seed) + UINT64(0x9E3779B97F4A7C15) * (ids + UINT64(1)))
    key = splitmix64(key ^ (UINT64(0xBF58476D1CE4E5B9) * UINT64(stream + 1)))
    return (key >> UINT64(11)).astype(np.float64) / TWO53


def hash_unit(seed, shell, stream):
    """Scalar hash used by the historical per-shell SO(3) rotations."""
    key = UINT64(seed) ^ (UINT64(0xD1B54A32D192ED03) * UINT64(shell + 1))
    key = key ^ (UINT64(0x94D049BB133111EB) * UINT64(stream + 1))
    return float(splitmix64(key) >> UINT64(11)) / TWO53


def shell_rotation(seed, shell):
    """Shoemake uniform-quaternion rotation matrix, as in the pgen."""
    u1 = hash_unit(seed, shell, 0)
    u2 = hash_unit(seed, shell, 1)
    u3 = hash_unit(seed, shell, 2)
    s1, s2 = math.sqrt(1.0 - u1), math.sqrt(u1)
    qx = s1 * math.sin(2.0 * math.pi * u2)
    qy = s1 * math.cos(2.0 * math.pi * u2)
    qz = s2 * math.sin(2.0 * math.pi * u3)
    qw = s2 * math.cos(2.0 * math.pi * u3)
    return np.array([
        [1.0 - 2.0*(qy*qy + qz*qz), 2.0*(qx*qy - qz*qw), 2.0*(qx*qz + qy*qw)],
        [2.0*(qx*qy + qz*qw), 1.0 - 2.0*(qx*qx + qz*qz), 2.0*(qy*qz - qx*qw)],
        [2.0*(qx*qz - qy*qw), 2.0*(qy*qz + qx*qw), 1.0 - 2.0*(qx*qx + qy*qy)],
    ])


_OCT_PERM = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2],
                      [1, 2, 0], [2, 0, 1], [2, 1, 0]])
_OCT_PARITY = np.array([1, -1, -1, 1, 1, -1])


def octahedral_rotate(group_index, a):
    iperm, isign = group_index // 4, group_index % 4
    s0 = -1.0 if (isign & 1) else 1.0
    s1 = -1.0 if (isign & 2) else 1.0
    s2 = float(_OCT_PARITY[iperm]) * s0 * s1
    p = _OCT_PERM[iperm]
    return np.array([s0 * a[p[0]], s1 * a[p[1]], s2 * a[p[2]]])


class Profile:
    """Trapezoidal rest-mass CDF table, identical to ConstructProfile()."""

    def __init__(self, compactness, xi, nradial):
        ntab = max(16384, 128 * nradial)
        h = 1.0 / ntab
        u = np.arange(ntab + 1, dtype=np.float64) * h
        u[0] = 0.0
        x = compactness * u * u
        v2 = xi * xi * x / (1.0 - 2.0 * x)
        inv_w = np.sqrt(np.maximum(1.0 - v2, 0.0))
        f = u * u * inv_w / np.sqrt(1.0 - 2.0 * x)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * h * (f[:-1] + f[1:]))])
        integral = cdf[-1]
        self.u = u
        self.rest_mass_over_m = 3.0 * integral
        self.cdf = cdf / integral
        self.cdf[-1] = 1.0

    def invert(self, p):
        """Piecewise-linear inverse CDF, matching InvertCDF()."""
        p = np.atleast_1d(np.asarray(p, dtype=np.float64))
        hi = np.searchsorted(self.cdf[1:], p, side="left") + 1
        hi = np.clip(hi, 1, len(self.cdf) - 1)
        lo = hi - 1
        denom = self.cdf[hi] - self.cdf[lo]
        frac = np.where(denom > 0.0, (p - self.cdf[lo]) / np.where(denom > 0.0, denom, 1.0), 0.0)
        return self.u[lo] + frac * (self.u[hi] - self.u[lo])


class Geometry:
    def __init__(self, mass, radius_over_mass, xi):
        self.mass = mass
        self.q = radius_over_mass
        self.xi = xi
        self.compactness = 1.0 / radius_over_mass
        self.radius = radius_over_mass * mass
        sq = math.sqrt(1.0 - 2.0 / radius_over_mass)
        self.r0 = 0.5 * self.radius * (1.0 - 1.0 / radius_over_mass + sq)
        self.cnum = (1.0 + sq) * self.r0 * self.radius * self.radius
        self.lapse_prefactor = (1.0 - 2.0 / radius_over_mass) ** 0.75

    def isotropic_radius(self, rs):
        disc = np.maximum(self.cnum**2 - 8.0*self.mass*rs*rs*self.r0**3, 0.0)
        return np.where(rs == 0.0, 0.0,
                        4.0*rs*self.r0**3 / (self.cnum + np.sqrt(disc)))

    def radial_state(self, u):
        """Return (riso, umag, alpha, W, v, A, rs) for dimensionless areal u."""
        rs = self.radius * u
        riso = self.isotropic_radius(rs)
        a = np.where(riso > 0.0, rs / np.where(riso > 0.0, riso, 1.0),
                     self.cnum / (2.0 * self.r0**3))
        x = self.compactness * u * u
        v = self.xi * np.sqrt(x / (1.0 - 2.0 * x))
        w = 1.0 / np.sqrt(1.0 - v * v)
        umag = a * w * v
        alpha = self.lapse_prefactor * (1.0 - 2.0 * x) ** -0.25
        return riso, umag, alpha, w, v, a, rs


SAMPLERS = ("shell_fibonacci", "radial_random", "angular_random",
            "monte_carlo", "stratified_random", "monte_carlo_antithetic",
            "stratified_antithetic")
INDEPENDENT = ("angular_random", "monte_carlo", "stratified_random",
               "monte_carlo_antithetic", "stratified_antithetic")
ANTITHETIC = ("monte_carlo_antithetic", "stratified_antithetic")


def realize(sampler, seed, mass=1.0, radius_over_mass=6.1, xi=1.0,
            nradial=128, nangular=1032, octahedral=True, center=(0.0, 0.0, 0.0)):
    """Return a dict of arrays for the full particle realization, tag-ordered."""
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler {sampler!r}")
    geo = Geometry(mass, radius_over_mass, xi)
    profile = Profile(geo.compactness, xi, nradial)
    npart = 4 * nradial * nangular
    particle_mass = mass * profile.rest_mass_over_m / npart
    seed64 = np.uint64(np.uint32(seed))
    center = np.asarray(center, dtype=np.float64)

    tag = np.arange(npart, dtype=np.int64)
    ir = tag // (4 * nangular)
    ia = (tag // 4) % nangular
    idir = tag % 4

    if sampler in INDEPENDENT:
        draw_id = (tag // 2) if sampler in ANTITHETIC else tag
        if sampler == "angular_random":
            u = profile.invert((ir + 0.5) / nradial)
        elif sampler == "stratified_random":
            u = profile.invert((tag + hash_unit_id(seed64, draw_id, 0)) / npart)
        elif sampler == "stratified_antithetic":
            # one equal-rest-mass stratum per +-t pair: stratum = pair = tag//2
            u = profile.invert(
                (draw_id + hash_unit_id(seed64, draw_id, 0)) / (npart // 2))
        else:
            u = profile.invert(hash_unit_id(seed64, draw_id, 0))
        riso, umag, alpha, w, v, conf_a, rs = geo.radial_state(u)

        cth = 1.0 - 2.0 * hash_unit_id(seed64, draw_id, 1)
        sth = np.sqrt(np.maximum(1.0 - cth*cth, 0.0))
        phi = 2.0 * math.pi * hash_unit_id(seed64, draw_id, 2)
        cph, sph = np.cos(phi), np.sin(phi)
        nvec = np.stack([sth*cph, sth*sph, cth], axis=1)
        eth = np.stack([cth*cph, cth*sph, -sth], axis=1)
        eph = np.stack([-sph, cph, np.zeros_like(sph)], axis=1)
        chi = 2.0 * math.pi * hash_unit_id(seed64, draw_id, 3)
        sign = np.ones(npart)
        if sampler in ANTITHETIC:
            sign = np.where(tag % 2 == 1, -1.0, 1.0)
        tvec = sign[:, None] * (np.cos(chi)[:, None]*eth + np.sin(chi)[:, None]*eph)
    else:
        if sampler == "radial_random":
            site_id = ir.astype(np.uint64) * np.uint64(nangular) + ia.astype(np.uint64)
            u = profile.invert(hash_unit_id(seed64, site_id, 0))
        else:
            u = profile.invert((ir + 0.5) / nradial)
        riso, umag, alpha, w, v, conf_a, rs = geo.radial_state(u)

        golden = 0.5 * (math.sqrt(5.0) - 1.0)
        nfib = nangular // 24 if octahedral else nangular
        ifib = ia // 24 if octahedral else ia
        igrp = ia % 24 if octahedral else np.zeros_like(ia)
        cth = 1.0 - 2.0 * (ifib + 0.5) / nfib
        sth = np.sqrt(np.maximum(1.0 - cth*cth, 0.0))
        phi = 2.0 * math.pi * np.mod(golden * ifib, 1.0)
        cph, sph = np.cos(phi), np.sin(phi)
        n0 = np.stack([sth*cph, sth*sph, cth], axis=1)
        eth0 = np.stack([cth*cph, cth*sph, -sth], axis=1)
        eph0 = np.stack([-sph, cph, np.zeros_like(sph)], axis=1)
        if octahedral:
            nsym = np.empty_like(n0); ethsym = np.empty_like(eth0)
            ephsym = np.empty_like(eph0)
            for g in range(24):
                sel = igrp == g
                if not np.any(sel):
                    continue
                iperm, isign = g // 4, g % 4
                s0 = -1.0 if (isign & 1) else 1.0
                s1 = -1.0 if (isign & 2) else 1.0
                s2 = float(_OCT_PARITY[iperm]) * s0 * s1
                p = _OCT_PERM[iperm]
                s = np.array([s0, s1, s2])
                nsym[sel] = s * n0[sel][:, p]
                ethsym[sel] = s * eth0[sel][:, p]
                ephsym[sel] = s * eph0[sel][:, p]
        else:
            nsym, ethsym, ephsym = n0, eth0, eph0
        nvec = np.empty_like(nsym); eth = np.empty_like(ethsym)
        eph = np.empty_like(ephsym)
        for shell in range(nradial):
            sel = ir == shell
            rot = shell_rotation(seed64, shell)
            nvec[sel] = nsym[sel] @ rot.T
            eth[sel] = ethsym[sel] @ rot.T
            eph[sel] = ephsym[sel] @ rot.T
        # quartet directions +e_theta, -e_theta, +e_phi, -e_phi
        basis = np.where((idir < 2)[:, None], eth, eph)
        qsign = np.where(idir % 2 == 0, 1.0, -1.0)
        tvec = qsign[:, None] * basis

    pos = center + riso[:, None] * nvec
    vel = umag[:, None] * tvec
    rel = pos - center
    lvec = np.cross(rel, vel)
    return {
        "tag": tag, "shell_index": ir, "site_index": ia, "quartet_index": idir,
        "u": u, "rs": rs, "riso": riso, "pos": pos, "vel": vel,
        "n": nvec, "t": tvec, "umag": umag, "alpha": alpha, "W": w, "v": v,
        "conformal_a": conf_a, "lvec": lvec, "particle_mass": particle_mass,
        "rest_mass_over_m": profile.rest_mass_over_m, "profile": profile,
        "geometry": geo, "sampler": sampler, "seed": int(seed),
        "nradial": nradial, "nangular": nangular, "npart": npart,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sampler", default="shell_fibonacci", choices=SAMPLERS)
    ap.add_argument("--seed", type=int, default=1985)
    ap.add_argument("--radius-over-mass", type=float, default=6.1)
    ap.add_argument("--nradial", type=int, default=128)
    ap.add_argument("--nangular", type=int, default=1032)
    args = ap.parse_args()
    r = realize(args.sampler, args.seed, radius_over_mass=args.radius_over_mass,
                nradial=args.nradial, nangular=args.nangular)
    mp = r["particle_mass"]
    print(json.dumps({
        "sampler": args.sampler, "seed": args.seed, "npart": r["npart"],
        "M0_over_M": r["rest_mass_over_m"], "particle_mass": mp,
        "P_total": (mp * r["vel"].sum(axis=0)).tolist(),
        "J_total": (mp * r["lvec"].sum(axis=0)).tolist(),
        "L_scalar": float(mp * np.linalg.norm(r["lvec"], axis=1).sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
