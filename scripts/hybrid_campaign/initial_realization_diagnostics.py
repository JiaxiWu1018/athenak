#!/usr/bin/env python3
"""Pre-evolution diagnostics for homogeneous-cluster particle realizations.

Computes, for each (sampler, seed) pair, the quantities requested by the
sampler-causality investigation:

  * empirical radial rest-mass CDF error against the continuum F(u)
  * radial density and deposited-energy profiles
  * particle-density multipoles through l = LMAX (default 8), volume-weighted
    and shell-resolved, with the l=4 amplitude and principal orientation
  * total rest mass, total vector momentum, total vector angular momentum
  * distribution of individual |L| (median / p95 / max / spread)
  * number of unique spatial positions and radial shell-peak structure

The realization comes from ``cluster_sampler_reference.py``, which mirrors the
pgen exactly and is separately validated against the executable's own t=0
particle dump.  Deposited-source and constraint diagnostics come from the
solver runs, not from this script.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import cluster_sampler_reference as ref

LMAX_DEFAULT = 8


def real_sph_harm_matrix(theta, phi, lmax):
    """Real orthonormal spherical harmonics Y_lm(theta,phi) for l<=lmax.

    Returns (names, values) where values has shape (n_modes, npoints).  Uses the
    standard real basis built from associated Legendre functions via a stable
    upward recursion, avoiding a scipy dependency.
    """
    x = np.cos(theta)
    sinth = np.sqrt(np.maximum(1.0 - x * x, 0.0))
    # P[l][m] for m>=0
    P = {}
    P[(0, 0)] = np.ones_like(x)
    for m in range(1, lmax + 1):
        P[(m, m)] = -(2 * m - 1) * sinth * P[(m - 1, m - 1)]
    for m in range(0, lmax):
        P[(m + 1, m)] = (2 * m + 1) * x * P[(m, m)]
    for m in range(0, lmax + 1):
        for l in range(m + 2, lmax + 1):
            P[(l, m)] = (((2 * l - 1) * x * P[(l - 1, m)]
                          - (l + m - 1) * P[(l - 2, m)]) / (l - m))
    names, vals = [], []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            norm = math.sqrt((2 * l + 1) / (4 * math.pi)
                             * math.factorial(l - am) / math.factorial(l + am))
            base = norm * P[(l, am)]
            if m == 0:
                vals.append(base)
            elif m > 0:
                vals.append(math.sqrt(2.0) * base * np.cos(m * phi))
            else:
                vals.append(math.sqrt(2.0) * base * np.sin(am * phi))
            names.append((l, m))
    return names, np.asarray(vals)


def angular_multipoles(unit_vectors, lmax, weights=None):
    """Return {l: power} and per-(l,m) coefficients of the empirical angular
    density of a point set on S^2.

    a_lm = (1/N) sum_p w_p Y_lm(n_p); the l=0 coefficient is the mean density.
    Reported power is sqrt(sum_m a_lm^2) normalized by a_00, i.e. the relative
    multipole amplitude of the angular distribution.
    """
    n = unit_vectors
    r = np.linalg.norm(n, axis=1)
    good = r > 0
    n = n[good] / r[good, None]
    w = np.ones(len(n)) if weights is None else np.asarray(weights)[good]
    theta = np.arccos(np.clip(n[:, 2], -1.0, 1.0))
    phi = np.arctan2(n[:, 1], n[:, 0])
    names, Y = real_sph_harm_matrix(theta, phi, lmax)
    coeffs = (Y * w).sum(axis=1) / w.sum()
    a00 = coeffs[0]
    power, per_lm = {}, {}
    for l in range(lmax + 1):
        sel = [i for i, (ll, _) in enumerate(names) if ll == l]
        power[l] = float(np.sqrt(np.sum(coeffs[sel] ** 2)) / abs(a00))
        for i in sel:
            per_lm[f"a_{names[i][0]}_{names[i][1]}"] = float(coeffs[i] / a00)
    return power, per_lm, dict(zip(names, coeffs))


def shell_coherence(unit_vectors, group_index, ngroup, lmax):
    """Shell-to-shell coherence of each multipole.

    The 2026-07-28 diagnostics found a *coherent* growing l=4 torque, not merely
    a large one.  Whole-cluster multipole power cannot distinguish a mode that
    every radial shell shares from independent per-shell noise of the same
    amplitude, so this measures both:

      power_rms[l]  = rms over shells of the per-shell multipole amplitude
      coherence[l]  = | mean over shells of the unit a_lm vector |

    For S independent shells, coherence ~ 1/sqrt(S) (0.088 for S=128); a mode
    common to every shell gives coherence -> 1.  The reported value is
    normalized so that pure noise sits near 1 and perfect coherence near
    sqrt(S): coherence_ratio = coherence * sqrt(S).
    """
    n = unit_vectors
    theta = np.arccos(np.clip(n[:, 2], -1.0, 1.0))
    phi = np.arctan2(n[:, 1], n[:, 0])
    names, Y = real_sph_harm_matrix(theta, phi, lmax)
    power_rms, coherence_ratio = {}, {}
    idx_by_l = {l: [i for i, (ll, _) in enumerate(names) if ll == l]
                for l in range(lmax + 1)}
    # a_lm per shell
    shells = [np.flatnonzero(group_index == g) for g in range(ngroup)]
    shells = [s for s in shells if len(s) > 0]
    A = np.array([Y[:, s].mean(axis=1) for s in shells])   # (nshell, nmode)
    a00 = A[:, 0]
    for l in range(1, lmax + 1):
        sel = idx_by_l[l]
        v = A[:, sel] / a00[:, None]
        amp = np.linalg.norm(v, axis=1)
        power_rms[l] = float(np.sqrt(np.mean(amp ** 2)))
        # Modes cancelled exactly by the octahedral expansion survive only at
        # roundoff; their "direction" is numerical noise, so coherence is
        # undefined rather than meaningful for them.
        good = amp > 1.0e-12
        if good.sum() < 2:
            coherence_ratio[l] = float("nan")
            continue
        unit = v[good] / amp[good, None]
        resultant = float(np.linalg.norm(unit.mean(axis=0)))
        coherence_ratio[l] = resultant * math.sqrt(good.sum())
    return power_rms, coherence_ratio


def l4_orientation(coeff_map):
    """Principal axis of the l=4 density mode.

    Builds the rank-2 orientation proxy M_ij = sum_m a_4m <n_i n_j>_lm by
    numerically projecting the l=4 reconstruction onto a fine sphere and taking
    the eigenvector of its second-moment tensor with the largest |eigenvalue|.
    This yields a well-defined body axis whose rotation can be tracked between
    realizations.
    """
    nsamp = 20000
    idx = np.arange(nsamp) + 0.5
    cth = 1.0 - 2.0 * idx / nsamp
    sth = np.sqrt(np.maximum(1.0 - cth * cth, 0.0))
    golden = 0.5 * (math.sqrt(5.0) - 1.0)
    ph = 2.0 * math.pi * np.mod(golden * idx, 1.0)
    n = np.stack([sth * np.cos(ph), sth * np.sin(ph), cth], axis=1)
    theta = np.arccos(np.clip(n[:, 2], -1.0, 1.0))
    phi = np.arctan2(n[:, 1], n[:, 0])
    names, Y = real_sph_harm_matrix(theta, phi, 4)
    f = np.zeros(nsamp)
    for i, (l, m) in enumerate(names):
        if l == 4:
            f += coeff_map[(l, m)] * Y[i]
    mom = np.einsum("p,pi,pj->ij", f, n, n) / nsamp
    evals, evecs = np.linalg.eigh(mom)
    k = int(np.argmax(np.abs(evals)))
    axis = evecs[:, k]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    return axis, float(evals[k]), float(np.sqrt(np.sum(evals ** 2)))


def diagnose(sampler, seed, q, nradial, nangular, lmax, mass=1.0, xi=1.0):
    r = ref.realize(sampler, seed, mass=mass, radius_over_mass=q, xi=xi,
                    nradial=nradial, nangular=nangular, octahedral=True)
    mp = r["particle_mass"]
    npart = r["npart"]
    pos, vel, lvec = r["pos"], r["vel"], r["lvec"]
    prof = r["profile"]
    geo = r["geometry"]

    # --- radial CDF error against the continuum rest-mass CDF -----------------
    u_sorted = np.sort(r["u"])
    f_emp = (np.arange(npart) + 0.5) / npart
    f_cont = np.interp(u_sorted, prof.u, prof.cdf)
    cdf_err = f_emp - f_cont
    ks = float(np.max(np.abs(cdf_err)))
    l1 = float(np.mean(np.abs(cdf_err)))
    l2 = float(np.sqrt(np.mean(cdf_err ** 2)))

    # --- conserved sums (never subtracted, only measured) --------------------
    P_tot = mp * vel.sum(axis=0)
    J_tot = mp * lvec.sum(axis=0)
    labs = np.linalg.norm(lvec, axis=1)
    L_scalar = float(mp * labs.sum())
    E0 = float(mp * (r["alpha"] * r["W"]).sum())
    M0 = mp * npart

    # --- unique spatial positions and shell structure ------------------------
    key = np.round(pos, 12)
    uniq = len(np.unique(key.view([("", key.dtype)] * 3)))
    riso = r["riso"]
    uniq_radii = len(np.unique(np.round(riso, 12)))
    # Shell structure is the concentration of particles onto discrete radii.
    # The cleanest scale-free measure is the fraction of particles sharing an
    # exactly repeated radius, plus the radial-histogram contrast at a
    # resolution comparable to the finest grid spacing (dx = 0.0859375 M).
    _, counts_per_radius = np.unique(np.round(riso, 12), return_counts=True)
    repeated_fraction = float(counts_per_radius[counts_per_radius > 1].sum() / npart)
    nbin_fine = max(1, int(np.ceil(geo.r0 / 0.0859375)))
    hist, _ = np.histogram(riso, bins=nbin_fine, range=(0.0, geo.r0 * 1.0000001))
    expected = npart / nbin_fine
    # Contrast: how much more peaked the occupied bins are than a uniform fill.
    occupied = hist[hist > 0]
    peak_contrast = float(occupied.max() / expected) if len(occupied) else float("nan")
    peak_bins = int(np.sum(hist > 4.0 * expected))
    peak_fraction = float(hist[hist > 4.0 * expected].sum() / npart)

    # --- radial rest-mass and energy profiles --------------------------------
    nbin = 64
    ubin = np.linspace(0.0, 1.0, nbin + 1)
    ucen = 0.5 * (ubin[1:] + ubin[:-1])
    counts, _ = np.histogram(r["u"], bins=ubin)
    m0_shell = counts * mp
    e_shell = np.histogram(r["u"], bins=ubin,
                           weights=mp * r["alpha"] * r["W"])[0]
    # Continuum expectations for the same bins.
    f_edges = np.interp(ubin, prof.u, prof.cdf)
    m0_shell_exact = np.diff(f_edges) * M0
    # Proper-volume normal-frame energy density rho = 3M/(4 pi R^3) is uniform,
    # so the *rest*-mass profile carries the 1/W weighting.
    rs_c = geo.radius * ucen
    dV_proper = (4.0 * math.pi * rs_c ** 2 * np.diff(ubin) * geo.radius
                 / np.sqrt(1.0 - 2.0 * geo.compactness * ucen ** 2))
    rho0_num = np.divide(m0_shell, dV_proper, out=np.zeros(nbin),
                         where=dV_proper > 0)
    x_c = geo.compactness * ucen ** 2
    v_c = xi * np.sqrt(x_c / (1.0 - 2.0 * x_c))
    W_c = 1.0 / np.sqrt(1.0 - v_c ** 2)
    rho_c = 3.0 * mass / (4.0 * math.pi * geo.radius ** 3)
    rho0_exact = rho_c / W_c

    # --- angular multipoles: full cluster, per shell group, per radial band --
    power_all, per_lm_all, coeff_all = angular_multipoles(r["n"], lmax)
    axis4, eig4, l4_norm = l4_orientation(coeff_all)
    # The tag group is an exact radial shell for samplers A/B/C and an
    # equal-rest-mass radial band for D/E/F, so shell coherence is comparable.
    power_shell, coh_shell = shell_coherence(
        r["n"], r["shell_index"], nradial, lmax)

    band_rows = []
    nband = 8
    order = np.argsort(r["u"])
    for b in range(nband):
        sel = order[b * npart // nband:(b + 1) * npart // nband]
        p_b, _, c_b = angular_multipoles(r["n"][sel], min(lmax, 8))
        band_rows.append({
            "band": b,
            "u_lo": float(r["u"][sel].min()), "u_hi": float(r["u"][sel].max()),
            **{f"P{l}": p_b[l] for l in range(min(lmax, 8) + 1)},
        })

    # --- deposited-energy proxy: CIC onto the finest-level cell size ---------
    # This is the sampling-only part of the T_munu mismatch; the solver runs
    # provide the true deposited source and constraints.
    dx = 0.0859375
    ncell = 128
    half = 0.5 * ncell * dx
    inside = np.all(np.abs(pos) < half - dx, axis=1)
    gidx = np.floor((pos[inside] + half) / dx).astype(int)
    flat = (gidx[:, 0] * ncell + gidx[:, 1]) * ncell + gidx[:, 2]
    dep = np.bincount(flat, weights=mp * r["alpha"][inside] * 0 + mp * r["W"][inside],
                      minlength=ncell ** 3)
    occupied = int(np.sum(dep > 0))
    dep_nz = dep[dep > 0]
    dep_cv = float(dep_nz.std() / dep_nz.mean()) if len(dep_nz) else float("nan")

    return {
        "sampler": sampler, "seed": int(seed), "radius_over_mass": q,
        "nradial": nradial, "nangular": nangular, "npart": npart,
        "particle_mass": mp, "M0": float(M0),
        "M0_over_M": float(r["rest_mass_over_m"]),
        "E0": E0, "L_scalar": L_scalar,
        "cdf_ks_error": ks, "cdf_l1_error": l1, "cdf_l2_error": l2,
        "P_x": float(P_tot[0]), "P_y": float(P_tot[1]), "P_z": float(P_tot[2]),
        "P_norm": float(np.linalg.norm(P_tot)),
        "J_x": float(J_tot[0]), "J_y": float(J_tot[1]), "J_z": float(J_tot[2]),
        "J_norm": float(np.linalg.norm(J_tot)),
        "P_norm_over_scale": float(np.linalg.norm(P_tot) / (mp * np.abs(r["umag"]).sum())),
        "J_norm_over_Lscalar": float(np.linalg.norm(J_tot) / L_scalar),
        "L_median": float(np.median(labs)), "L_p95": float(np.percentile(labs, 95)),
        "L_max": float(labs.max()), "L_min": float(labs.min()),
        "L_mean": float(labs.mean()), "L_std": float(labs.std()),
        "unique_positions": uniq, "unique_positions_frac": uniq / npart,
        "unique_radii": uniq_radii, "unique_radii_frac": uniq_radii / npart,
        "repeated_radius_fraction": repeated_fraction,
        "shell_peak_bins": peak_bins, "shell_peak_fraction": peak_fraction,
        "shell_peak_contrast": peak_contrast,
        "cic_occupied_cells": occupied, "cic_density_cv": dep_cv,
        **{f"P{l}": power_all[l] for l in range(lmax + 1)},
        **{f"shellP{l}": power_shell[l] for l in range(1, lmax + 1)},
        **{f"coh{l}": coh_shell[l] for l in range(1, lmax + 1)},
        "l4_axis_x": float(axis4[0]), "l4_axis_y": float(axis4[1]),
        "l4_axis_z": float(axis4[2]), "l4_eigen": eig4, "l4_norm": l4_norm,
        "_per_lm": per_lm_all, "_bands": band_rows,
        "_profile": {
            "u": ucen.tolist(),
            "rho0_numeric": rho0_num.tolist(),
            "rho0_exact": rho0_exact.tolist(),
            "m0_shell": m0_shell.tolist(),
            "m0_shell_exact": m0_shell_exact.tolist(),
            "e_shell": e_shell.tolist(),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--samplers", nargs="+", default=list(ref.SAMPLERS))
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[1985, 20260801, 424242, 90210, 7])
    ap.add_argument("--radius-over-mass", nargs="+", type=float, default=[6.1])
    ap.add_argument("--nradial", type=int, default=128)
    ap.add_argument("--nangular", type=int, default=1032)
    ap.add_argument("--lmax", type=int, default=LMAX_DEFAULT)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows, detail = [], {}
    for q in args.radius_over_mass:
        for sampler in args.samplers:
            for seed in args.seeds:
                d = diagnose(sampler, seed, q, args.nradial, args.nangular,
                             args.lmax)
                key = f"q{q}_{sampler}_seed{seed}"
                detail[key] = {"per_lm": d.pop("_per_lm"),
                               "bands": d.pop("_bands"),
                               "profile": d.pop("_profile")}
                rows.append(d)
                print(f"{key}: P4={d['P4']:.5e} KS={d['cdf_ks_error']:.3e} "
                      f"|J|={d['J_norm']:.4e} uniq={d['unique_positions_frac']:.3f}",
                      flush=True)

    with (args.output / "initial_realization_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "initial_realization_detail.json").write_text(
        json.dumps(detail, indent=1))
    print(f"wrote {len(rows)} realizations to {args.output}")


if __name__ == "__main__":
    main()
