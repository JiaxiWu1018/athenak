#!/usr/bin/env python3
"""Identify the surviving multipole of the octahedral quiet start exactly.

`cluster_octahedral_quiet_start=true` expands each Fibonacci seed direction
through the 24 proper rotations of a cube.  The resulting per-shell point set is
therefore invariant under the octahedral group O, and its angular density can
only contain those spherical harmonics that possess an O-invariant combination.

There is no such combination for l = 1, 2, 3, 5, 7.  The first one beyond the
monopole occurs at l = 4 and is unique:

    K_4  proportional to  Y_40 + sqrt(10/14) * Y_44        (real basis)

(the factor is sqrt(5/14) in the complex basis; the real Y_44 already carries
the extra sqrt(2)).  The next are at l = 6 and l = 8.

This script verifies that identification against the point set the pgen
actually builds, and checks the crucial orientation fact: `OctahedralRotate`
permutes and sign-flips the Cartesian components, so the cube -- and hence the
imprinted l=4 pattern -- is aligned with the Cartesian mesh axes.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from initial_realization_diagnostics import real_sph_harm_matrix
import cluster_sampler_reference as ref


def shell_directions(nangular, octahedral=True, rotation=None):
    golden = 0.5 * (math.sqrt(5.0) - 1.0)
    nfib = nangular // 24 if octahedral else nangular
    pts = []
    for ia in range(nangular):
        ifib = ia // 24 if octahedral else ia
        igrp = ia % 24 if octahedral else 0
        cth = 1.0 - 2.0 * (ifib + 0.5) / nfib
        sth = math.sqrt(max(1.0 - cth * cth, 0.0))
        ph = 2.0 * math.pi * math.fmod(golden * ifib, 1.0)
        n0 = np.array([sth * math.cos(ph), sth * math.sin(ph), cth])
        n = ref.octahedral_rotate(igrp, n0) if octahedral else n0
        pts.append(n)
    n = np.array(pts)
    if rotation is not None:
        n = n @ rotation.T
    return n


def spectrum(n, lmax=8):
    th = np.arccos(np.clip(n[:, 2], -1.0, 1.0))
    ph = np.arctan2(n[:, 1], n[:, 0])
    names, Y = real_sph_harm_matrix(th, ph, lmax)
    a = Y.mean(axis=1)
    a = a / a[0]
    power = {}
    for l in range(1, lmax + 1):
        idx = [i for i, (ll, _) in enumerate(names) if ll == l]
        power[l] = float(np.sqrt(np.sum(a[idx] ** 2)))
    return names, a, power


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nangular", type=int, default=1032)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    n = shell_directions(args.nangular, octahedral=True)
    names, a, power = spectrum(n)
    idx4 = [i for i, (l, _) in enumerate(names) if l == 4]
    ms = [names[i][1] for i in idx4]
    v = a[idx4]

    k = math.sqrt(10.0 / 14.0)
    K = np.zeros(9)
    K[ms.index(0)] = 1.0
    K[ms.index(4)] = k
    K = K / np.linalg.norm(K)
    overlap = abs(float(np.dot(v / np.linalg.norm(v), K)))
    ratio = float(v[ms.index(4)] / v[ms.index(0)])

    # Control: the same Fibonacci seeds without the octahedral expansion.
    n_plain = shell_directions(args.nangular, octahedral=False)
    _, _, power_plain = spectrum(n_plain)

    # Orientation: OctahedralRotate acts by permuting/flipping Cartesian
    # components, so the invariant axes are exactly the mesh axes e_x, e_y, e_z.
    axis_test = []
    for g in range(24):
        b = ref.octahedral_rotate(g, np.array([1.0, 0.0, 0.0]))
        axis_test.append(b.tolist())
    maps_axes_to_axes = all(
        sorted(abs(np.array(b)).tolist()) == [0.0, 0.0, 1.0] for b in axis_test)

    result = {
        "nangular": args.nangular,
        "nfibonacci_seeds": args.nangular // 24,
        "per_shell_multipole_power_octahedral": power,
        "per_shell_multipole_power_plain_fibonacci": power_plain,
        "l4_m_order": ms,
        "l4_coefficients": v.tolist(),
        "l4_a44_over_a40": ratio,
        "cubic_invariant_ratio_real_basis_sqrt_10_over_14": k,
        "relative_deviation_from_cubic_harmonic": abs(ratio - k) / k,
        "overlap_with_cubic_harmonic_K4": overlap,
        "octahedral_group_maps_cartesian_axes_to_cartesian_axes":
            bool(maps_axes_to_axes),
        "conclusion": (
            "The octahedral quiet start imprints, on every radial shell, exactly "
            "the unique cubic-invariant l=4 spherical harmonic "
            "K4 ~ Y40 + sqrt(10/14) Y44, oriented along the Cartesian mesh axes. "
            "l=1,2,3,5,7 vanish identically because the octahedral group admits "
            "no invariant at those l; l=4 is the lowest surviving nonspherical "
            "mode, followed by l=6 and l=8."),
    }

    print(f"octahedral-Fibonacci shell, N_angular={args.nangular} "
          f"({args.nangular//24} Fibonacci seeds x 24 cube rotations)\n")
    print(f"{'l':>3s} {'octahedral':>14s} {'plain Fibonacci':>18s}")
    for l in range(1, 9):
        print(f"{l:3d} {power[l]:14.4e} {power_plain[l]:18.4e}")
    print(f"\nl=4 coefficients (m = {ms}):")
    for m, x in zip(ms, v):
        frac = 100.0 * x * x / float(np.sum(v ** 2))
        print(f"   a_4,{m:+d} = {x:+.6e}   ({frac:5.1f}% of l=4 power)")
    print(f"\n a_4,+4 / a_4,0                  = {ratio:.6f}")
    print(f" sqrt(10/14) (real-basis cubic)  = {k:.6f}")
    print(f" relative deviation              = {abs(ratio-k)/k:.3e}")
    print(f" overlap with K4                 = {overlap:.8f}")
    print(f"\n octahedral group maps Cartesian axes to Cartesian axes: "
          f"{maps_axes_to_axes}")
    print(f"\n{result['conclusion']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
