#!/usr/bin/env python3
"""Real orthonormal spherical harmonics Y_lm, l = 0..4, in Cartesian form.

Input are unit vectors (x, y, z); output arrays are Y_lm evaluated pointwise.
Normalisation: \\int Y_lm Y_l'm' dOmega = delta.  numpy only (no scipy).
"""
import numpy as np

S = np.sqrt
PI = np.pi


def ylm_all(x, y, z, lmax=4):
    """Return dict {l: (2l+1, n) array of Y_lm} for l = 0..lmax on unit vectors."""
    x2, y2, z2 = x * x, y * y, z * z
    out = {}
    out[0] = np.array([np.full_like(x, 0.5 * S(1.0 / PI))])
    if lmax >= 1:
        c = S(3.0 / (4 * PI))
        out[1] = np.array([c * y, c * z, c * x])
    if lmax >= 2:
        out[2] = np.array([
            0.5 * S(15.0 / PI) * x * y,
            0.5 * S(15.0 / PI) * y * z,
            0.25 * S(5.0 / PI) * (3 * z2 - 1.0),
            0.5 * S(15.0 / PI) * x * z,
            0.25 * S(15.0 / PI) * (x2 - y2),
        ])
    if lmax >= 3:
        out[3] = np.array([
            0.25 * S(35.0 / (2 * PI)) * y * (3 * x2 - y2),
            0.5 * S(105.0 / PI) * x * y * z,
            0.25 * S(21.0 / (2 * PI)) * y * (5 * z2 - 1.0),
            0.25 * S(7.0 / PI) * z * (5 * z2 - 3.0),
            0.25 * S(21.0 / (2 * PI)) * x * (5 * z2 - 1.0),
            0.25 * S(105.0 / PI) * z * (x2 - y2),
            0.25 * S(35.0 / (2 * PI)) * x * (x2 - 3 * y2),
        ])
    if lmax >= 4:
        out[4] = np.array([
            0.75 * S(35.0 / PI) * x * y * (x2 - y2),
            0.75 * S(35.0 / (2 * PI)) * y * z * (3 * x2 - y2),
            0.75 * S(5.0 / PI) * x * y * (7 * z2 - 1.0),
            0.75 * S(5.0 / (2 * PI)) * y * z * (7 * z2 - 3.0),
            (3.0 / 16.0) * S(1.0 / PI) * (35 * z2 * z2 - 30 * z2 + 3.0),
            0.75 * S(5.0 / (2 * PI)) * x * z * (7 * z2 - 3.0),
            (3.0 / 8.0) * S(5.0 / PI) * (x2 - y2) * (7 * z2 - 1.0),
            0.75 * S(35.0 / (2 * PI)) * x * z * (x2 - 3 * y2),
            (3.0 / 16.0) * S(35.0 / PI) * (x2 * (x2 - 3 * y2) - y2 * (3 * x2 - y2)),
        ])
    return {l: v for l, v in out.items() if l <= lmax}


def mode_amplitude(x, y, z, lmax=4, weights=None):
    """Normalised mass-multipole amplitudes A_l of a set of directions.

        a_lm = <Y_lm>,   A_l = sqrt( 4pi/(2l+1) * sum_m a_lm^2 )

    A_l = 1 for a delta function in angle; for N isotropic points E[A_l^2] = 1/N,
    so A_l^shot = N^{-1/2} is the finite-N null.  Returns (A, a) where A[l] is the
    amplitude and a[l] the (2l+1,) coefficient vector.
    """
    Y = ylm_all(x, y, z, lmax)
    if weights is None:
        n = x.size
        A, a = {}, {}
        for l, arr in Y.items():
            coef = arr.mean(axis=1)
            a[l] = coef
            A[l] = np.sqrt(4 * PI / (2 * l + 1) * np.sum(coef ** 2))
        A["N"] = n
        return A, a
    w = weights / weights.sum()
    A, a = {}, {}
    neff = 1.0 / np.sum(w ** 2)
    for l, arr in Y.items():
        coef = arr @ w
        a[l] = coef
        A[l] = np.sqrt(4 * PI / (2 * l + 1) * np.sum(coef ** 2))
    A["N"] = neff
    return A, a


def shot_floor(n):
    return 1.0 / np.sqrt(n)


def debias(A_l, n):
    """Shot-noise-subtracted amplitude: sqrt(max(0, A^2 - 1/N))."""
    return np.sqrt(max(0.0, A_l ** 2 - 1.0 / n))
