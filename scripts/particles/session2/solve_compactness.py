#!/usr/bin/env python3
"""Session 2: solve for the Plummer scale b that realises a target effective
compactness, and verify the resulting model.

Compactness definition (as specified for this scan):

    (R/M)_eff = 1 / max_r [ m(r) / r ]

with m(r) the areal-radius mass function of the truncated relativistic Plummer
Einstein cluster of analysis/plummer_1d.py, r_t / b = 20 held fixed, hard cutoff,
vacuum outside, and the ADM mass renormalised to M = 1 exactly.

For the truncated profile, m(r)/r = M_P r^2 (r^2+b^2)^{-3/2} inside r_t and
M/r outside, so the interior maximum sits at r = b sqrt(2) and

    max(m/r) = 2 M_P / (3 sqrt(3) b)   ->   (R/M)_eff = 3 sqrt(3) b f_t / (2 M),

because M_P = M/f_t and f_t = (r_t/b)^3 / (1 + (r_t/b)^2)^{3/2} depends only on
the FIXED ratio r_t/b.  The relation is therefore exactly linear in b, but this
script does not rely on that: it root-finds on the numerically measured maximum
and then cross-checks against the closed form.
"""
import argparse
import json
import sys

import numpy as np
from scipy.optimize import brentq, minimize_scalar

sys.path.insert(0, str(__file__).rsplit('/', 1)[0])
from plummer_1d import PlummerModel

RT_OVER_B = 20.0


def max_m_over_r(mod, n=400001):
    """Numerically measured max of m(r)/r over the whole radial range, plus its
    location and the closed-form interior value for cross-check."""
    # interior scan, dense and log-spaced so the b*sqrt(2) peak is well resolved
    r = np.geomspace(1.0e-6 * mod.b, mod.rt, n)
    q = mod.m(r) / r
    i = int(np.argmax(q))
    # polish with a bounded 1D maximisation around the discrete peak
    lo, hi = r[max(i - 1, 0)], r[min(i + 1, n - 1)]
    res = minimize_scalar(lambda x: -float(mod.m(np.array([x]))[0]) / x,
                          bounds=(lo, hi), method='bounded',
                          options={'xatol': 1.0e-14})
    r_star = float(res.x)
    q_star = float(mod.m(np.array([r_star]))[0]) / r_star
    # exterior branch: m/r = M/r, monotone decreasing, max at r_t
    q_ext = mod.M / mod.rt
    closed = 2.0 * mod.MP / (3.0 * np.sqrt(3.0) * mod.b)
    return dict(max_m_over_r=max(q_star, q_ext),
                r_at_max=r_star if q_star >= q_ext else mod.rt,
                interior_max=q_star, interior_r=r_star,
                exterior_max_at_rt=q_ext, closed_form_interior=closed,
                closed_form_rel_err=abs(q_star - closed) / closed)


def compactness(b, M=1.0, **kw):
    mod = PlummerModel(M=M, b=b, rt=RT_OVER_B * b, **kw)
    return 1.0 / max_m_over_r(mod)['max_m_over_r'], mod


def solve_b(target, M=1.0, **kw):
    """Root-find b such that (R/M)_eff = target."""
    f = lambda b: compactness(b, M=M, **kw)[0] - target
    # bracket from the closed form: b0 = target * 2 / (3 sqrt(3) f_t)
    ft = RT_OVER_B**3 / (1.0 + RT_OVER_B**2)**1.5
    b0 = target * 2.0 / (3.0 * np.sqrt(3.0) * ft) * M
    lo, hi = 0.5 * b0, 2.0 * b0
    b = brentq(f, lo, hi, xtol=1.0e-14, rtol=8.9e-16, maxiter=200)
    return b


def characterise(b, M=1.0, npanel=6000, ngl=16, label=''):
    rt = RT_OVER_B * b
    mod = PlummerModel(M=M, b=b, rt=rt, npanel=npanel, ngl=ngl)
    mm = max_m_over_r(mod)
    P_half, r_half, alpha_half, vc_half = mod.P_half()
    R_half = float(mod.R_of_r(np.array([r_half]))[0])
    vcmax, r_vcmax = mod.vc_max()
    alpha0 = float(np.exp(mod.Phi_exact(np.array([1.0e-12 * b]))[0]))
    psi0 = float(np.exp(-0.5 * mod.j_exact(np.array([1.0e-12 * b]))[0]))
    # minimum lapse over the whole grid (monotone increasing outward, so alpha(0))
    r_dense = np.geomspace(1.0e-8 * b, 1.5 * rt, 200001)
    alpha_dense = mod.alpha(r_dense)
    i_amin = int(np.argmin(alpha_dense))

    # --- required pre-production checks ---------------------------------------
    # occupied range: r in (0, rt).  Use the sampler's actual reach later; here
    # the continuum support.
    r_occ = np.geomspace(1.0e-8 * b, rt * (1.0 - 1.0e-12), 200001)
    x = mod.m(r_occ) / r_occ
    # (1) circular geodesic exists  <=>  r > 3m  <=>  m/r < 1/3
    chk_geodesic = float(np.max(x))
    # (2) individual circular orbit radially stable: r^2 m' + r m - 6 m^2 > 0
    rs = mod.radial_stability(r_occ)
    # dimensionless margin: divide by the positive first term scale r^2 m'
    scale = r_occ**2 * mod.dmdr(r_occ) + r_occ * mod.m(r_occ)
    chk_stab_min = float(np.min(rs / np.where(scale > 0, scale, 1.0)))
    chk_stab_min_abs = float(np.min(rs))
    # (3) no horizon / pathological geometry: 2m/r < 1 everywhere, alpha > 0
    chk_2m_over_r = float(np.max(2.0 * mod.m(r_occ) / r_occ))
    chk_alpha_min = float(np.min(alpha_dense))
    # (4) continuum constraint identities
    rc = np.geomspace(1.0e-4 * b, rt * (1.0 - 1.0e-9), 4001)
    resH, resL, scH, scL = mod.constraint_residuals(rc)
    relH = float(np.max(np.abs(resH) / np.maximum(scH, 1.0e-300)))
    relL = float(np.max(np.abs(resL) / np.maximum(scL, 1.0e-300)))

    out = dict(
        label=label,
        M=M, b=b, rt=rt, rt_over_b=RT_OVER_B,
        f_t=mod.ft, M_P=mod.MP, M_0=mod.M0,
        r_half=r_half, R_half=R_half, r_half_over_b=r_half / b,
        P_half=P_half, alpha_at_r_half=alpha_half, vc_at_r_half=vc_half,
        R_t=mod.Rt,
        max_m_over_r=mm['max_m_over_r'], r_at_max_m_over_r=mm['r_at_max'],
        closed_form_interior_max=mm['closed_form_interior'],
        closed_form_rel_err=mm['closed_form_rel_err'],
        RM_eff=1.0 / mm['max_m_over_r'],
        vc_max=vcmax, r_at_vc_max=r_vcmax, r_at_vc_max_over_b=r_vcmax / b,
        alpha_0=alpha0, psi_0=psi0,
        alpha_min=chk_alpha_min, r_at_alpha_min=float(r_dense[i_amin]),
        check_max_m_over_r_lt_third=chk_geodesic,
        check_radial_stability_min_dimensionless=chk_stab_min,
        check_radial_stability_min_abs=chk_stab_min_abs,
        check_max_2m_over_r=chk_2m_over_r,
        check_constraint_H_rel=relH, check_constraint_L_rel=relL,
        P_half_x3=3.0 * P_half, P_half_x5=5.0 * P_half,
    )
    return out, mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', type=float, nargs='+', default=[10.0, 6.5])
    ap.add_argument('--labels', nargs='+', default=None)
    ap.add_argument('--npanel', type=int, default=40000)
    ap.add_argument('--ngl', type=int, default=20)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    labels = a.labels or [('R%g' % t).replace('.', 'p') for t in a.targets]

    # Session-1 regression: b = 20, rt = 400 must reproduce the published numbers
    s1, _ = characterise(20.0, npanel=a.npanel, ngl=a.ngl, label='session1_b20')
    print('=== Session-1 regression (b = 20 M, r_t = 400 M) ===')
    for k in ('f_t', 'M_P', 'M_0', 'r_half', 'R_half', 'P_half', 'alpha_at_r_half',
              'vc_at_r_half', 'R_t', 'alpha_0', 'psi_0', 'max_m_over_r', 'vc_max',
              'r_at_vc_max', 'RM_eff'):
        print('  %-26s %.12g' % (k, s1[k]))
    print()

    results = {'session1_regression': s1, 'cases': {}, 'convergence': {}}
    for t, lab in zip(a.targets, labels):
        b = solve_b(t, npanel=a.npanel, ngl=a.ngl)
        rec, mod = characterise(b, npanel=a.npanel, ngl=a.ngl, label=lab)
        rec['target_RM_eff'] = t
        rec['RM_eff_abs_err'] = abs(rec['RM_eff'] - t)
        results['cases'][lab] = rec
        print('=== %s : target (R/M)_eff = %g ===' % (lab, t))
        for k in ('b', 'rt', 'f_t', 'M_P', 'M_0', 'r_half', 'R_half', 'R_t',
                  'P_half', 'P_half_x3', 'P_half_x5', 'alpha_at_r_half',
                  'vc_at_r_half', 'max_m_over_r', 'r_at_max_m_over_r', 'RM_eff',
                  'RM_eff_abs_err', 'closed_form_rel_err', 'vc_max', 'r_at_vc_max',
                  'alpha_0', 'psi_0', 'alpha_min',
                  'check_max_m_over_r_lt_third',
                  'check_radial_stability_min_dimensionless',
                  'check_radial_stability_min_abs', 'check_max_2m_over_r',
                  'check_constraint_H_rel', 'check_constraint_L_rel'):
            print('  %-42s %.12g' % (k, rec[k]))
        # verdicts
        print('  VERDICT circular geodesic exists everywhere : %s (max m/r = %.6g < 1/3)'
              % ('PASS' if rec['check_max_m_over_r_lt_third'] < 1.0 / 3.0 else 'FAIL',
                 rec['check_max_m_over_r_lt_third']))
        print('  VERDICT individual circular orbit stable    : %s (min margin = %.6g > 0)'
              % ('PASS' if rec['check_radial_stability_min_abs'] > 0.0 else 'FAIL',
                 rec['check_radial_stability_min_dimensionless']))
        print('  VERDICT no horizon / regular geometry       : %s (max 2m/r = %.6g < 1, min alpha = %.6g > 0)'
              % ('PASS' if (rec['check_max_2m_over_r'] < 1.0 and rec['alpha_min'] > 0.0) else 'FAIL',
                 rec['check_max_2m_over_r'], rec['alpha_min']))
        print('  VERDICT continuum constraints               : %s (rel H = %.3g, rel L = %.3g)'
              % ('PASS' if max(rec['check_constraint_H_rel'], rec['check_constraint_L_rel']) < 1.0e-8 else 'FAIL',
                 rec['check_constraint_H_rel'], rec['check_constraint_L_rel']))
        print()

        # quadrature convergence: refine the panel count and compare
        conv = {}
        for npn in (a.npanel, 2 * a.npanel, 4 * a.npanel):
            r2, _ = characterise(b, npanel=npn, ngl=a.ngl, label=lab)
            conv[str(npn)] = {k: r2[k] for k in
                              ('P_half', 'alpha_0', 'psi_0', 'M_0', 'alpha_at_r_half',
                               'R_half', 'R_t', 'check_constraint_H_rel',
                               'check_constraint_L_rel')}
        results['convergence'][lab] = conv
        base = conv[str(a.npanel)]
        fine = conv[str(4 * a.npanel)]
        print('  quadrature convergence npanel %d -> %d:' % (a.npanel, 4 * a.npanel))
        for k in base:
            if 'check_' in k:
                print('    %-22s %.6g -> %.6g' % (k, base[k], fine[k]))
            else:
                d = abs(fine[k] - base[k]) / max(abs(fine[k]), 1e-300)
                print('    %-22s rel change %.3g' % (k, d))
        print()

    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(results, fh, indent=2, sort_keys=True)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
