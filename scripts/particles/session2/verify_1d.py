#!/usr/bin/env python3
"""Verify the 1D Plummer construction against the campaign reference values."""
import numpy as np
from plummer_1d import PlummerModel, PlummerInfinite

def report(name, got, ref, tol=None):
    if ref is None:
        print(f"  {name:24s} = {got!r}")
        return
    d = abs(got - ref)
    rel = d/abs(ref) if ref != 0 else d
    flag = "OK " if (tol is None or rel < tol) else "*** MISMATCH ***"
    print(f"  {name:24s} = {got:.12f}   ref {ref:.12f}   rel {rel:.3e}  {flag}")

print("="*78)
print("TRUNCATED MODEL:  M=1, b=20, r_t=400 (=20b)")
print("="*78)
for npan, ngl in [(2000,12),(6000,16),(20000,20)]:
    P = PlummerModel(npanel=npan, ngl=ngl)
    Ph, rh, ah, vh = P.P_half()
    Rh = float(P.R_of_r(np.array([rh]))[0])
    print(f"\n-- npanel={npan} ngl={ngl}")
    report("f_t", P.ft, P.rt**3/(P.rt**2 + P.b**2)**1.5, 1e-15)
    report("M_P/M", P.MP, 1.003752342774352, 1e-13)
    report("M_0/M", P.M0/P.M, 1.00738, 1e-4)
    report("r_1/2 [M]", rh, 26.00761423, 1e-8)
    report("R_1/2 [M]", Rh, 25.20934281, 1e-8)
    report("R_t [M]", P.Rt, 398.99937343, 1e-10)
    report("alpha(r_1/2)", ah, 0.96911272, 1e-7)
    report("v_c(r_1/2)", vh, 0.14139982, 1e-7)
    report("P_1/2 [M]", Ph, 1192.496781, 1e-8)
    report("alpha(0)", float(P.alpha_tab[0]), None)
    report("psi(0)", float(P.psi_tab[0]), None)
    mm = P.m(P.r_tab[1:]); rr = P.r_tab[1:]
    report("max m/r", float(np.max(mm/rr)), None)
    report("max v_c", float(np.sqrt(np.max(P.vc2(rr)))), None)
    report("min(r^2 m'+r m-6m^2)", float(np.min(P.radial_stability(rr))), None)
    # mass normalisation by independent quadrature
    from scipy.integrate import quad
    Mnum = quad(lambda s: 4*np.pi*s*s*float(P.eps(np.array([s]))[0]), 0, 400.0,
                limit=400, epsabs=1e-14, epsrel=1e-13)[0]
    report("4pi int eps r^2 dr", Mnum, 1.0, 1e-10)
    M0num = quad(lambda s: float(4*np.pi*s*s*P.eps(np.array([s]))[0]
                                 *P.B(np.array([s]))[0]/P.W(np.array([s]))[0]),
                 0, 400.0, limit=400, epsabs=1e-14, epsrel=1e-13)[0]
    report("M_0 (scipy quad)", M0num, P.M0/1.0, 1e-10)

print()
print("="*78)
print("INFINITE MODEL CHECK (PDF p.4):  M_P/b = 0.05  -> M_P=1, b=20")
print("="*78)
for rmax, npan in [(1e6, 20000), (1e7, 40000), (1e8, 60000)]:
    Q = PlummerInfinite(MP=1.0, b=20.0, rmax=rmax, npanel=npan, ngl=20)
    print(f"\n-- rmax={rmax:.0e} npanel={npan}")
    report("alpha(0)", float(Q.alpha_tab[0]), 0.950003710, 1e-8)
    report("psi(0)", float(Q.psi_tab[0]), 1.025808423, 1e-8)
    report("M_0/M_P", Q.M0/Q.MP, 1.007325025, 1e-7)
    rr = Q.r_tab[1:]
    report("v_c,max", float(np.sqrt(np.max(Q.vc2(rr)))), 0.141475801, 1e-8)

print()
print("="*78)
print("CONTINUUM CONSTRAINT RESIDUALS (truncated model)")
print("="*78)
P = PlummerModel(npanel=20000, ngl=20)
rs = np.concatenate([np.linspace(1e-3, 1.0, 40),
                     np.geomspace(1.0, 399.9, 400)])
rH, rL, sH, sL = P.constraint_residuals(rs)
print(f"  max |Lap psi + 2pi psi^5 eps| / scale        = {np.max(np.abs(rH)/sH):.3e}")
print(f"  max |Lap(a psi) - 2pi a psi^5 (eps+2S)|/scale = {np.max(np.abs(rL)/sL):.3e}")
for rq in [0.1, 1.0, 5.0, 20.0, 26.00761423, 100.0, 300.0, 399.0]:
    a, bq, c, d = P.constraint_residuals(np.array([rq]))
    print(f"    r={rq:9.4f}  H rel {abs(a[0])/c[0]:.2e}   L rel {abs(bq[0])/d[0]:.2e}")

print()
print("="*78)
print("SANITY: exterior matching, monotonicity, occupancy")
print("="*78)
print(f"  alpha(r_t^-) interior = {P.alpha_tab[-1]:.12f}")
print(f"  sqrt(1-2M/r_t)        = {np.sqrt(1-2/400.0):.12f}")
print(f"  R(r_t) interior       = {P.R_tab[-1]:.12f}")
print(f"  R_t analytic          = {P.Rt:.12f}")
print(f"  R(r) monotone         = {bool(np.all(np.diff(P.R_tab) > 0))}")
print(f"  F0 monotone           = {bool(np.all(np.diff(P.F0_tab) >= 0))}")
rr = P.r_tab[1:]
print(f"  max m/r  = {np.max(P.m(rr)/rr):.9f}  (<1/3 required: {np.max(P.m(rr)/rr) < 1/3})")
print(f"  argmax m/r at r = {rr[np.argmax(P.m(rr)/rr)]:.4f} M")
print(f"  min radial-stability fn = {np.min(P.radial_stability(rr)):.6e} (>0 required)")
