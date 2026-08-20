#!/usr/bin/env python3
"""Static spherical Einstein-Vlasov equilibrium of the GI-in-cluster construction.

Reference: einstein_vlasov_cluster_initial_data.pdf (2026-08-11), Sec. II.

Distribution function (Eq. 7):   f0(E) = A (Ecut - E)_+ ,
with E = e^{mu(r)} sqrt(1+|v|^2) and v the orthonormal specific momentum
(spatial four-velocity in the orthonormal frame), W = sqrt(1+|v|^2).

Metric (Eq. 5):  ds^2 = -e^{2mu} dt^2 + e^{2lam} dr^2 + r^2 dOmega^2,
e^{-2lam} = 1 - 2m/r.

Shooting variables (Eq. 11):  y = ln(Ecut) - mu ,  B = A*Ecut.
Closed-form moments (Eq. 12), vy = sqrt(e^{2y}-1), uy = asinh(vy):
  rho0 = 4 pi B [ (vy e^y (2 vy^2 + 1) - uy)/8 - e^{-y}(vy^3/3 + vy^5/5) ]
  p0   = (4 pi B / 3) [ (vy e^y (2 vy^2 - 3) + 3 uy)/8 - e^{-y} vy^5/5 ]
Rest-mass density (n0 = int f0 d^3v, no W factor; derived here, verified
against direct quadrature below):
  n0   = 4 pi B [ vy^3/3 - vy e^{2y}/4 + vy/8 + uy e^{-y}/8 ]

Field equations (Eq. 10):
  dm/dr = 4 pi r^2 rho0 ,   dy/dr = -(m + 4 pi r^3 p0) / (r (r - 2m)).

Scaling reduction (used here; exact property of Eqs. 10+12):
under r -> lam r, m -> lam m, B -> B/lam^2 the system maps to itself with
y(r) unchanged.  Hence with B=1 the compactness 2 M0/R0 is a function of
y_c alone; solve y_c from the target compactness, then rescale to R0.

Targets (Eq. 14): M0 = 0.61, R0 = 30, 2M0/R0 = 0.0406666667.
Reference solution (Eq. 15):
  y_c = 0.0550409773, Ecut = 0.9794556311, A = 0.0389329770, B = 0.0381331236.

Isotropic transform (Eq. 16-17): dR/R = e^{lam} dr/r, r = psi0^2 R,
psi0 = sqrt(r/R); at the boundary R(R0) = (R0 - M0 + sqrt(R0(R0-2M0)))/2.
Integrated here as h(r) = int_0^r (e^{lam}-1)/r' dr' (regular at 0), with
R(r) = C r e^{h(r)} and C fixed by the boundary value.

Outputs: profile table R, r(R), psi0(R), alpha0(R)=e^mu, rho0(R), n0(R),
p0(R), q0(R)=psi0^5 rho0, and the scalar constants; plus internal checks:
  - closed forms vs direct quadrature of Eq. 8,
  - flat-Laplacian identity  (1/R^2) d/dR (R^2 dpsi0/dR) = -2 pi q0,
  - psi0(Riso_surface) = 1 + M0/(2 Riso_surface) (Schwarzschild matching),
  - clump table checks: sum Ma = 0.39, sum Ma Xa = 0.

Deterministic; no RNG. Step-size convergence is reported.
"""

import argparse
import json
import math
import sys

import numpy as np


# ----------------------------------------------------------------------------
# closed-form moments in y (B = A*Ecut absorbed; these return rho0/(4 pi B) etc.)

def _vy_uy(y):
    vy = math.sqrt(max(math.expm1(2.0*y), 0.0))
    uy = math.asinh(vy)
    return vy, uy


def rho0_over_4piB(y):
    if y <= 0.0:
        return 0.0
    vy, uy = _vy_uy(y)
    ey = math.exp(y)
    return (vy*ey*(2.0*vy*vy + 1.0) - uy)/8.0 - math.exp(-y)*(vy**3/3.0 + vy**5/5.0)


def p0_over_4piB(y):
    if y <= 0.0:
        return 0.0
    vy, uy = _vy_uy(y)
    ey = math.exp(y)
    return ((vy*ey*(2.0*vy*vy - 3.0) + 3.0*uy)/8.0 - math.exp(-y)*vy**5/5.0)/3.0


def n0_over_4piB(y):
    if y <= 0.0:
        return 0.0
    vy, uy = _vy_uy(y)
    e2y = math.exp(2.0*y)
    return vy**3/3.0 - vy*e2y/4.0 + vy/8.0 + uy*math.exp(-y)/8.0


def moments_by_quadrature(y, nq=4000):
    """Direct quadrature of Eq. (8) in y-variables: e^mu = Ecut e^{-y}.
    Returns (rho0, p0, n0)/(4 pi B). Integrand: (e^{-y'} ... ) with
    f0/A = (Ecut - Ecut e^{-y} W) = Ecut (1 - e^{-y} W); dividing by Ecut
    to express in units of B = A*Ecut."""
    vy, _ = _vy_uy(y)
    if vy == 0.0:
        return 0.0, 0.0, 0.0
    v = np.linspace(0.0, vy, nq)
    W = np.sqrt(1.0 + v*v)
    base = 1.0 - np.exp(-y)*W
    rho = np.trapz(base*W*v*v, v)
    p = np.trapz(base*(v**4)/W, v)/3.0
    n = np.trapz(base*v*v, v)
    return rho, p, n


# ----------------------------------------------------------------------------
# ODE integration (B=1 scaled system)

def rhs(r, m, y, h):
    ro = 4.0*math.pi*rho0_over_4piB(y)      # rho0 with 4 pi B = 4 pi
    pr = 4.0*math.pi*p0_over_4piB(y)
    dm = 4.0*math.pi*r*r*ro
    denom = r*(r - 2.0*m)
    dy = -(m + 4.0*math.pi*r**3*pr)/denom
    elam = 1.0/math.sqrt(max(1.0 - 2.0*m/r, 1e-300))
    dh = (elam - 1.0)/r
    return dm, dy, dh


def integrate(yc, dr, rmax_factor=2000.0, store=False):
    """RK4 with fixed step from a regular center series; stop at first y=0
    crossing (linear interpolation).  B=1 units. Returns dict with surface
    values and, if store, the full profile arrays."""
    rho_c = 4.0*math.pi*rho0_over_4piB(yc)   # physical rho0(yc) in B=1 units
    p_c = 4.0*math.pi*p0_over_4piB(yc)
    # Regular center series through the first grid point r = dr:
    #   m ~ (4 pi/3) rho_c r^3,  y ~ yc - 2 pi (rho_c/3 + p_c) r^2,
    #   h ~ (2 pi/3) rho_c r^2   [from dh/dr ~ m/r^2].
    eps = dr
    r = eps
    m = (4.0*math.pi/3.0)*rho_c*eps**3
    y = yc - 2.0*math.pi*(rho_c/3.0 + p_c)*eps*eps
    h = (2.0*math.pi/3.0)*rho_c*eps*eps
    prof = {'r': [0.0, r], 'm': [0.0, m], 'y': [yc, y], 'h': [0.0, h]}

    nmax = int(rmax_factor/dr)
    surface = None
    for _ in range(nmax):
        k1 = rhs(r, m, y, h)
        k2 = rhs(r + 0.5*dr, m + 0.5*dr*k1[0], y + 0.5*dr*k1[1], h + 0.5*dr*k1[2])
        k3 = rhs(r + 0.5*dr, m + 0.5*dr*k2[0], y + 0.5*dr*k2[1], h + 0.5*dr*k2[2])
        k4 = rhs(r + dr, m + dr*k3[0], y + dr*k3[1], h + dr*k3[2])
        m_new = m + dr*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6.0
        y_new = y + dr*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6.0
        h_new = h + dr*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6.0
        r_new = r + dr
        if not (math.isfinite(m_new) and math.isfinite(y_new) and math.isfinite(h_new)):
            raise RuntimeError('ODE became non-finite at r=%g (yc=%g)' % (r, yc))
        if y_new <= 0.0:
            frac = y/(y - y_new)
            r_s = r + frac*dr
            m_s = m + frac*(m_new - m)
            h_s = h + frac*(h_new - h)
            if store:
                prof['r'].append(r_s)
                prof['m'].append(m_s)
                prof['y'].append(0.0)
                prof['h'].append(h_s)
            surface = (r_s, m_s, h_s)
            break
        r, m, y, h = r_new, m_new, y_new, h_new
        if store:
            prof['r'].append(r)
            prof['m'].append(m)
            prof['y'].append(y)
            prof['h'].append(h)
    if surface is None:
        raise RuntimeError('no surface found for yc=%g' % yc)
    out = {'R0': surface[0], 'M0': surface[1], 'hs': surface[2]}
    if store:
        out['prof'] = {k: np.array(v) for k, v in prof.items()}
    return out


def compactness(yc, dr):
    s = integrate(yc, dr)
    return 2.0*s['M0']/s['R0']


def solve_yc(target_c, dr, lo=0.005, hi=0.5, tol=1e-14):
    """Brent-free bisection+secant hybrid on compactness(yc) - target."""
    flo = compactness(lo, dr) - target_c
    fhi = compactness(hi, dr) - target_c
    if flo*fhi > 0:
        raise RuntimeError('yc bracket does not straddle target compactness '
                           '(f(%g)=%g, f(%g)=%g)' % (lo, flo, hi, fhi))
    for _ in range(200):
        mid = 0.5*(lo + hi)
        fmid = compactness(mid, dr) - target_c
        if fmid == 0.0 or (hi - lo) < tol*max(1.0, mid):
            return mid
        if flo*fmid < 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5*(lo + hi)


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M0', type=float, default=0.61)
    ap.add_argument('--R0', type=float, default=30.0)
    ap.add_argument('--dr', type=float, default=0.001,
                    help='ODE step in B=1 units (surface radius is O(10))')
    ap.add_argument('--table-out', default=None, help='write profile table (npz)')
    ap.add_argument('--json-out', default=None, help='write scalar constants (json)')
    args = ap.parse_args()

    target_c = 2.0*args.M0/args.R0
    ecut = math.sqrt(1.0 - target_c)

    # --- check closed-form moments against quadrature
    print('# moment closed-form vs quadrature check')
    worst = 0.0
    for y in (0.005, 0.02, 0.0550409773, 0.1, 0.3):
        rq, pq, nq = moments_by_quadrature(y)
        rc, pc, nc = rho0_over_4piB(y), p0_over_4piB(y), n0_over_4piB(y)
        for a, b in ((rq, rc), (pq, pc), (nq, nc)):
            worst = max(worst, abs(a - b)/max(abs(b), 1e-300))
        print('  y=%-12g rho:%.3e  p:%.3e  n:%.3e' %
              (y, abs(rq-rc)/rc, abs(pq-pc)/pc, abs(nq-nc)/nc))
    if worst > 1e-6:
        sys.exit('FATAL: closed-form moments disagree with quadrature (%.3e)' % worst)

    # --- 1D shooting in yc (B=1), then rescale
    yc = solve_yc(target_c, args.dr)
    sol = integrate(yc, args.dr, store=True)
    lam = args.R0/sol['R0']
    B = 1.0/lam**2
    A = B/ecut
    M0_num = lam*sol['M0']

    print('\n# solution')
    print('  yc    = %.10f   (reference 0.0550409773)' % yc)
    print('  Ecut  = %.10f   (reference 0.9794556311)' % ecut)
    print('  A     = %.10f   (reference 0.0389329770)' % A)
    print('  B     = %.10f   (reference 0.0381331236)' % B)
    print('  M0    = %.10f   (target %.10f)' % (M0_num, args.M0))
    print('  R0    = %.10f   (target %.10f)' % (lam*sol['R0'], args.R0))

    # --- step-size convergence of A (PDF: <1.2e-9 absolute over dr 0.004..0.0005)
    print('\n# step-size convergence (A vs dr)')
    for drr in (0.004, 0.002, 0.001, 0.0005):
        ycc = solve_yc(target_c, drr)
        s = integrate(ycc, drr)
        ll = args.R0/s['R0']
        print('  dr=%-8g yc=%.12f A=%.12f' % (drr, ycc, (1.0/ll**2)/ecut))

    # --- physical profile in isotropic coordinates
    prof = sol['prof']
    r = lam*prof['r']            # areal radius, physical units
    m = lam*prof['m']
    y = prof['y']
    h = prof['h']
    R0_s = lam*sol['R0']
    Riso_s = 0.5*(args.R0 - args.M0 + math.sqrt(args.R0*(args.R0 - 2.0*args.M0)))
    # R(r) = C r e^{h}; C from boundary
    C = Riso_s/(R0_s*math.exp(sol['hs']))
    with np.errstate(divide='ignore', invalid='ignore'):
        R = C*r*np.exp(h)
    R[0] = 0.0
    psi0 = np.ones_like(r)
    psi0[1:] = np.sqrt(r[1:]/R[1:])
    # exact center limit: R ~ C r near 0 (h->0), so psi0(0) = 1/sqrt(C)
    psi0[0] = 1.0/math.sqrt(C)
    alpha0 = ecut*np.exp(-y)
    rho0 = np.array([4.0*math.pi*B*rho0_over_4piB(yy) for yy in y])
    p0 = np.array([4.0*math.pi*B*p0_over_4piB(yy) for yy in y])
    n0 = np.array([4.0*math.pi*B*n0_over_4piB(yy) for yy in y])
    q0 = psi0**5*rho0

    # --- Schwarzschild matching check at the surface
    match = abs(psi0[-1] - (1.0 + args.M0/(2.0*Riso_s)))
    print('\n# surface matching: |psi0(Rs) - (1+M0/2Rs)| = %.3e' % match)

    # --- flat-Laplacian identity on the tabulated profile (interior points)
    # non-uniform three-point Laplacian of psi0(R)
    lap = np.zeros_like(R)
    for i in range(2, len(R) - 2):
        dRm = R[i] - R[i-1]
        dRp = R[i+1] - R[i]
        d2 = 2.0*(psi0[i-1]*dRp - psi0[i]*(dRm + dRp) + psi0[i+1]*dRm)/(dRm*dRp*(dRm+dRp))
        d1 = (psi0[i+1]*dRm*dRm - psi0[i-1]*dRp*dRp +
              psi0[i]*(dRp*dRp - dRm*dRm))/(dRm*dRp*(dRm + dRp))
        lap[i] = d2 + 2.0*d1/R[i]
    rhs_h = -2.0*math.pi*q0
    sel = slice(5, len(R) - 5)
    num = np.linalg.norm((lap - rhs_h)[sel])
    den = np.linalg.norm(rhs_h[sel])
    print('# flat Laplacian check: ||lap psi0 + 2 pi q0|| / ||2 pi q0|| = %.3e '
          '(interior, table resolution)' % (num/den))

    # --- masses
    M_ADM_from_psi = 2.0*R[-1]*(psi0[-1] - 1.0)
    elam = 1.0/np.sqrt(np.maximum(1.0 - 2.0*m/np.maximum(r, 1e-30), 1e-30))
    M0_rest = np.trapz(4.0*math.pi*r*r*elam*n0, r)
    M0_rest_iso = np.trapz(4.0*math.pi*R*R*psi0**6*n0, R)
    q0_int = np.trapz(4.0*math.pi*R*R*q0, R)
    print('\n# masses')
    print('  M0 (areal)                 = %.10f' % m[-1])
    print('  2 R (psi0-1) at surface    = %.10f' % M_ADM_from_psi)
    print('  int 4pi R^2 q0 dR          = %.10f  (= M0 by Eq. 20 + Gauss)' % q0_int)
    print('  rest mass  int e^lam n0    = %.10f' % M0_rest)
    print('  rest mass  int psi0^6 n0   = %.10f  (same integral, isotropic form)'
          % M0_rest_iso)

    # --- clump table checks (Eq. 28 / Table I)
    clumps = [(0.12, (-3.0, 0.0, 0.0), 0.70, 0.02),
              (0.12, (3.0, 0.0, 0.0), 0.70, 0.02),
              (0.09, (0.0, 8.0, 0.0), 0.85, 0.02),
              (0.06, (0.0, -12.0, 0.0), 0.90, 0.02)]
    sM = sum(c[0] for c in clumps)
    sMX = [sum(c[0]*c[1][k] for c in clumps) for k in range(3)]
    print('\n# clump table: sum Ma = %.10f (0.39), sum Ma Xa = (%g, %g, %g)'
          % (sM, *sMX))
    print('  M_ADM (continuum) = M0 + sum Ma = %.10f' % (M0_num + sM))

    if args.table_out:
        np.savez(args.table_out, R=R, r=r, psi0=psi0, alpha0=alpha0, rho0=rho0,
                 p0=p0, n0=n0, q0=q0, y=y, m=m)
        print('\nwrote table: %s (%d rows)' % (args.table_out, len(R)))
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump({'yc': yc, 'Ecut': ecut, 'A': A, 'B': B,
                       'M0': M0_num, 'R0': lam*sol['R0'], 'Riso_surface': Riso_s,
                       'M0_rest': float(M0_rest), 'psi0_center': float(psi0[0]),
                       'alpha0_center': float(alpha0[0]), 'dr': args.dr}, f, indent=1)
        print('wrote constants: %s' % args.json_out)


if __name__ == '__main__':
    main()
