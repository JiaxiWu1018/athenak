#!/usr/bin/env python3
"""Relativistic Plummer Einstein-cluster: 1D continuum construction.

Builds the static spherical circular-orbit ("Einstein cluster") solution whose
static-observer energy density eps(r) = T_{mu nu} n^mu n^nu has the Plummer form
in AREAL radius r, truncated hard at r = r_t with vacuum outside, and normalised
so that the ADM mass is exactly M.

Conventions (G = c = 1):
    eps(r)  = 3 M_P /(4 pi b^3) (1 + r^2/b^2)^{-5/2},  r <  r_t ;  0 for r > r_t
    m(r)    = M_P r^3/(r^2+b^2)^{3/2},                 r <= r_t ;  M for r >= r_t
    f_t     = r_t^3/(r_t^2+b^2)^{3/2},   M_P = M/f_t
    B       = (1-2m/r)^{-1/2}
    v_c^2   = m/(r-2m)
    W       = sqrt((r-2m)/(r-3m))
    p_r = 0,   p_t = eps v_c^2/2
    Phi' = m/(r(r-2m)),           Phi = ln alpha
    j'   = (B-1)/r,               j   = ln(R/r)
Matched at r_t to the exact Schwarzschild exterior:
    Phi(r_t) = 1/2 ln(1-2M/r_t)
    R_t      = (r_t - M + sqrt(r_t (r_t-2M)))/2,   j(r_t) = ln(R_t/r_t)

Everything is integrated INWARD from r_t with composite Gauss-Legendre panels
(the integrands are analytic on (0, r_t]), so the table is accurate to close to
double-precision roundoff without any ODE-stepper error control worries.

This module is the single source of truth for the campaign's continuum numbers.
"""
import numpy as np

# ---------------------------------------------------------------- parameters
class PlummerModel:
    def __init__(self, M=1.0, b=20.0, rt=400.0, npanel=6000, ngl=16):
        self.M, self.b, self.rt = float(M), float(b), float(rt)
        self.ft = rt**3 / (rt*rt + b*b)**1.5
        self.MP = self.M / self.ft
        self.npanel, self.ngl = npanel, ngl
        self._build()

    # ------------------------------------------------------ analytic profile
    def eps(self, r):
        r = np.asarray(r, dtype=float)
        out = np.zeros_like(r)
        i = r < self.rt
        out[i] = 3.0*self.MP/(4.0*np.pi*self.b**3) * (1.0 + (r[i]/self.b)**2)**-2.5
        return out

    def m(self, r):
        r = np.asarray(r, dtype=float)
        out = np.full_like(r, self.M)
        i = r <= self.rt
        out[i] = self.MP * r[i]**3 / (r[i]**2 + self.b**2)**1.5
        return out

    def dmdr(self, r):          # = 4 pi r^2 eps
        r = np.asarray(r, dtype=float)
        out = np.zeros_like(r)
        i = r < self.rt
        out[i] = 3.0*self.MP*self.b**2 * r[i]**2 / (r[i]**2 + self.b**2)**2.5
        return out

    def d2mdr2(self, r):
        r = np.asarray(r, dtype=float)
        out = np.zeros_like(r)
        i = r < self.rt
        x, b2 = r[i], self.b**2
        out[i] = 3.0*self.MP*b2 * (2.0*x*(x*x+b2)**2.5 - x*x*2.5*(x*x+b2)**1.5*2.0*x) \
                 / (x*x+b2)**5
        return out

    # --------------------------------------------------- derived local fields
    @staticmethod
    def _Bm1_of_x(x):
        """B-1 = (1-2x)^{-1/2} - 1 with x = m/r, evaluated without
        catastrophic cancellation as x -> 0 (series: x + 3x^2/2 + 5x^3/2 + ...)."""
        x = np.asarray(x, float)
        small = np.abs(x) < 1.0e-4
        out = np.empty_like(x)
        xs = x[small]
        out[small] = xs*(1.0 + 1.5*xs*(1.0 + (5.0/3.0)*xs*(1.0 + 1.75*xs)))
        xb = x[~small]
        out[~small] = (1.0 - 2.0*xb)**-0.5 - 1.0
        return out

    def Bm1(self, r):
        r = np.atleast_1d(np.asarray(r, float))
        return self._Bm1_of_x(self.m(r)/r)

    def B(self, r):
        return 1.0 + self.Bm1(r)

    def vc2(self, r):
        r = np.asarray(r, float); mm = self.m(r)
        return mm/(r - 2.0*mm)

    def W(self, r):
        r = np.asarray(r, float); mm = self.m(r)
        return np.sqrt((r - 2.0*mm)/(r - 3.0*mm))

    def pt(self, r):
        return 0.5*self.eps(r)*self.vc2(r)

    def dPhidr(self, r):
        r = np.asarray(r, float); mm = self.m(r)
        return mm/(r*(r - 2.0*mm))

    def djdr(self, r):
        r = np.atleast_1d(np.asarray(r, float))
        return self.Bm1(r)/r

    # ------------------------------------------------------------ quadrature
    def _panels(self):
        """Node edges clustered near 0 and near the cutoff."""
        b, rt, n = self.b, self.rt, self.npanel
        # sinh-stretched in log-ish space: dense at small r, still fine near rt
        s = np.linspace(0.0, 1.0, n + 1)
        edges = rt * (np.expm1(4.0*s)/np.expm1(4.0))
        edges[0] = 0.0
        edges[-1] = rt
        return edges

    def _build(self):
        edges = self._panels()
        x, w = np.polynomial.legendre.leggauss(self.ngl)
        a, c = edges[:-1], edges[1:]
        half, mid = 0.5*(c - a), 0.5*(c + a)
        # quadrature abscissae, shape (npanel, ngl)
        rq = mid[:, None] + half[:, None]*x[None, :]
        wq = half[:, None]*w[None, :]

        # panel integrals of Phi', j', and the rest-mass measure
        IPhi = np.sum(wq*self.dPhidr(rq), axis=1)
        Ij = np.sum(wq*self.djdr(rq), axis=1)
        rest = 4.0*np.pi*rq**2*self.eps(rq)*self.B(rq)/self.W(rq)
        IM0 = np.sum(wq*rest, axis=1)

        # boundary values at r_t from the exact exterior match
        Phi_t = 0.5*np.log(1.0 - 2.0*self.M/self.rt)
        Rt = 0.5*(self.rt - self.M + np.sqrt(self.rt*(self.rt - 2.0*self.M)))
        j_t = np.log(Rt/self.rt)
        self.Rt = Rt

        # integrate inward: Phi(r_k) = Phi_t - sum_{panels above r_k}
        tail_Phi = np.concatenate(([0.0], np.cumsum(IPhi[::-1])))[::-1]
        tail_j = np.concatenate(([0.0], np.cumsum(Ij[::-1])))[::-1]
        self.r_tab = edges
        self.Phi_tab = Phi_t - tail_Phi
        self.j_tab = j_t - tail_j
        self.M0_tab = np.concatenate(([0.0], np.cumsum(IM0)))
        self.M0 = self.M0_tab[-1]

        self.alpha_tab = np.exp(self.Phi_tab)
        self.R_tab = self.r_tab*np.exp(self.j_tab)
        self.psi_tab = np.exp(-0.5*self.j_tab)
        self.F0_tab = self.M0_tab/self.M0

    # ------------------------------------------------------- interpolation
    def _interp(self, r, tab):
        return np.interp(np.asarray(r, float), self.r_tab, tab)

    def Phi(self, r):
        return self._interp(r, self.Phi_tab)

    def alpha(self, r):
        r = np.asarray(r, float)
        out = np.exp(self._interp(r, self.Phi_tab))
        ext = r > self.rt
        if np.any(ext):                     # exact Schwarzschild outside
            out[ext] = np.sqrt(1.0 - 2.0*self.M/r[ext])
        return out

    def j(self, r):
        return self._interp(r, self.j_tab)

    def R_of_r(self, r):
        r = np.asarray(r, float)
        out = r*np.exp(self._interp(r, self.j_tab))
        ext = r > self.rt
        if np.any(ext):
            out[ext] = 0.5*(r[ext] - self.M + np.sqrt(r[ext]*(r[ext] - 2.0*self.M)))
        return out

    def r_of_R(self, R):
        """Invert the monotone map R(r).  Exterior uses the exact inverse."""
        R = np.asarray(R, float)
        out = np.interp(R, self.R_tab, self.r_tab)
        ext = R > self.Rt
        if np.any(ext):
            out[ext] = R[ext]*(1.0 + self.M/(2.0*R[ext]))**2
        return out

    def psi_of_R(self, R):
        R = np.asarray(R, float)
        out = np.exp(-0.5*np.interp(R, self.R_tab, self.j_tab))
        ext = R > self.Rt
        if np.any(ext):
            out[ext] = 1.0 + self.M/(2.0*R[ext])
        return out

    def alpha_of_R(self, R):
        R = np.asarray(R, float)
        out = np.exp(np.interp(R, self.R_tab, self.Phi_tab))
        ext = R > self.Rt
        if np.any(ext):
            u = self.M/(2.0*R[ext])
            out[ext] = (1.0 - u)/(1.0 + u)
        return out

    # ------------------------------------------------------------ key radii
    def r_half(self):
        """m(r_half) = M/2, solved analytically from the Plummer mass law."""
        # M_P r^3/(r^2+b^2)^{3/2} = M/2  ->  let y = (r/b)^2
        # M_P y^{3/2} = (M/2)(1+y)^{3/2}  ->  y/(1+y) = ((M/(2 M_P))^{2/3})
        k = (0.5*self.M/self.MP)**(2.0/3.0)
        y = k/(1.0 - k)
        return self.b*np.sqrt(y)

    def P_half(self):
        rh = self.r_half()
        a = float(self.alpha(np.array([rh]))[0])
        v = float(np.sqrt(self.vc2(np.array([rh]))[0]))
        return 2.0*np.pi*rh/(a*v), rh, a, v


    # ------------------------------------------------- exact (non-interpolated)
    def _exact_from_node(self, r, tab, deriv):
        """Phi(r) or j(r) evaluated exactly: take the tabulated value at the
        nearest node ABOVE r and subtract the Gauss-Legendre integral of the
        derivative from r up to that node.  No interpolation error."""
        r = np.atleast_1d(np.asarray(r, float))
        k = np.searchsorted(self.r_tab, r, side='left')
        k = np.clip(k, 0, len(self.r_tab) - 1)
        rk = self.r_tab[k]
        x, w = np.polynomial.legendre.leggauss(self.ngl)
        half = 0.5*(rk - r)
        mid = 0.5*(rk + r)
        rq = mid[:, None] + half[:, None]*x[None, :]
        wq = half[:, None]*w[None, :]
        integ = np.sum(wq*deriv(rq), axis=1)
        return tab[k] - integ

    def Phi_exact(self, r):
        return self._exact_from_node(r, self.Phi_tab, self.dPhidr)

    def j_exact(self, r):
        return self._exact_from_node(r, self.j_tab, self.djdr)

    def vc_max(self):
        """Exact max of v_c over the occupied range, by scalar root-finding on
        d(v_c^2)/dr = 0."""
        from scipy.optimize import brentq
        def dvc2(r):
            mm = float(self.m(np.array([r]))[0])
            mp = float(self.dmdr(np.array([r]))[0])
            # d/dr [ m/(r-2m) ] = [m'(r-2m) - m(1-2m')]/(r-2m)^2
            return mp*(r - 2*mm) - mm*(1.0 - 2.0*mp)
        lo, hi = 1e-6*self.b, min(self.rt*0.999999, 1e4*self.b)
        rs = np.geomspace(lo, hi, 20000)
        f = np.array([dvc2(x) for x in rs])
        idx = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]
        if len(idx) == 0:
            i = int(np.argmax(self.vc2(rs)))
            return float(np.sqrt(self.vc2(np.array([rs[i]]))[0])), float(rs[i])
        rstar = brentq(dvc2, rs[idx[0]], rs[idx[0]+1], xtol=1e-15, rtol=1e-15)
        return float(np.sqrt(self.vc2(np.array([rstar]))[0])), float(rstar)


    def F0_exact(self, r):
        """F0(r) = M0(r)/M0 evaluated exactly (node value + GL remainder)."""
        r = np.atleast_1d(np.asarray(r, float))
        k = np.searchsorted(self.r_tab, r, side='left')
        k = np.clip(k, 0, len(self.r_tab) - 1)
        rk = self.r_tab[k]
        x, w = np.polynomial.legendre.leggauss(self.ngl)
        half = 0.5*(rk - r)
        mid = 0.5*(rk + r)
        rq = mid[:, None] + half[:, None]*x[None, :]
        wq = half[:, None]*w[None, :]
        integ = np.sum(wq*self.dM0dr(rq), axis=1)
        return (self.M0_tab[k] - integ)/self.M0

    def dM0dr(self, r):
        r = np.asarray(r, float)
        return 4.0*np.pi*r**2*self.eps(r)*self.B(r)/self.W(r)

    def invert_F0_exact(self, q, niter=4):
        """r = F0^{-1}(q) to machine precision: table bracket + Newton on F0_exact."""
        q = np.atleast_1d(np.asarray(q, float))
        r = np.interp(q, self.F0_tab, self.r_tab)
        for _ in range(niter):
            f = self.F0_exact(r) - q
            fp = self.dM0dr(r)/self.M0
            step = f/fp
            r = np.clip(r - step, 1.0e-12, self.rt)
        return r

    # ---------------------------------------------- constraint residual check
    def constraint_residuals(self, r):
        """Return (res_H, res_L, scale_H, scale_L) for
        Lap_flat psi + 2 pi psi^5 eps = 0  and
        Lap_flat(alpha psi) - 2 pi alpha psi^5 (eps + 2 S) = 0,  S = 2 p_t,
        with Lap_flat f = f'' + (2/R) f' in isotropic radius R.
        All derivatives are analytic (chain rule through r)."""
        r = np.asarray(r, float)
        mm, mp = self.m(r), self.dmdr(r)
        mpp = self.d2mdr2(r)
        h = 1.0 - 2.0*mm/r
        Bm1v = self._Bm1_of_x(mm/r)
        Bv = 1.0 + Bm1v
        hp = -2.0*(mp*r - mm)/r**2
        Bp = -0.5*Bv**3*hp
        jp = Bm1v/r
        jpp = Bp/r - Bm1v/r**2
        D = r*(r - 2.0*mm)
        Dp = 2.0*r - 2.0*mp*r - 2.0*mm
        Php = mm/D
        Phpp = mp/D - mm*Dp/D**2

        jv = self.j_exact(r)
        Phv = self.Phi_exact(r)
        psi = np.exp(-0.5*jv)
        alp = np.exp(Phv)
        Rv = r*np.exp(jv)
        dRdr = Bv*np.exp(jv)

        # psi
        g = -0.5*jp*np.exp(-1.5*jv)/Bv                       # dpsi/dR
        dgdr = np.exp(-1.5*jv)*(-0.5*jpp/Bv + 0.75*jp*jp/Bv + 0.5*jp*Bp/Bv**2)
        d2psi = dgdr/dRdr
        lap_psi = d2psi + 2.0*g/Rv

        # alpha*psi
        A = alp*psi
        q = Php - 0.5*jp
        qp = Phpp - 0.5*jpp
        G = q*A*np.exp(-jv)/Bv                                # d(alpha psi)/dR
        dAdr = q*A
        dfac = np.exp(-jv)*(-jp)/Bv - np.exp(-jv)*Bp/Bv**2
        dGdr = (qp*A + q*dAdr)*np.exp(-jv)/Bv + q*A*dfac
        d2A = dGdr/dRdr
        lap_A = d2A + 2.0*G/Rv

        e = self.eps(r)
        S = 2.0*self.pt(r)
        srcH = -2.0*np.pi*psi**5*e
        srcL = 2.0*np.pi*alp*psi**5*(e + 2.0*S)
        return (lap_psi - srcH, lap_A - srcL,
                np.maximum(np.abs(lap_psi), np.abs(srcH)),
                np.maximum(np.abs(lap_A), np.abs(srcL)))

    # ---------------------------------------------- radial orbit stability
    def radial_stability(self, r):
        """r^2 m' + r m - 6 m^2 > 0 for a stable individual circular orbit."""
        r = np.asarray(r, float)
        return r**2*self.dmdr(r) + r*self.m(r) - 6.0*self.m(r)**2

    # ------------------------------------------------------ sampler support
    def invert_F0(self, q):
        """r = F0^{-1}(q), monotone interpolation on the built table."""
        return np.interp(np.asarray(q, float), self.F0_tab, self.r_tab)


# ------------------------------------------------------ infinite-model check
class PlummerInfinite(PlummerModel):
    """Uncut Plummer profile (rt -> infinity); used only for the PDF's
    published reference numbers at M_P/b = 0.05."""
    def __init__(self, MP=1.0, b=20.0, rmax=1.0e7, npanel=20000, ngl=16):
        self.M = MP          # ADM mass equals M_P for the uncut model
        self.b = float(b)
        self.rt = float(rmax)
        self.ft = 1.0
        self.MP = float(MP)
        self.npanel, self.ngl = npanel, ngl
        self._build()

    def eps(self, r):
        r = np.asarray(r, dtype=float)
        return 3.0*self.MP/(4.0*np.pi*self.b**3)*(1.0 + (r/self.b)**2)**-2.5

    def m(self, r):
        r = np.asarray(r, dtype=float)
        return self.MP*r**3/(r**2 + self.b**2)**1.5

    def dmdr(self, r):
        r = np.asarray(r, dtype=float)
        return 3.0*self.MP*self.b**2*r**2/(r**2 + self.b**2)**2.5

    def _panels(self):
        # geometric-ish stretch out to rmax
        s = np.linspace(0.0, 1.0, self.npanel + 1)
        edges = self.b*(np.expm1(s*np.log1p(self.rt/self.b)))
        edges[0] = 0.0
        edges[-1] = self.rt
        return edges

    def _build(self):
        edges = self._panels()
        x, w = np.polynomial.legendre.leggauss(self.ngl)
        a, c = edges[:-1], edges[1:]
        half, mid = 0.5*(c - a), 0.5*(c + a)
        rq = mid[:, None] + half[:, None]*x[None, :]
        wq = half[:, None]*w[None, :]
        IPhi = np.sum(wq*self.dPhidr(rq), axis=1)
        Ij = np.sum(wq*self.djdr(rq), axis=1)
        rest = 4.0*np.pi*rq**2*self.eps(rq)*self.B(rq)/self.W(rq)
        IM0 = np.sum(wq*rest, axis=1)
        # asymptotic boundary values at r = rmax (Schwarzschild with m ~ M_P)
        Phi_t = 0.5*np.log(1.0 - 2.0*self.MP/self.rt)
        Rt = 0.5*(self.rt - self.MP + np.sqrt(self.rt*(self.rt - 2.0*self.MP)))
        j_t = np.log(Rt/self.rt)
        self.Rt = Rt
        tail_Phi = np.concatenate(([0.0], np.cumsum(IPhi[::-1])))[::-1]
        tail_j = np.concatenate(([0.0], np.cumsum(Ij[::-1])))[::-1]
        self.r_tab = edges
        self.Phi_tab = Phi_t - tail_Phi
        self.j_tab = j_t - tail_j
        self.M0_tab = np.concatenate(([0.0], np.cumsum(IM0)))
        self.M0 = self.M0_tab[-1]
        self.alpha_tab = np.exp(self.Phi_tab)
        self.R_tab = self.r_tab*np.exp(self.j_tab)
        self.psi_tab = np.exp(-0.5*self.j_tab)
        self.F0_tab = self.M0_tab/self.M0
