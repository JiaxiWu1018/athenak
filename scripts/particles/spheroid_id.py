#!/usr/bin/env python3
"""Particle-consistent initial data for the Shapiro-Teukolsky prolate spheroid.

Implements the prescription of naked_singularity_initial_data_moving_puncture.pdf:

  * continuum reference   Psi_bar = 1 - Phi_N  (homogeneous prolate spheroid potential,
    interior Eq. (13), exterior from the ellipsoidal-coordinate closed form),
  * equal-mass particles  m_p = M0/N, M0 = M + 3 M^2/(10 b e) ln((1+e)/(1-e)),
    positions sampled with p(x) ~ Psi_bar(x) inside the ellipsoid, all at rest (u_i = 0),
  * symmetric realization: every base point is replicated under z -> -z and the
    azimuthal antipodal map (x,y) -> (-x,-y) (fold=4; fold=8 adds x -> -x), so the
    deposited centre of mass vanishes exactly while the realization stays genuinely 3D,
  * CIC deposit of sigma = sum_p m_p W_p on a uniform fine grid that coincides with the
    finest initial AthenaK level (cell centres at xmin + (i+1/2) dx; left-centre index and
    weights exactly as particles_tmunu.cpp),
  * nonlinear Hamiltonian constraint  Lap Psi = -(2 pi / Psi) sigma  solved by Picard
    iteration with an isolated-boundary FFT Green's function (cube-averaged 1/r kernel),
  * one-shot rescaling of the particle masses so that the discrete ADM mass
    M_ADM = sum_cells (sigma/Psi) dV equals the target M.

Output: a single binary file read by src/pgen/particles/nr_pic_spheroid.cpp containing the
particles, the fine Psi table and the converged source list s_c dV (so the pgen evaluates
Psi outside the table by the identical discrete convolution), plus a JSON diagnostics
file.
"""
import argparse
import json
import sys
import time

import numpy as np

MAGIC = b"NRPICSPH"
VERSION = 1
# int_{unit cube} d^3x / |x|  (potential at the centre of a unit cube)
CUBE_SELF = 2.3800774


# ----------------------------------------------------------------------------------------
# analytic continuum reference Psi_bar
# ----------------------------------------------------------------------------------------
class Spheroid:
    def __init__(self, M=1.0, e=0.9, b=10.0):
        self.M, self.e, self.b = float(M), float(e), float(b)
        self.a = self.b * np.sqrt(1.0 - self.e**2)
        self.beta0 = np.arctanh(self.e)
        self.eps = self.b * self.e                  # sqrt(b^2 - a^2)
        self.rhoN = 3.0 * self.M / (8.0 * np.pi * self.a**2 * self.b)
        # rest mass Eq. (19)
        self.M0 = (self.M + 3.0 * self.M**2 / (10.0 * self.b * self.e)
                   * np.log((1 + self.e) / (1 - self.e)))

    def inside(self, x, y, z):
        return (x * x + y * y) / self.a**2 + z * z / self.b**2 <= 1.0

    def phi_interior(self, x, y, z):
        M, b, e, b0 = self.M, self.b, self.e, self.beta0
        w2 = x * x + y * y
        return (-3 * M * b0 / (4 * b * e)
                + 3 * M / (8 * b**3 * e**3) * (e / (1 - e * e) - b0) * w2
                + 3 * M / (4 * b**3 * e**3) * (b0 - e) * z * z)

    def lam(self, x, y, z):
        """largest root of w^2/(a^2+l) + z^2/(b^2+l) = 1 (confocal coordinate)."""
        a2, b2 = self.a**2, self.b**2
        w2 = x * x + y * y
        z2 = z * z
        B = a2 + b2 - w2 - z2
        C = a2 * b2 - w2 * b2 - z2 * a2
        disc = np.sqrt(np.maximum(B * B - 4.0 * C, 0.0))
        lam = 0.5 * (-B + disc)
        # numerically robust alternative when -B + disc suffers cancellation (B > 0):
        alt = np.where(np.abs(B + disc) > 0, -2.0 * C / (B + disc), lam)
        lam = np.where(B > 0, alt, lam)
        return np.maximum(lam, 0.0)

    def phi_exterior(self, x, y, z):
        """Phi = -(3M/8) [I(l) - w^2 A1(l) - z^2 A3(l)], closed forms with
        s = sqrt(b^2+l)."""
        eps = self.eps
        lam = self.lam(x, y, z)
        s = np.sqrt(self.b**2 + lam)
        at = np.arctanh(eps / s)
        Ilam = 2.0 * at / eps
        A1 = s / (eps**2 * (self.a**2 + lam)) - at / eps**3
        A3 = 2.0 * at / eps**3 - 2.0 / (eps**2 * s)
        w2 = x * x + y * y
        return -(3.0 * self.M / 8.0) * (Ilam - w2 * A1 - z * z * A3)

    def phi(self, x, y, z):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        z = np.asarray(z, float)
        ins = self.inside(x, y, z)
        out = np.empty(np.broadcast(x, y, z).shape)
        out[...] = self.phi_exterior(x, y, z)
        phin = self.phi_interior(x, y, z)
        out = np.where(ins, phin, out)
        return out

    def psi_bar(self, x, y, z):
        return 1.0 - self.phi(x, y, z)


# ----------------------------------------------------------------------------------------
# particle sampling
# ----------------------------------------------------------------------------------------
def sample_particles(sph, N, fold, seed):
    """Equal-mass particles with p(x) ~ Psi_bar inside the ellipsoid, replicated under the
    chosen reflection group. Returns positions (N,3) and the base count."""
    if N % fold:
        raise ValueError("N must be a multiple of fold")
    nbase = N // fold
    rng = np.random.default_rng(seed)
    psi0 = sph.psi_bar(0.0, 0.0, 0.0)
    pts = []
    got = 0
    while got < nbase:
        n = int(1.2 * (nbase - got)) + 1000
        u = rng.random((n, 3))
        # uniform in the unit ball: direction * cbrt(U)
        g = rng.standard_normal((n, 3))
        g /= np.linalg.norm(g, axis=1)[:, None]
        r = np.cbrt(u[:, 0])
        p = g * r[:, None]
        p[:, 0] *= sph.a
        p[:, 1] *= sph.a
        p[:, 2] *= sph.b
        # fundamental domain of the fold group
        if fold >= 2:
            p[:, 2] = np.abs(p[:, 2])
        if fold >= 4:
            p[:, 1] = np.abs(p[:, 1])
        if fold >= 8:
            p[:, 0] = np.abs(p[:, 0])
        acc = u[:, 1] * psi0 <= sph.psi_bar(p[:, 0], p[:, 1], p[:, 2])
        p = p[acc]
        pts.append(p)
        got += len(p)
    base = np.concatenate(pts)[:nbase]
    imgs = [base]
    if fold >= 2:
        imgs = imgs + [q * np.array([1, 1, -1.0]) for q in imgs]
    if fold >= 4:
        imgs = imgs + [q * np.array([-1, -1, 1.0]) for q in imgs]
    if fold >= 8:
        imgs = imgs + [q * np.array([-1, 1, 1.0]) for q in imgs]
    pos = np.concatenate(imgs)
    assert pos.shape == (N, 3)
    return pos, nbase


# ----------------------------------------------------------------------------------------
# CIC deposit replicating particles_tmunu.cpp (cell centres at xmin + (i+1/2) dx)
# ----------------------------------------------------------------------------------------
class Grid:
    def __init__(self, xmin, xmax, nx):
        self.xmin = np.asarray(xmin, float)
        self.xmax = np.asarray(xmax, float)
        self.n = np.asarray(nx, int)
        self.dx = (self.xmax - self.xmin) / self.n
        assert np.allclose(self.dx, self.dx[0]), "cubic cells required"
        self.h = float(self.dx[0])
        self.dV = self.h**3

    def centers(self, d):
        return self.xmin[d] + (np.arange(self.n[d]) + 0.5) * self.dx[d]

    def deposit(self, pos, mass):
        """sigma = sum_p m_p W_p / dV on the grid (coordinate density). Particles whose
        stencil leaves the grid are an error (the grid must enclose the support)."""
        sig = np.zeros(self.n[::-1], dtype=float)  # [z][y][x]
        idx = np.empty((len(pos), 3), int)
        dl = np.empty((len(pos), 3), float)
        for d in range(3):
            t = (pos[:, d] - self.xmin[d]) / self.dx[d] - 0.5
            i = np.floor(t).astype(int)                # largest i with centre <= x
            delta = np.clip(t - i, 0.0, 1.0)
            if (i < 0).any() or (i + 1 > self.n[d] - 1).any():
                raise RuntimeError(
                    "particle CIC stencil leaves the fine grid in dim %d" % d)
            idx[:, d] = i
            dl[:, d] = delta
        flat = sig.reshape(-1)
        nx, ny, nz = self.n
        for kz in (0, 1):
            wz = dl[:, 2] if kz else 1.0 - dl[:, 2]
            for ky in (0, 1):
                wy = dl[:, 1] if ky else 1.0 - dl[:, 1]
                for kx in (0, 1):
                    wx = dl[:, 0] if kx else 1.0 - dl[:, 0]
                    lin = (((idx[:, 2] + kz) * ny + (idx[:, 1] + ky)) * nx
                           + (idx[:, 0] + kx))
                    flat += np.bincount(lin, weights=mass * wx * wy * wz,
                                        minlength=flat.size)
        return sig / self.dV


# ----------------------------------------------------------------------------------------
# isolated-boundary Poisson solver:  u(x) = (1/2) sum_c s_c dV K(x - x_c)
# ----------------------------------------------------------------------------------------
class GreensFFT:
    """Hockney-Eastwood zero-padded convolution with the kernel K(r) = 1/r (r != 0),
    K(0) = CUBE_SELF/h. The pgen evaluates the same discrete sum directly for cells
    outside the table, so both representations agree to roundoff."""

    def __init__(self, grid):
        self.g = grid
        n = grid.n
        self.np_ = 2 * n
        h = grid.h
        ker = np.zeros(self.np_[::-1])  # [z][y][x]
        ax = []
        for d in range(3):
            i = np.arange(self.np_[d])
            i = np.where(i > n[d], i - self.np_[d], i)  # minimum-image separation index
            ax.append(i * h)
        Z, Y, X = np.meshgrid(ax[2], ax[1], ax[0], indexing="ij")
        r = np.sqrt(X * X + Y * Y + Z * Z)
        with np.errstate(divide="ignore"):
            ker = np.where(r > 0, 1.0 / r, CUBE_SELF / h)
        self.kfft = np.fft.rfftn(ker)
        del ker, X, Y, Z, r

    def potential(self, s):
        """u = (1/2) K * (s dV)"""
        n = self.g.n
        pad = np.zeros(self.np_[::-1])
        pad[: n[2], : n[1], : n[0]] = s * self.g.dV
        u = np.fft.irfftn(np.fft.rfftn(pad) * self.kfft, s=self.np_[::-1], axes=(0, 1, 2))
        return 0.5 * u[: n[2], : n[1], : n[0]]


def solve_hamiltonian(grid, sigma, psi_guess, tol=1e-11, maxit=60, log=print, G=None):
    if G is None:
        G = GreensFFT(grid)
    psi = psi_guess.copy()
    hist = []
    for it in range(maxit):
        s = sigma / psi
        psi_new = 1.0 + G.potential(s)
        d = float(np.max(np.abs(psi_new - psi)))
        hist.append(d)
        psi = psi_new
        log("  picard it=%2d  max|dPsi|=%.3e  Psi(min,max)=(%.8f,%.8f)  M_ADM=%.10f"
            % (it, d, psi.min(), psi.max(), float(np.sum(sigma / psi) * grid.dV)))
        if d < tol:
            break
    return psi, hist, G


def fd6_first(f, axis, h):
    """6th-order centred first derivative (AthenaK NGHOST=4 Dx stencil); zero within
    3 cells of the array boundary."""
    out = np.zeros_like(f)
    sl = [slice(3, -3)] * 3

    def shift(n):
        s = list(sl)
        s[axis] = slice(3 + n, f.shape[axis] - 3 + n)
        return tuple(s)
    out[tuple(sl)] = (-f[shift(-3)] / 60.0 + 3.0 * f[shift(-2)] / 20.0
                      - 3.0 * f[shift(-1)] / 4.0 + 3.0 * f[shift(1)] / 4.0
                      - 3.0 * f[shift(2)] / 20.0 + f[shift(3)] / 60.0) / h
    return out


def fd6_second(f, axis, h):
    """6th-order centred second derivative (AthenaK NGHOST=4 Dxx stencil)."""
    out = np.zeros_like(f)
    sl = [slice(3, -3)] * 3

    def shift(n):
        s = list(sl)
        s[axis] = slice(3 + n, f.shape[axis] - 3 + n)
        return tuple(s)
    out[tuple(sl)] = (f[shift(-3)] / 90.0 - 3.0 * f[shift(-2)] / 20.0
                      + 3.0 * f[shift(-1)] / 2.0 - 49.0 * f[shift(0)] / 18.0
                      + 3.0 * f[shift(1)] / 2.0 - 3.0 * f[shift(2)] / 20.0
                      + f[shift(3)] / 90.0) / h**2
    return out


def hamiltonian_code(psi, sigma, h):
    """Replica of AthenaK's ADMConstraints Hamiltonian for gamma_ij = Psi^4 delta_ij, K=0:
    H = R - 16 pi E with R = 3|Dq|^2/(2q^3) - 2 D^2 q/q^2, q = Psi^4 (6th-order stencils),
    E = sigma/Psi^6 (cell-centred sqrt(gamma)). Valid 3 cells inside the array
    boundary."""
    q = psi**4
    grad2 = sum(fd6_first(q, ax, h)**2 for ax in range(3))
    lap = sum(fd6_second(q, ax, h) for ax in range(3))
    H = 1.5 * grad2 / q**3 - 2.0 * lap / q**2 - 16.0 * np.pi * sigma / psi**6
    H[:3] = H[-3:] = 0.0
    H[:, :3] = H[:, -3:] = 0.0
    H[:, :, :3] = H[:, :, -3:] = 0.0
    return H


def defect_correct(grid, sigma, psi, G, maxit=40, tol=1e-12, log=print):
    """Drive the code's discrete Hamiltonian residual to zero on the table interior using
    the FFT Green's function as the approximate inverse of the linearised operator
    (F(Psi + d) ~ F(Psi) - 8 Psi^-5 Lap d  =>  Lap d = Psi^5 F/8)."""
    inner = (slice(3, -3),) * 3
    hist = []
    acc = np.zeros_like(psi)                           # accumulated correction sources
    for it in range(maxit):
        F = hamiltonian_code(psi, sigma, grid.h)
        r = float(np.sqrt(np.mean(F[inner]**2)))
        hist.append(r)
        log("  defect it=%2d  rms H_code(interior)=%.3e  max|H|=%.3e"
            % (it, r, float(np.abs(F).max())))
        if r < tol:
            break
        # G.potential(s) solves Lap u = -2 pi s
        rhs = -(psi**5) * F / (16.0 * np.pi)
        d = G.potential(rhs)
        psi = psi + d
        acc += rhs
    return psi, hist, acc


def laplacian7(f, h):
    """second-order 7-point Laplacian on the interior (zero on the boundary layer)."""
    L = np.zeros_like(f)
    L[1:-1, 1:-1, 1:-1] = (f[2:, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1] + f[1:-1, 2:, 1:-1]
                           + f[1:-1, :-2, 1:-1] + f[1:-1, 1:-1, 2:] + f[1:-1, 1:-1, :-2]
                           - 6.0 * f[1:-1, 1:-1, 1:-1]) / h**2
    return L


# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--M", type=float, default=1.0)
    ap.add_argument("--e", type=float, default=0.9)
    ap.add_argument("--b", type=float, default=10.0)
    ap.add_argument("--N", type=int, default=2000000)
    ap.add_argument("--fold", type=int, default=4, choices=[1, 2, 4, 8])
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--box", type=float, nargs=6, default=[-8, 8, -8, 8, -16, 16],
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                    help="fine table box; must coincide with the finest initial level "
                         "footprint")
    ap.add_argument("--dx", type=float, default=0.125, help="fine table spacing")
    ap.add_argument("--rescale-iters", type=int, default=3,
                    help="mass rescaling passes to hit M_ADM=M")
    ap.add_argument("--tol", type=float, default=1e-11)
    ap.add_argument("--src-threshold", type=float, default=1e-13,
                    help="drop effective-source cells below this fraction of the maximum")
    ap.add_argument("--defect-correct", type=int, default=30,
                    help="iterations driving AthenaK's 6th-order discrete Hamiltonian "
                         "to zero (0 = off)")
    ap.add_argument("--out", default="spheroid_id.bin")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    t0 = time.time()
    sph = Spheroid(args.M, args.e, args.b)

    def log(*a):
        print(*a, flush=True)
    log("Spheroid: M=%g e=%g b=%g -> a=%.8f beta0=%.8f M0=%.8f Psi(0)=%.8f "
        "alpha(0)=%.8f rhoN=%.6e"
        % (sph.M, sph.e, sph.b, sph.a, sph.beta0, sph.M0, sph.psi_bar(0, 0, 0),
           sph.psi_bar(0, 0, 0)**-2, sph.rhoN))

    # analytic self-checks: interior Laplacian, surface continuity, far field
    h = 1e-3

    def lap(f, x, y, z):
        return (f(x + h, y, z) + f(x - h, y, z) + f(x, y + h, z) + f(x, y - h, z)
                + f(x, y, z + h) + f(x, y, z - h) - 6 * f(x, y, z)) / h**2
    lin = lap(sph.phi_interior, 0.3, -0.7, 2.1) / (4 * np.pi * sph.rhoN)
    lout = lap(sph.phi_exterior, 3.0, 4.0, 9.0) * (sph.a**3) / sph.M
    th = np.linspace(0.05, np.pi / 2 - 0.05, 7)
    xs, zs = sph.a * np.sin(th), sph.b * np.cos(th)
    cont = np.max(np.abs(sph.phi_interior(xs, 0, zs) - sph.phi_exterior(xs, 0, zs)))
    far = sph.phi_exterior(0.0, 0.0, 3000.0) * 3000.0 / (-sph.M / 2)
    log("analytic checks: Lap(Phi_in)/(4 pi rhoN)=%.8f  Lap(Phi_out)*a^3/M=%.2e  "
        "surface |dPhi|=%.2e  far-field ratio=%.8f"
        % (lin, lout, cont, far))

    # particles
    pos, nbase = sample_particles(sph, args.N, args.fold, args.seed)
    mp = sph.M0 / args.N
    mass = np.full(args.N, mp)
    log("sampled N=%d (fold=%d, base=%d) m_p=%.10e  sum m=%.10f  in %.1fs"
        % (args.N, args.fold, nbase, mp, mass.sum(), time.time() - t0))
    com = (pos * mass[:, None]).sum(0) / mass.sum()
    q = (pos**2 * mass[:, None]).sum(0) / mass.sum()
    # expectation values for p ~ Psi_bar over the ellipsoid (Monte Carlo of the continuum)
    log("COM = (%.3e, %.3e, %.3e)   <x^2>,<y^2>,<z^2> = %.5f %.5f %.5f"
        "  -> a_eff=%.5f b_eff=%.5f e_eff=%.6f"
        % (*com, *q, np.sqrt(5 * 0.5 * (q[0] + q[1])), np.sqrt(5 * q[2]),
           np.sqrt(1 - (q[0] + q[1]) / (2 * q[2]))))

    # grid and deposit
    box = np.array(args.box, float).reshape(3, 2)
    nx = np.rint((box[:, 1] - box[:, 0]) / args.dx).astype(int)
    grid = Grid(box[:, 0], box[:, 1], nx)
    assert np.allclose(grid.dx, args.dx)
    log("fine grid: n=%s  dx=%.6f  box=%s" % (list(nx), grid.h, box.tolist()))
    Zc, Yc, Xc = np.meshgrid(grid.centers(2), grid.centers(1), grid.centers(0),
                             indexing="ij")
    psibar = sph.psi_bar(Xc, Yc, Zc)

    result = {}
    G = None
    dhist = []
    for it in range(args.rescale_iters):
        sigma = grid.deposit(pos, mass)
        log("deposit: sum sigma dV = %.10f (sum m_p = %.10f)  max sigma=%.4e  cells>0: %d"
            % (sigma.sum() * grid.dV, mass.sum(), sigma.max(), int((sigma > 0).sum())))
        psi, hist, G = solve_hamiltonian(grid, sigma, psibar, tol=args.tol, log=log, G=G)
        # Picard fixed point: psi = 1 + G*(sigma/psi)
        s_eff = sigma / psi
        if args.defect_correct > 0:
            psi, dhist, acc = defect_correct(grid, sigma, psi, G,
                                             maxit=args.defect_correct, log=log)
            s_eff = s_eff + acc                   # psi == 1 + G*s_eff by linearity
        M_adm = float(np.sum(s_eff) * grid.dV)
        log("pass %d: M_ADM = %.10f  (target %.10f)" % (it, M_adm, sph.M))
        if abs(M_adm - sph.M) < 1e-9 or it == args.rescale_iters - 1:
            break
        mass *= sph.M / M_adm
        log("  -> rescale particle masses by %.10f (new m_p=%.10e, M0'=%.10f)"
            % (sph.M / M_adm, mass[0], mass.sum()))

    # diagnostics of the solution (s = effective source whose convolution IS the table)
    s = s_eff
    recon = 1.0 + G.potential(s)
    log("representation check: max |Psi_table - (1 + G*s_eff)| = %.3e ; "
        "M_ADM(s_eff)=%.10f  sum(sigma/Psi)dV=%.10f"
        % (float(np.abs(psi - recon).max()), float(s.sum() * grid.dV),
           float(np.sum(sigma / psi) * grid.dV)))
    smax = float(np.abs(s).max())
    keep = np.abs(s) > args.src_threshold * smax
    log("source list: %d cells above %.1e*max (of %d nonzero); dropped monopole %.3e"
        % (int(keep.sum()), args.src_threshold, int((s != 0).sum()),
           float(np.abs(s[~keep]).sum() * grid.dV)))
    inside = sph.inside(Xc, Yc, Zc)
    Hres = -8.0 * psi**-5 * laplacian7(psi, grid.h) - 16.0 * np.pi * sigma / psi**6
    Hbar = (-8.0 * psibar**-5 * laplacian7(psibar, grid.h)
            - 16.0 * np.pi * sigma / psibar**6)
    core = (inside & (np.abs(Xc) < grid.xmax[0] - 2 * grid.h)
            & (np.abs(Yc) < grid.xmax[1] - 2 * grid.h)
            & (np.abs(Zc) < grid.xmax[2] - 2 * grid.h))

    def rms(f):
        return float(np.sqrt(np.mean(f[core]**2)))
    log("Psi: centre(table)=%.8f  min=%.8f max=%.8f  | Psi-Psibar | max=%.3e "
        "rms(inside)=%.3e"
        % (psi[nx[2] // 2, nx[1] // 2, nx[0] // 2], psi.min(), psi.max(),
           float(np.max(np.abs(psi - psibar))),
           float(np.sqrt(np.mean((psi - psibar)[inside]**2)))))
    log("7-point Hamiltonian residual inside (rms): solved Psi %.3e   "
        "analytic Psi_bar + particles %.3e   16 pi <E> = %.3e"
        % (rms(Hres), rms(Hbar),
           16 * np.pi * float(np.mean((sigma / psi**6)[inside]))))
    Hc = hamiltonian_code(psi, sigma, grid.h)
    Hcbar = hamiltonian_code(psibar, sigma, grid.h)
    log("AthenaK-replica (6th-order) Hamiltonian inside (rms): final Psi %.3e  "
        "max %.3e | analytic Psi_bar + particles %.3e"
        % (rms(Hc), float(np.abs(Hc[core]).max()), rms(Hcbar)))
    s_trunc = np.where(keep, s, 0.0)
    log("truncated-source consistency: max |Psi_table - (1 + G*s_trunc)| = %.3e"
        % float(np.abs(psi - (1.0 + G.potential(s_trunc))).max()))
    # multipole moments of the source
    stot = s.sum() * grid.dV
    qxx = float(np.sum(s * Xc * Xc) * grid.dV / stot)
    qyy = float(np.sum(s * Yc * Yc) * grid.dV / stot)
    qzz = float(np.sum(s * Zc * Zc) * grid.dV / stot)
    dip = [float(np.sum(s * C) * grid.dV / stot) for C in (Xc, Yc, Zc)]
    log("source moments: M_ADM=%.10f dipole=%s  <x^2>,<y^2>,<z^2>=%.5f %.5f %.5f"
        % (stot, dip, qxx, qyy, qzz))

    # write the binary file
    nz_ = np.nonzero(s_trunc.reshape(-1))[0]
    sw = (s.reshape(-1)[nz_] * grid.dV).astype(np.float64)
    kk, jj, ii = np.unravel_index(nz_, s.shape)
    sx = grid.centers(0)[ii]
    sy = grid.centers(1)[jj]
    sz = grid.centers(2)[kk]
    with open(args.out, "wb") as f:
        f.write(MAGIC)
        np.array([VERSION, args.N, args.fold, nx[0], nx[1], nx[2], len(sw), args.seed],
                 dtype=np.int64).tofile(f)
        np.array([sph.M, sph.e, sph.b, sph.a, mass.sum(), float(mass[0]), stot,
                  psi[nx[2] // 2, nx[1] // 2, nx[0] // 2],
                  grid.xmin[0], grid.xmin[1], grid.xmin[2], grid.h, CUBE_SELF],
                 dtype=np.float64).tofile(f)
        pos[:, 0].astype(np.float64).tofile(f)
        pos[:, 1].astype(np.float64).tofile(f)
        pos[:, 2].astype(np.float64).tofile(f)
        mass.astype(np.float64).tofile(f)
        psi.astype(np.float64).tofile(f)          # [z][y][x], x fastest
        sx.astype(np.float64).tofile(f)
        sy.astype(np.float64).tofile(f)
        sz.astype(np.float64).tofile(f)
        sw.tofile(f)
    log("wrote %s (%.1f MB) in %.1fs"
        % (args.out,
           np.float64(0).nbytes
           and (8 * (4 * args.N + psi.size + 4 * len(sw)) + 200) / 1e6,
           time.time() - t0))

    result.update(dict(M=sph.M, e=sph.e, b=sph.b, a=sph.a, beta0=sph.beta0, M0_pdf=sph.M0,
                       N=args.N, fold=args.fold, seed=args.seed, m_p=float(mass[0]),
                       sum_m=float(mass.sum()),
                       M_ADM=stot,
                       psi_center_table=float(psi[nx[2] // 2, nx[1] // 2, nx[0] // 2]),
                       psi_center_analytic=float(sph.psi_bar(0, 0, 0)),
                       psi_min=float(psi.min()), psi_max=float(psi.max()),
                       com=com.tolist(), second_moments=q.tolist(), source_dipole=dip,
                       source_second_moments=[qxx, qyy, qzz],
                       H7_rms_solved=rms(Hres), H7_rms_analytic=rms(Hbar),
                       Hcode_rms_final=rms(Hc),
                       Hcode_max_final=float(np.abs(Hc[core]).max()),
                       Hcode_rms_analytic=rms(Hcbar), defect_history=dhist,
                       picard_history=hist, nsrc=int(len(sw)), grid_n=nx.tolist(),
                       grid_box=box.tolist(), dx=grid.h,
                       analytic_checks=dict(lap_in_over_4pirho=lin, lap_out_scaled=lout,
                                            surface_jump=cont, far_ratio=far)))
    with open(args.json or (args.out + ".json"), "w") as f:
        json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
