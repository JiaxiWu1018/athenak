"""Shared readers/reducers for the ST-migration campaign.

Everything here is deliberately dependency-light (numpy only, scipy optional) so the
same file runs on Perseus and on the AMD login/compute nodes (AMD python3 is 3.9 with
numpy but no scipy).
"""
from __future__ import annotations

import os
import re
import struct

import numpy as np


# ----------------------------------------------------------------------------------
# t=0 particle dump written by the pgen (problem/cluster_t0_dump)
# ----------------------------------------------------------------------------------
T0_FIELDS = ("tag", "reflected", "x", "y", "z", "ux", "uy", "uz")


def read_t0_dump(path):
    """Read a `cluster_t0_dump` file.

    Layout: one ASCII header line ('STMIGP01 key=value ...'), then npart records of
    `nfield` little-endian float64.  Returns (meta_dict, dict_of_arrays).
    """
    with open(path, "rb") as fh:
        header = b""
        while True:
            ch = fh.read(1)
            if not ch or ch == b"\n":
                break
            header += ch
        head = header.decode("ascii")
        if not head.startswith("STMIGP01"):
            raise ValueError(f"{path}: not an STMIGP01 dump (header={head[:40]!r})")
        meta = {}
        for tok in head.split()[1:]:
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            try:
                meta[k] = int(v) if re.fullmatch(r"[-+]?\d+", v) else float(v)
            except ValueError:
                meta[k] = v
        npart = int(meta["npart"])
        nfield = int(meta["nfield"])
        raw = np.fromfile(fh, dtype="<f8", count=npart*nfield)
    if raw.size != npart*nfield:
        raise ValueError(f"{path}: short read {raw.size} != {npart*nfield}")
    raw = raw.reshape(npart, nfield)
    fields = meta.get("fields", ",".join(T0_FIELDS)).split(",")
    data = {name: raw[:, i] for i, name in enumerate(fields)}
    data["tag"] = data["tag"].astype(np.int64)
    data["reflected"] = data["reflected"].astype(np.int8)
    return meta, data


# ----------------------------------------------------------------------------------
# equilibrium profile (an exact port of the pgen ODE; see st_equilibrium_sequence.py)
# ----------------------------------------------------------------------------------
def cluster_profile(yc, dx=1.0e-4):
    """Fixed-step RK4 port of ConstructProfile() -> dict of profile arrays + scalars."""
    import math
    zeta = (1.0 - yc)/3.0
    eps = min(1.0e-6, 0.01*dx)

    def rhs(x, s):
        y = min(max(s[0], yc), 1.0)
        r = max(1.0/y - 1.0, 0.0)/(1.0/yc - 1.0)
        rho = (y/yc)**(-1.5)*math.sqrt(r)
        rho0 = math.sqrt(yc)*(y/yc)**(-1.0)*math.sqrt(r)
        pr = (y/yc)**(-0.5)*r**1.5
        comp = 1.0 - 2.0*zeta*s[1]/x
        rf = 1.0/math.sqrt(comp)
        return (2.0*zeta*y*(s[1] + zeta*x**3*pr)/(x*x*comp), x*x*rho,
                x*x*rho0*rf, (rf - 1.0)/x)

    xs = [0.0, eps]
    ys = [yc]
    vs = [0.0]
    v0s = [0.0]
    hs = [0.0]
    st = (yc + zeta*yc*(1.0/3.0 + zeta)*eps*eps, eps**3/3.0, eps**3/3.0, zeta*eps*eps/6.0)
    ys.append(st[0]); vs.append(st[1]); v0s.append(st[2]); hs.append(st[3])
    x = eps
    hit = False
    for _ in range(1000000):
        xo, old = x, st
        k1 = rhs(xo, old)
        s2 = tuple(old[i] + 0.5*dx*k1[i] for i in range(4)); k2 = rhs(xo + 0.5*dx, s2)
        s3 = tuple(old[i] + 0.5*dx*k2[i] for i in range(4)); k3 = rhs(xo + 0.5*dx, s3)
        s4 = tuple(old[i] + dx*k3[i] for i in range(4));     k4 = rhs(xo + dx, s4)
        st = tuple(old[i] + dx*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6.0 for i in range(4))
        x = xo + dx
        if st[0] >= 1.0:
            f = (1.0 - old[0])/(st[0] - old[0])
            x = xo + f*dx
            st = (1.0, old[1] + f*(st[1] - old[1]), old[2] + f*(st[2] - old[2]),
                  old[3] + f*(st[3] - old[3]))
            hit = True
        xs.append(x); ys.append(st[0]); vs.append(st[1]); v0s.append(st[2]); hs.append(st[3])
        if hit:
            break
    if not hit:
        raise RuntimeError("ODE did not reach y=1")

    x = np.array(xs); y = np.array(ys); v = np.array(vs)
    v0 = np.array(v0s); h = np.array(hs)
    R_over_M = x[-1]/(v[-1]*zeta)
    Riso_over_M = 0.5*(R_over_M - 1.0 + np.sqrt(R_over_M*(R_over_M - 2.0)))
    alpha_s = np.sqrt(1.0 - 2.0/R_over_M)
    xfrac = np.where(np.arange(x.size) == 0, 0.0, x/x[-1])
    riso_over_m = Riso_over_M*xfrac*np.exp(h - h[-1])
    conformal_a = (R_over_M/Riso_over_M)*np.exp(h[-1] - h)
    return dict(x=x, y=y, v=v, v0=v0, h=h,
                riso_over_m=riso_over_m, conformal_a=conformal_a,
                lapse=alpha_s*np.sqrt(y), r_over_m=x/(v[-1]*zeta),
                xs=x[-1], vs=v[-1], v0s=v0[-1], hs=h[-1],
                R_over_M=R_over_M, Riso_over_M=Riso_over_M,
                M0_over_M=v0[-1]/v[-1], alpha_s=alpha_s,
                alpha_c=alpha_s*np.sqrt(yc), zc=1.0/(alpha_s*np.sqrt(yc)) - 1.0,
                binding_fraction=1.0 - v[-1]/v0[-1], yc=yc, zeta=zeta)


def profile_interp(prof, riso, key):
    """Interpolate a profile column at isotropic radius riso (in units of M)."""
    return np.interp(riso, prof["riso_over_m"], prof[key],
                     left=prof[key][0], right=prof[key][-1])


# ----------------------------------------------------------------------------------
# t=0 derived per-particle quantities
# ----------------------------------------------------------------------------------
def t0_derived(meta, data, prof=None):
    """Radius, radial direction, |u|, mu, speed, -u_t, orthonormal momentum."""
    yc = float(meta["yc"])
    mass = float(meta.get("mass", 1.0))
    if prof is None:
        prof = cluster_profile(yc, dx=float(meta.get("profile_dx", 1.0e-4)))
    cx, cy, cz = (0.0, 0.0, 0.0)
    if isinstance(meta.get("center"), str):
        cx, cy, cz = (float(t) for t in meta["center"].split(","))
    rx = data["x"] - cx
    ry = data["y"] - cy
    rz = data["z"] - cz
    r = np.sqrt(rx*rx + ry*ry + rz*rz)
    inv = np.where(r > 0, 1.0/np.where(r > 0, r, 1.0), 0.0)
    nx, ny, nz = rx*inv, ry*inv, rz*inv
    ux, uy, uz = data["ux"], data["uy"], data["uz"]
    umag = np.sqrt(ux*ux + uy*uy + uz*uz)
    udotn = ux*nx + uy*ny + uz*nz
    mu = np.where(umag > 0, udotn/np.where(umag > 0, umag, 1.0), 0.0)

    # equilibrium quantities at the particle's radius (r is in code units = M*riso/M)
    riso_over_m = r/mass
    ylocal = profile_interp(prof, riso_over_m, "y")
    A = profile_interp(prof, riso_over_m, "conformal_a")
    alpha = profile_interp(prof, riso_over_m, "lapse")
    # exterior continuation, in case a particle sits outside the surface
    out = riso_over_m > prof["riso_over_m"][-1]
    if np.any(out):
        q = 0.5/riso_over_m[out]
        A[out] = (1.0 + q)**2
        alpha[out] = (1.0 - q)/(1.0 + q)
        ylocal[out] = 1.0

    equilibrium_speed = np.sqrt(np.maximum(1.0 - ylocal, 0.0))
    u2 = (umag/A)**2                                     # gamma^{ij} u_i u_j
    lorentz = np.sqrt(1.0 + u2)
    speed = np.sqrt(u2/(1.0 + u2))                       # actual post-map |v_hat|
    minus_ut = alpha*lorentz                             # -u_t (beta = 0)
    vr = (udotn/A)/lorentz
    return dict(r=r, riso_over_m=riso_over_m, nx=nx, ny=ny, nz=nz, umag=umag,
                udotn=udotn, mu=mu, ylocal=ylocal, A=A, alpha=alpha,
                equilibrium_speed=equilibrium_speed, speed=speed, lorentz=lorentz,
                vr=vr, minus_ut=minus_ut,
                phat_x=ux/A, phat_y=uy/A, phat_z=uz/A, phat_r=udotn/A, prof=prof)


# ----------------------------------------------------------------------------------
# .hst reader.  Column order is fixed by src/outputs/history.cpp LoadZ4cHistoryData.
# ----------------------------------------------------------------------------------
HST_COLS = ("time", "dt", "C-norm2", "H-norm2", "M-norm2", "Z-norm2",
            "Mx-norm2", "My-norm2", "Mz-norm2", "Theta-norm2", "Volume")


def read_hst(path):
    a = np.loadtxt(path, comments="#", ndmin=2)
    n = min(a.shape[1], len(HST_COLS))
    return {HST_COLS[i]: a[:, i] for i in range(n)}


# ----------------------------------------------------------------------------------
# AthenaK binary (.bin) reader -- thin wrapper around the in-tree vis/python reader, so
# the campaign never maintains a second copy of the format.
# ----------------------------------------------------------------------------------
_BIN_CONVERT = None


def _bin_convert(vis_python=None):
    """Load the in-tree vis/python/bin_convert.py, stubbing h5py.

    bin_convert imports h5py at module scope for its athdf WRITER, which we never use;
    the AMD python has no h5py, so the import is stubbed out the way
    reconstruction_ST/04b_.../scripts/st_lib.py does it.
    """
    global _BIN_CONVERT
    if _BIN_CONVERT is None:
        import types
        import sys as _sys
        cands = [vis_python, os.environ.get("ATHENAK_VIS_PYTHON", ""),
                 "/data/jiaxiwu/NRPIC/ST_migration/code/vis/python",
                 "/work1/eliasmost/jiaxiwu/st_migration_s04_20260901/athenak/vis/python"]
        for c in cands:
            f = os.path.join(c, "bin_convert.py") if c else ""
            if f and os.path.exists(f):
                _sys.modules.setdefault("h5py", types.ModuleType("h5py"))
                mod = types.ModuleType("bin_convert_intree")
                with open(f) as fh:
                    exec(compile(fh.read(), f, "exec"), mod.__dict__)  # noqa: S102
                _BIN_CONVERT = mod
                break
        else:
            raise RuntimeError("bin_convert.py not found; set ATHENAK_VIS_PYTHON")
    return _BIN_CONVERT


def read_binary(path, vis_python=None):
    """Read an AthenaK `file_type = bin` output via the in-tree vis/python reader."""
    return _bin_convert(vis_python).read_binary(path)


def bin_cell_coords(fd, mb):
    """Cell-centre coordinate arrays (x1, x2, x3) and cell sizes of MeshBlock `mb`.

    IMPORTANT: fd['mb_geometry'][m] is (x1min, x1max, x2min, x2max, x3min, x3max) --
    the BLOCK BOUNDS.  bin_convert.py's own docstring claims (x1i,x2i,x3i,dx1,dx2,dx3)
    and is stale; this was already diagnosed and documented by the previous campaign in
    reconstruction_ST/04b_ST_paper_hires_suite_20260821/scripts/st_lib.py.  Cell size is
    therefore (x1max-x1min)/nx1_out.
    """
    g = fd["mb_geometry"][mb]
    i0, i1, j0, j1, k0, k1 = fd["mb_index"][mb]
    n1, n2, n3 = (i1 - i0 + 1), (j1 - j0 + 1), (k1 - k0 + 1)
    d1 = (g[1] - g[0])/n1
    d2 = (g[3] - g[2])/n2
    d3 = (g[5] - g[4])/n3
    x1 = g[0] + (np.arange(n1) + 0.5)*d1
    x2 = g[2] + (np.arange(n2) + 0.5)*d2
    x3 = g[4] + (np.arange(n3) + 0.5)*d3
    return x1, x2, x3, float(d1), float(d2), float(d3)


def bin_flatten(fd, var, slice_x3=None):
    """Flatten one variable into (r, value, weight) arrays.

    Works for a full 3-D dump and for a 2-D slice alike.  For a slice the third
    direction contributes a single cell whose stored x3 is the block's lower bound plus
    half a cell, which is NOT the slice plane; pass `slice_x3` (usually 0.0) to override
    it so that r is the in-plane radius.  `weight` is the cell volume (3-D) or the cell
    area times one cell width (2-D); only like-for-like ratios are meaningful in 2-D.
    """
    data = fd["mb_data"][var]
    rs, vs, ws = [], [], []
    for mb in range(fd["n_mbs"]):
        x1, x2, x3, d1, d2, d3 = bin_cell_coords(fd, mb)
        if slice_x3 is not None and x3.size == 1:
            x3 = np.array([slice_x3])
        X3, X2, X1 = np.meshgrid(x3, x2, x1, indexing="ij")
        rs.append(np.sqrt(X1*X1 + X2*X2 + X3*X3).ravel())
        vs.append(np.asarray(data[mb], dtype=np.float64).ravel())
        ws.append(np.full(rs[-1].size, abs(d1*d2*d3)))
    return np.concatenate(rs), np.concatenate(vs), np.concatenate(ws)


def radial_bin(r, w, values=None, nbins=120, rmax=None):
    """Mass/volume-weighted radial binning. Returns (r_centres, sum_w, mean_value)."""
    rmax = rmax if rmax is not None else float(np.max(r))
    edges = np.linspace(0.0, rmax, nbins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, nbins - 1)
    sw = np.bincount(idx, weights=w, minlength=nbins)
    rc = 0.5*(edges[1:] + edges[:-1])
    if values is None:
        return rc, sw, None
    sv = np.bincount(idx, weights=w*values, minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mv = np.where(sw > 0, sv/np.where(sw > 0, sw, 1.0), np.nan)
    return rc, sw, mv


def mass_quantile_radii(r, m, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    """Lagrangian shell radii: radius enclosing the given fractions of the rest mass."""
    o = np.argsort(r)
    rs = r[o]
    cw = np.cumsum(m[o])
    cw = cw/cw[-1]
    return np.array([rs[np.searchsorted(cw, q)] if q < 1.0 else rs[-1] for q in qs])


# ----------------------------------------------------------------------------------
# Comparison against the equilibrium family: what a given y_c would look like through
# exactly the diagnostics we measure in the evolution.
# ----------------------------------------------------------------------------------
def equilibrium_cdf(prof):
    """Normalised enclosed-rest-mass CDF of Eq. (A18), as ConstructProfile builds it."""
    x, y = prof["x"], prof["y"]
    w = x*x/y*np.sqrt(np.maximum(1.0/y - 1.0, 0.0))
    cdf = np.zeros_like(x)
    cdf[1:] = np.cumsum(0.5*(w[:-1] + w[1:])*np.diff(x))
    cdf /= cdf[-1]
    cdf[-1] = 1.0
    return cdf


def equilibrium_observables(yc, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9, 0.99),
                            dx=1.0e-4, prof=None):
    """The macroscopic observables of the y_c equilibrium, in the SAME conventions the
    evolution diagnostics use: enclosed-rest-mass radii in isotropic radius / M, the
    central lapse, the surface radius, z_c and the binding fraction."""
    if prof is None:
        prof = cluster_profile(yc, dx=dx)
    cdf = equilibrium_cdf(prof)
    rq = np.interp(np.asarray(quantiles), cdf, prof["riso_over_m"])
    return dict(yc=prof["yc"], quantiles=tuple(quantiles),
                shell_radii=rq, alpha_c=prof["alpha_c"], alpha_s=prof["alpha_s"],
                Riso_over_M=prof["Riso_over_M"], R_over_M=prof["R_over_M"],
                zc=prof["zc"], M0_over_M=prof["M0_over_M"],
                binding_fraction=prof["binding_fraction"], prof=prof)


def equilibrium_family(yc_values, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9, 0.99),
                       dx=1.0e-3):
    """Tabulate `equilibrium_observables` over a range of y_c (coarser ODE step: the
    shell radii converge much faster than the surface location does)."""
    rows = []
    for yc in yc_values:
        rows.append(equilibrium_observables(float(yc), quantiles, dx=dx))
    return rows


def best_fit_yc(measured_radii, family, quantiles_used=(0, 1, 2, 3, 4)):
    """Which member of the equilibrium family best matches a measured set of
    enclosed-rest-mass radii, in a relative-error least-squares sense.

    Returns (yc_best, relative_rms_at_best, yc_array, relative_rms_array).
    """
    m = np.asarray(measured_radii, dtype=float)
    ycs = np.array([f["yc"] for f in family])
    err = np.empty(ycs.size)
    for i, f in enumerate(family):
        pred = np.asarray(f["shell_radii"], dtype=float)
        idx = np.asarray(quantiles_used, dtype=int)
        err[i] = np.sqrt(np.mean(((m[idx] - pred[idx])/pred[idx])**2))
    j = int(np.argmin(err))
    return float(ycs[j]), float(err[j]), ycs, err


def adm_mass_from_psi4(r, psi4, w=None, rmin=12.0, rmax=22.0):
    """Far-field ADM-mass proxy from the conformally flat isotropic form
    psi = 1 + M/(2 r)  =>  M = 2 r (psi4**0.25 - 1).

    Averaged over a shell rmin < r < rmax.  This is a diagnostic, not a rigorous ADM
    integral: it assumes the exterior stays close to conformally flat and it is affected
    by the outgoing gauge wave, so it should be read as a trend, not as a precise mass.
    """
    sel = (r > rmin) & (r < rmax) & np.isfinite(psi4) & (psi4 > 0)
    if not np.any(sel):
        return float("nan")
    m = 2.0*r[sel]*(np.power(psi4[sel], 0.25) - 1.0)
    ww = np.ones_like(m) if w is None else w[sel]
    return float(np.average(m, weights=ww))
