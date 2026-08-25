#!/usr/bin/env python3
"""Independent validation of AthenaK FastFlow apparent-horizon output.

WHY THIS EXISTS.  FastFlow's own summary file cannot be trusted as a record of finds.
`FastFlow::Write` (src/z4c/fastflow.cpp:379-400) prints `ah_prop[]` on EVERY in-window
cycle, but `ah_prop[]` is written only inside `if (ah_found)` in FastFlowLoop and
`ah_found` is reset false at the top of every call.  So the summary shows `nan` before
the first success and then a VERBATIM REPEAT of the last successful row on every
subsequent failure.  The ST session-04c campaign was misled by exactly this.
The shape file, by contrast, is opened only under `if (ah_found)` (fastflow.cpp:402-422),
so **one "# iter = ..., Time = ..." block in the shape file == one genuine converged
find**.  That is the find counter used here.

Second, convergence inside FastFlow is `|mass_prev - mass| < mass_tol` -- a fixed point of
the flow, NOT a test that the surface is a horizon.  The only blow-up guards are
`|hmean| < hmean_tol`, `meanradius < 0` and `mass < 1e-10`.  Session 04c showed a
converged, reported surface with irreducible mass 3.2 (total system mass 1.0), mean
coordinate radius 156.9 and mean expansion -19, accepted because the surface integrals
were accumulated over the 2 of 200 collocation points still on the mesh.  Every gate
below exists because of a specific way that failure can happen.

Columns of <basename>.horizon_summary_<n>.txt (fastflow.cpp:381-393):
  1 iter (the RK STAGE, constant 4 for rk4 -- NOT the cycle)   2 time
  3 mass (Christodoulou: sqrt(M_irr^2 + (S/2 M_irr)^2))        4-6 Sx Sy Sz   7 S
  8 area   9 hrms (= int H^2 dA / area, a mean square)   10 hmean (= int H dA,
  UN-normalised)   11 meanradius (= a0(0)/sqrt(4 pi), coordinate)   12 minradius
Shape file lines are the spectral coefficients [a0(0..lmax) | (ac,as) interleaved].
The surface CENTRE is not in either file; with use_puncture_<n> >= 0 it is the tracker
position, read here from <basename>.co_<n>.txt.

usage: analyze_ah.py <run_dir> [--dx-fine DX] [--rmax-gate R] [--mass-adm M]
                     [--expect-radius R] [--expect-mass M] [--label L]
"""
import argparse, glob, math, os, re, sys
import numpy as np

# ----------------------------------------------- AthenaK's OWN spherical harmonics
# Transcribed verbatim from src/utils/spherical_harm.hpp SphericalHarm() (Wigner-d form,
# Eq. II.7-II.8 of arXiv:0709.0093) rather than reimplemented from a textbook, because a
# normalisation or Condon-Shortley mismatch would silently corrupt every radius below.
def _fac(i):
    r = 1.0
    while i > 0:
        r *= i
        i -= 1
    return r


def spherical_harm(l, m, theta, phi):
    """(Re Y_lm, Im Y_lm) exactly as SphericalHarm() computes them, m >= 0."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    c, s = np.cos(theta/2.0), np.sin(theta/2.0)
    w = np.zeros_like(theta)
    for k in range(max(0, m), min(l + m, l) + 1):
        w = w + ((-1.0)**k * c**(2*l + m - 2*k) * s**(2*k - m)
                 / (_fac(l + m - k)*_fac(l - k)*_fac(k)*_fac(k - m)))
    w = w * math.sqrt((2*l + 1)/(4.0*math.pi)) * _fac(l) \
        * math.sqrt(_fac(l + m)) * math.sqrt(_fac(l - m))
    return w*np.cos(m*phi), w*np.sin(m*phi)


def lmindex(l, m, lmax):
    """src/utils/spherical_harm.hpp:143 -- l*(lmax+1) + m."""
    return l*(lmax + 1) + m


def parse_shape_line(vals, lmax):
    """Unpack the shape-file stream into (a0, ac, as).

    fastflow.cpp:411-420 writes, in this exact order:
        for l = 0..lmax:  a0[l],  then for m = 1..l:  ac[lmindex(l,m)], as[lmindex(l,m)]
    i.e. the m>0 coefficients are INTERLEAVED with the m=0 ones, not stored in blocks.
    """
    n = (lmax + 1)*(lmax + 1) + (lmax + 1) - (lmax + 1)   # = (lmax+1)^2 ... see check
    a0 = np.zeros(lmax + 1)
    ac = {}
    as_ = {}
    i = 0
    for l in range(lmax + 1):
        a0[l] = vals[i]; i += 1
        for m in range(1, l + 1):
            ac[(l, m)] = vals[i]; i += 1
            as_[(l, m)] = vals[i]; i += 1
    assert i == len(vals), f"consumed {i} of {len(vals)} shape coefficients"
    return a0, ac, as_


def shape_ncoef(lmax):
    """number of numbers on one shape line: (lmax+1) a0 plus 2 per (l,m>0) pair."""
    return (lmax + 1) + 2*sum(range(1, lmax + 1))


def surface_radius(vals, lmax, theta, phi):
    """R(theta,phi), mirroring FastFlow::RadiiFromSphericalHarmonics (fastflow.cpp:1011)
        R = sum_l a0[l] Y0[l] + sum_l sum_{m>0} (ac Yc + as Ys)
    with Yc = sqrt(2) Re Y_lm and Ys = sqrt(2) Im Y_lm (fastflow.cpp:1531-1532)."""
    a0, ac, as_ = parse_shape_line(vals, lmax)
    sqrt2 = math.sqrt(2.0)
    r = np.zeros_like(np.asarray(theta, dtype=float))
    for l in range(lmax + 1):
        yR, _ = spherical_harm(l, 0, theta, phi)
        r = r + a0[l]*yR
        for m in range(1, l + 1):
            yR, yI = spherical_harm(l, m, theta, phi)
            r = r + ac[(l, m)]*sqrt2*yR + as_[(l, m)]*sqrt2*yI
    return r


# ------------------------------------------------------------------------------- readers
def read_summary(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 12:
            continue
        rows.append([float(x) for x in p[:12]])
    return np.array(rows) if rows else np.zeros((0, 12))


def read_shape(path):
    """-> list of (iter, time, coef array).  One entry per GENUINE find."""
    out, hdr = [], None
    for line in open(path):
        line = line.strip()
        if line.startswith("#"):
            m = re.search(r"iter\s*=\s*(\d+),\s*Time\s*=\s*([-+0-9.eE]+)", line)
            hdr = (int(m.group(1)), float(m.group(2))) if m else None
        elif line and hdr is not None:
            out.append((hdr[0], hdr[1], np.array([float(x) for x in line.split()])))
            hdr = None
    return out


def read_tracker(path):
    rows = []
    for line in open(path):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 5:
            rows.append([float(p[1]), float(p[2]), float(p[3]), float(p[4])])
    return np.array(rows) if rows else np.zeros((0, 4))


def read_verbose(path):
    """-> (n_found_lines, list of failure reasons with counts, list of 'Found' times)"""
    txt = open(path, errors="ignore").read()
    found = len(re.findall(r"Found horizon", txt))
    reasons = {}
    for m in re.finditer(r"Failed, ([^\n]*)", txt):
        reasons[m.group(1).strip()] = reasons.get(m.group(1).strip(), 0) + 1
    return found, reasons


# ---------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--dx-fine", type=float, default=None,
                    help="finest grid spacing; minradius must exceed 2*dx_fine")
    ap.add_argument("--rmax-gate", type=float, default=None,
                    help="reject any surface whose max radius exceeds this (put it at "
                         "the innermost refined-region radius)")
    ap.add_argument("--mass-adm", type=float, default=0.73,
                    help="total ADM mass; a horizon cannot be heavier")
    ap.add_argument("--mass-tol", type=float, default=0.0,
                    help="fractional headroom on the MASS gate: it fires only when "
                         "m_irr > mass_adm*(1+mass_tol). The gate m_irr <= M_ADM is a "
                         "physical bound and is right for a collapse run, where "
                         "m_irr << M_ADM. It is DEGENERATE for a vacuum-puncture "
                         "calibration, where m_irr should EQUAL M_ADM, so any positive "
                         "discretisation error trips it -- that cost a wrong reading of "
                         "the ah_cal_* runs (REPORT.md 14b-33). Use ~0.05 for punctures.")
    ap.add_argument("--expect-radius", type=float, default=None)
    ap.add_argument("--expect-mass", type=float, default=None)
    ap.add_argument("--hmean-rel", type=float, default=0.20,
                    help="gate on |int H dA| / area")
    ap.add_argument("--persist", type=int, default=3,
                    help="require this many consecutive genuine finds")
    ap.add_argument("--label", default=None)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    rd = a.run_dir
    out = os.path.join(rd, "out")
    cand = [rd, out] + glob.glob(os.path.join(out, "*"))
    def find1(pat):
        for d in cand:
            g = sorted(glob.glob(os.path.join(d, pat)))
            if g:
                return g[0]
        return None

    fsum = find1("*.horizon_summary_0.txt")
    fshp = find1("*.horizon_shape_0.txt")
    fvrb = find1("*.horizon_verbose_0.txt")
    ftrk = find1("*.co_0.txt")

    figs = a.outdir or os.path.join(rd, "analysis")
    os.makedirs(figs, exist_ok=True)
    rpt = open(os.path.join(figs, "ah_report.txt"), "w")
    def emit(s=""):
        print(s); rpt.write(s + "\n")

    lab = a.label or os.path.basename(os.path.normpath(rd))
    emit("=" * 96)
    emit(f"APPARENT-HORIZON VALIDATION  --  {lab}")
    emit("=" * 96)
    emit(f"summary file : {fsum}")
    emit(f"shape file   : {fshp}")
    emit(f"verbose file : {fvrb}")
    emit(f"tracker file : {ftrk}")
    emit()

    if fsum is None:
        emit("NO FASTFLOW OUTPUT AT ALL -- the finder never ran (check <fastflow> "
             "num_horizons and start_time_0).")
        emit("VERDICT: NO AH DATA")
        return 0

    S = read_summary(fsum)
    nrow = S.shape[0]
    uniq = len({tuple(r[1:]) for r in S})
    nnan = int(np.sum(~np.isfinite(S[:, 2]))) if nrow else 0
    emit(f"summary rows                     : {nrow}")
    emit(f"  distinct (time-independent) rows: {uniq}   all-nan rows: {nnan}")
    emit("  NOTE: the row count is NOT the find count -- see the module docstring.")

    shapes = read_shape(fshp) if fshp and os.path.exists(fshp) else []
    emit(f"GENUINE converged finds (shape-file blocks): {len(shapes)}")
    if fvrb and os.path.exists(fvrb):
        nf, reasons = read_verbose(fvrb)
        emit(f"verbose 'Found horizon' lines    : {nf}"
             + ("  [consistent]" if nf == len(shapes) else "  [MISMATCH]"))
        if reasons:
            emit("  flow failure reasons: "
                 + ", ".join(f"{k} x{v}" for k, v in sorted(reasons.items())))
    trk = read_tracker(ftrk) if ftrk and os.path.exists(ftrk) else np.zeros((0, 4))
    if len(trk):
        emit(f"tracker rows {len(trk)}: t=[{trk[0,0]:.4f},{trk[-1,0]:.4f}] "
             f"pos0=({trk[0,1]:+.5f},{trk[0,2]:+.5f},{trk[0,3]:+.5f}) "
             f"posN=({trk[-1,1]:+.5f},{trk[-1,2]:+.5f},{trk[-1,3]:+.5f}) "
             f"drift={np.linalg.norm(trk[-1,1:]-trk[0,1:]):.5f}")
    emit()

    if not shapes:
        emit("No genuine find in the whole run.")
        emit("VERDICT: NO APPARENT HORIZON DETECTED")
        rpt.close()
        return 0

    # angular grid for the on-region test: dense enough to catch a lobe
    nth, nph = 41, 80
    th = np.linspace(1e-6, math.pi - 1e-6, nth)
    ph = np.linspace(0.0, 2*math.pi, nph, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")

    emit(f"{'t':>10s} {'M_chr':>11s} {'M_irr':>11s} {'area':>11s} "
         f"{'r_mean':>10s} {'r_min':>10s} {'r_max_surf':>10s} "
         f"{'|hmean|/A':>10s} {'S':>10s}  gates")
    rows = []
    for it, t, coef in shapes:
        # nearest summary row at this time
        if nrow:
            k = int(np.argmin(np.abs(S[:, 1] - t)))
            _, _, mchr, Sx, Sy, Sz, Stot, area, hrms, hmean, rmean, rmin = S[k]
        else:
            mchr = area = hrms = hmean = rmean = rmin = Stot = np.nan
        lmax = None
        # the shape line length is (lmax+1) + 2*sum_{l} l = (lmax+1) + lmax(lmax+1)
        n = len(coef)
        for L in range(0, 33):
            if shape_ncoef(L) == n:
                lmax = L
                break
        if lmax is None:
            emit(f"  !! cannot infer lmax from {n} coefficients at t={t}")
            continue
        R = surface_radius(coef, lmax, TH, PH)
        rmax_s, rmin_s = float(np.nanmax(R)), float(np.nanmin(R))
        m_irr = math.sqrt(area/(16.0*math.pi)) if area > 0 else float("nan")
        hrel = abs(hmean)/area if area > 0 else float("inf")

        g = []
        if a.dx_fine is not None and not (rmin_s > 2.0*a.dx_fine):
            g.append(f"MINRAD<2dx({rmin_s:.4g}<{2*a.dx_fine:.4g})")
        if rmin_s <= 0.0:
            g.append("NEGATIVE-RADIUS")
        if a.rmax_gate is not None and rmax_s > a.rmax_gate:
            g.append(f"OFF-FINE-REGION({rmax_s:.4g}>{a.rmax_gate:.4g})")
        if not (hrel < a.hmean_rel):
            g.append(f"EXPANSION({hrel:.3g})")
        if not (0.0 < m_irr <= a.mass_adm*(1.0 + a.mass_tol)):
            g.append(f"MASS({m_irr:.4g})")
        if not np.isfinite([mchr, area, rmean, rmin]).all():
            g.append("NONFINITE")
        ok = not g
        rows.append(dict(t=t, m_chr=mchr, m_irr=m_irr, area=area, r_mean=rmean,
                         r_min=rmin, r_min_surf=rmin_s, r_max_surf=rmax_s,
                         hrel=hrel, S=Stot, ok=ok, gates=g, lmax=lmax))
        emit(f"{t:10.4f} {mchr:11.6f} {m_irr:11.6f} {area:11.5e} "
             f"{rmean:10.6f} {rmin:10.6f} {rmax_s:10.6f} "
             f"{hrel:10.4f} {Stot:10.3e}  {'PASS' if ok else '/'.join(g)}")

    emit()
    good = [r for r in rows if r["ok"]]
    emit(f"finds passing every gate: {len(good)} of {len(rows)}")

    # persistence: longest run of consecutive passing finds
    best, cur = 0, 0
    first_persistent = None
    for r in rows:
        if r["ok"]:
            cur += 1
            if cur >= a.persist and first_persistent is None:
                first_persistent = rows[rows.index(r) - a.persist + 1]["t"]
            best = max(best, cur)
        else:
            cur = 0
    emit(f"longest run of consecutive passing finds: {best} "
         f"(persistence requirement: {a.persist})")

    if best >= a.persist and good:
        g0 = good[0]
        emit()
        emit("VERDICT: APPARENT HORIZON DETECTED AND VALIDATED")
        emit(f"  first validated find at t = {g0['t']:.5f}"
             + (f" (first of a persistent run at t = {first_persistent:.5f})"
                if first_persistent is not None else ""))
        emit(f"  irreducible mass  M_irr = {g0['m_irr']:.6f}"
             f"   Christodoulou M = {g0['m_chr']:.6f}")
        emit(f"  area              A     = {g0['area']:.6e}"
             f"   areal radius sqrt(A/4pi) = {math.sqrt(g0['area']/(4*math.pi)):.6f}")
        emit(f"  coordinate radius r     = [{g0['r_min_surf']:.6f}, "
             f"{g0['r_max_surf']:.6f}], mean {g0['r_mean']:.6f}")
        if len(trk):
            k = int(np.argmin(np.abs(trk[:, 0] - g0["t"])))
            emit(f"  centre (tracker at t={trk[k,0]:.4f}) = "
                 f"({trk[k,1]:+.6f}, {trk[k,2]:+.6f}, {trk[k,3]:+.6f})")
        if a.dx_fine:
            emit(f"  cells across the horizon radius: "
                 f"{g0['r_min_surf']/a.dx_fine:.2f} (min) .. "
                 f"{g0['r_max_surf']/a.dx_fine:.2f} (max)")
        mm = np.array([r["m_irr"] for r in good])
        emit(f"  M_irr over all validated finds: min {mm.min():.6f} max {mm.max():.6f} "
             f"spread {(mm.max()-mm.min())/max(mm.mean(),1e-30):.3e}")
    else:
        emit()
        emit("VERDICT: NO VALIDATED APPARENT HORIZON")
        emit("  (converged surfaces exist but none survives the validity gates, or none "
             "persists)")

    if a.expect_radius is not None and good:
        g0 = good[0]
        emit()
        emit("--- comparison with the ANALYTIC answer ---")
        emit(f"  expected coordinate radius {a.expect_radius:.6f}; "
             f"got mean {g0['r_mean']:.6f}  "
             f"rel err {(g0['r_mean']/a.expect_radius - 1):+.3e}")
        if a.expect_mass is not None:
            emit(f"  expected irreducible mass  {a.expect_mass:.6f}; "
                 f"got {g0['m_irr']:.6f}  "
                 f"rel err {(g0['m_irr']/a.expect_mass - 1):+.3e}")
        emit(f"  surface non-sphericity (r_max-r_min)/r_mean = "
             f"{(g0['r_max_surf']-g0['r_min_surf'])/g0['r_mean']:.3e}")

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tt = np.array([r["t"] for r in rows])
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].plot(tt, [r["m_irr"] for r in rows], "o-", ms=3, label="M_irr")
        ax[0].plot(tt, [r["m_chr"] for r in rows], ".", ms=3, label="M_Christodoulou")
        ax[0].set_xlabel("t"); ax[0].set_ylabel("mass"); ax[0].legend(fontsize=7)
        ax[1].plot(tt, [r["r_mean"] for r in rows], "o-", ms=3, label="r_mean")
        ax[1].plot(tt, [r["r_min_surf"] for r in rows], "-", lw=1, label="r_min(surface)")
        ax[1].plot(tt, [r["r_max_surf"] for r in rows], "-", lw=1, label="r_max(surface)")
        if a.dx_fine:
            ax[1].axhline(2*a.dx_fine, color="r", ls=":", label="2 dx_fine")
        ax[1].set_xlabel("t"); ax[1].set_ylabel("coordinate radius")
        ax[1].legend(fontsize=7)
        ax[2].semilogy(tt, [max(r["hrel"], 1e-16) for r in rows], "o-", ms=3)
        ax[2].axhline(a.hmean_rel, color="r", ls=":")
        ax[2].set_xlabel("t"); ax[2].set_ylabel("|int H dA| / area")
        for x in ax:
            x.grid(alpha=.3)
        fig.suptitle(f"FastFlow validation: {lab}")
        fig.tight_layout()
        p = os.path.join(figs, "ah_validation.png")
        fig.savefig(p, dpi=130)
        emit(f"\nfigure: {p}")
    except Exception as e:                                    # noqa: BLE001
        emit(f"(figure skipped: {e})")

    rpt.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
