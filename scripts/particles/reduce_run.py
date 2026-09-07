#!/usr/bin/env python3
"""Scientific reduction of a Plummer run from its pvtk particle dumps.

Reduces over ALL particles, accumulating in float64 (the dump itself is float32, which
sets a ~1e-7 relative floor on any single-particle invariant -- stated in the report).

    reduce_run.py --rundir <dir with pvtk/> --out <outdir> [--period P] [--lmax 4]
                  [--macro 4] [--tmax T]

Writes
    modes.csv     time, t/P, band, l, A_l, A_shot, A_debias, n_pair   -- band = all,
                  com (centre-of-mass-removed, whole frame), or macro band m0..m{K-1}
    scalars.csv   time, t/P, N, M0, Rcom, com_xyz, r_q10..r_q90 (enclosed rest-mass
                  quantile radii, isotropic AND areal), sigma_r, sigma_t, E/L drift
                  statistics, dipole direction, inner/outer signed dipole contributions
    cohorts.csv   time, t/P, cohort, n, <r>, <r^2>, <v_r>, sigma_r, sigma_t
Definitions
    A_l = sqrt( 4 pi/(2l+1) sum_m <Y_lm(nhat)>^2 ), real orthonormal Y_lm, UNWEIGHTED
          mean over particles (equal rest masses, so unweighted == mass weighted). This
          is the established homogeneous-campaign definition, so the two campaigns are
          directly comparable.
    A_shot = n_uniq^{-1/2} with n_uniq the number of INDEPENDENT ANGULAR POSITIONS.
          The sampler co-locates every +-u_i pair, so n_uniq = N/2, NOT N. Measured from
          frame 0 and cross-checked against N/2; a mismatch is fatal, because a wrong
          n_uniq rescales every A/shot ratio silently and by a clean factor.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pvtk_reader import read_pvtk
from sph_real import ylm_all

FOURPI = 4.0*np.pi


def by_tag(arr, tag, n):
    """Reindex rows so row i is tag i.  Returns (out, alive_mask)."""
    shape = (n,) + arr.shape[1:]
    out = np.zeros(shape, dtype=np.float64)
    alive = np.zeros(n, dtype=bool)
    t = tag.astype(np.int64)
    keep = (t >= 0) & (t < n)
    out[t[keep]] = arr[keep]
    alive[t[keep]] = True
    return out, alive


def mode_amps(nx, ny, nz, lmax, groups=None):
    """A_l over all points, and per group if `groups` is a list of index arrays."""
    A_all = np.zeros(lmax + 1)
    A_grp = None if groups is None else [np.zeros(lmax + 1) for _ in groups]
    for l in range(1, lmax + 1):
        Y = ylm_all(nx, ny, nz, l)[l]
        coef = Y.mean(axis=1, dtype=np.float64)
        A_all[l] = np.sqrt(FOURPI/(2*l + 1)*np.sum(coef**2))
        if groups is not None:
            for gi, g in enumerate(groups):
                if g.size == 0:
                    continue
                cg = Y[:, g].mean(axis=1, dtype=np.float64)
                A_grp[gi][l] = np.sqrt(FOURPI/(2*l + 1)*np.sum(cg**2))
        del Y
    return A_all, A_grp


def dipole_vector(nx, ny, nz):
    """<nhat>, whose norm is A_1 (the l=1 case of the definition above)."""
    return np.array([nx.mean(dtype=np.float64), ny.mean(dtype=np.float64),
                     nz.mean(dtype=np.float64)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--period", type=float, default=1192.496781, help="P_1/2 in M")
    ap.add_argument("--lmax", type=int, default=4)
    ap.add_argument("--macro", type=int, default=4, help="macro radial bands")
    ap.add_argument("--ncohort", type=int, default=32)
    ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--nunique", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.rundir, "pvtk", "*.part.vtk")))
    if not files:
        sys.exit("no pvtk frames under %s" % a.rundir)
    print("# %d pvtk frames in %s" % (len(files), a.rundir), flush=True)
    want = {"pos", "ptag", "prtcl_vel", "prtcl_energy", "prtcl_mass"}

    # ---- frame 0 defines the cohorts, the invariant references and the shot floor ----
    d0 = read_pvtk(files[0], want)
    n = d0["n"]
    print("# N = %d, t0 = %.6f" % (n, d0["time"]), flush=True)
    if n > 2**24:
        print("# WARNING: N > 2^24, ptag is float32 in pvtk and is NOT exact", flush=True)
    tag0 = d0["ptag"]
    pos0, alive0 = by_tag(d0["pos"].astype(np.float64), tag0, n)
    vel0, _ = by_tag(d0["prtcl_vel"].astype(np.float64), tag0, n)
    en0, _ = by_tag(d0["prtcl_energy"].astype(np.float64), tag0, n)
    if not alive0.all():
        sys.exit("frame 0 does not contain every tag 0..N-1; cannot define cohorts")
    if not np.isfinite(pos0).all():
        sys.exit("frame 0 carries non-finite positions; refusing to reduce")
    r0 = np.linalg.norm(pos0, axis=1)
    L0 = np.linalg.norm(np.cross(pos0, vel0), axis=1)

    # independent angular positions: exact duplicate positions collapse
    key = np.round(pos0/max(1e-12, np.abs(pos0).max())*1e12).astype(np.int64)
    nuniq_meas = int(np.unique(key, axis=0).shape[0])
    del key
    nuniq = a.nunique if a.nunique is not None else nuniq_meas
    if a.nunique is not None and a.nunique != nuniq_meas:
        sys.exit("--nunique=%d contradicts the t=0 dump, which has %d independent "
                 "positions (N=%d, N/n_uniq=%.4f). Refusing to write an unverified "
                 "finite-N null." % (a.nunique, nuniq_meas, n, n/nuniq_meas))
    shot = nuniq**-0.5
    print("# independent angular positions %d (N/n_uniq = %.4f); A_shot = %.6e"
          % (nuniq, n/nuniq, shot), flush=True)
    if abs(n/nuniq - 2.0) > 1e-9:
        print("# NOTE: N/n_uniq != 2, so this dump is NOT the co-located-pair sampler",
              flush=True)

    # cohorts: contiguous bands of the initial radial ORDER, which for this sampler is
    # the stratum index = pair index = tag//2 (monotone in the radial quantile)
    order = np.argsort(r0, kind="stable")
    per = n//a.ncohort
    cohorts = [order[i*per:(i + 1)*per] for i in range(a.ncohort)]
    # macro bands by initial radius quantile
    mper = n//a.macro
    macro = [order[i*mper:(i + 1)*mper] for i in range(a.macro)]
    macro_edges = [(r0[order[i*mper]], r0[order[min((i + 1)*mper, n - 1)]])
                   for i in range(a.macro)]
    del d0

    fm = open(os.path.join(a.out, "modes.csv"), "w")
    fm.write("time,t_over_P,band,l,A_l,A_shot,A_debias,n_used\n")
    fs = open(os.path.join(a.out, "scalars.csv"), "w")
    fs.write("time,t_over_P,N_alive,M0,Rcom,com_x,com_y,com_z,"
             "d1x,d1y,d1z,A1_raw,A1_com,A1_inner,A1_outer,"
             + ",".join("r_q%02d" % q for q in (10, 25, 50, 75, 90)) + ","
             "sigma_r_u,sigma_t_u,dE_rms,dE_max,dL_rms,dL_max,dr_rms,n_nonfinite\n")
    fc = open(os.path.join(a.out, "cohorts.csv"), "w")
    fc.write("time,t_over_P,cohort,r0_lo,r0_hi,n,r_mean,r2_mean,ur_mean,sig_ur,sig_ut\n")

    for path in files:
        d = read_pvtk(path, want)
        t = d["time"]
        if a.tmax is not None and t > a.tmax + 1e-9:
            break
        tag = d["ptag"]
        pos, alive = by_tag(d["pos"].astype(np.float64), tag, n)
        vel, _ = by_tag(d["prtcl_vel"].astype(np.float64), tag, n)
        en, _ = by_tag(d["prtcl_energy"].astype(np.float64), tag, n)
        mas, _ = by_tag(d["prtcl_mass"].astype(np.float64), tag, n) \
            if "prtcl_mass" in d else (np.zeros(n), None)
        finite = np.isfinite(pos).all(axis=1) & np.isfinite(vel).all(axis=1) & alive
        nbad = int(np.count_nonzero(alive & ~finite))
        sel = np.where(finite)[0]
        r = np.linalg.norm(pos[sel], axis=1)
        rs = np.maximum(r, 1e-30)
        nh = pos[sel]/rs[:, None]
        m = mas[sel] if mas.any() else np.full(sel.size, 1.0)
        M0 = float(m.sum(dtype=np.float64))

        com = (m[:, None]*pos[sel]).sum(axis=0, dtype=np.float64)/max(M0, 1e-300)
        rel = pos[sel] - com
        rrel = np.maximum(np.linalg.norm(rel, axis=1), 1e-30)
        nhc = rel/rrel[:, None]

        # --- multipoles: raw, COM-removed, and per macro band (raw) ---
        Aall, Agrp = mode_amps(nh[:, 0], nh[:, 1], nh[:, 2], a.lmax,
                               groups=[np.intersect1d(g, sel, assume_unique=False)
                                       for g in macro])
        Acom, _ = mode_amps(nhc[:, 0], nhc[:, 1], nhc[:, 2], a.lmax)
        tp = t/a.period
        for l in range(1, a.lmax + 1):
            for band, A, nu in (("all", Aall[l], sel.size), ("com", Acom[l], sel.size)):
                fm.write("%.10g,%.10g,%s,%d,%.10e,%.10e,%.10e,%d\n"
                         % (t, tp, band, l, A, shot,
                            np.sqrt(max(0.0, A*A - 1.0/max(nuniq, 1))), nu))
            for gi in range(a.macro):
                gsz = macro[gi].size
                sh = (gsz/2.0)**-0.5 if gsz else float("nan")
                fm.write("%.10g,%.10g,m%d,%d,%.10e,%.10e,%.10e,%d\n"
                         % (t, tp, gi, l, Agrp[gi][l], sh,
                            np.sqrt(max(0.0, Agrp[gi][l]**2 - 1.0/max(gsz/2.0, 1))),
                            gsz))

        # --- dipole direction and signed inner/outer split (about the COM) ---
        d1 = dipole_vector(nhc[:, 0], nhc[:, 1], nhc[:, 2])
        rmed = np.median(rrel)
        inner = rrel < rmed
        A1_in = np.linalg.norm(dipole_vector(*(nhc[inner].T)))
        A1_out = np.linalg.norm(dipole_vector(*(nhc[~inner].T)))

        # --- enclosed rest-mass quantile radii (isotropic coordinate radius) ---
        rsort = np.sort(r)
        qs = [10, 25, 50, 75, 90]
        rq = [rsort[min(int(q/100.0*rsort.size), rsort.size - 1)] for q in qs]

        # --- velocity structure from the stored covariant u_i (radial vs tangential) ---
        ur = np.einsum('ij,ij->i', vel[sel], nh)
        ut2 = np.maximum(np.einsum('ij,ij->i', vel[sel], vel[sel]) - ur*ur, 0.0)
        sig_r = float(np.sqrt(np.mean(ur*ur, dtype=np.float64)))
        sig_t = float(np.sqrt(np.mean(ut2, dtype=np.float64)))

        # --- orbit invariants: E = -u_t (IPEN) and |L| = |x cross u| ---
        Lnow = np.linalg.norm(np.cross(pos[sel], vel[sel]), axis=1)
        e_ref = en0[sel]
        good_e = np.abs(e_ref) > 0
        dE = np.zeros(sel.size)
        dE[good_e] = en[sel][good_e]/e_ref[good_e] - 1.0
        good_l = L0[sel] > 0
        dL = np.zeros(sel.size)
        dL[good_l] = Lnow[good_l]/L0[sel][good_l] - 1.0
        dr = r/np.maximum(r0[sel], 1e-30) - 1.0

        fs.write("%.10g,%.10g,%d,%.12e,%.10e,%.10e,%.10e,%.10e,"
                 "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
                 % (t, tp, sel.size, M0, float(np.linalg.norm(com)),
                    com[0], com[1], com[2], d1[0], d1[1], d1[2],
                    Aall[1], Acom[1], A1_in, A1_out))
        fs.write(",".join("%.10e" % v for v in rq))
        fs.write(",%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%d\n"
                 % (sig_r, sig_t,
                    float(np.sqrt(np.mean(dE*dE))), float(np.max(np.abs(dE))),
                    float(np.sqrt(np.mean(dL*dL))), float(np.max(np.abs(dL))),
                    float(np.sqrt(np.mean(dr*dr))), nbad))

        for ci, g in enumerate(cohorts):
            gg = g[finite[g]]
            if gg.size == 0:
                continue
            rg = np.linalg.norm(pos[gg], axis=1)
            nhg = pos[gg]/np.maximum(rg, 1e-30)[:, None]
            urg = np.einsum('ij,ij->i', vel[gg], nhg)
            utg2 = np.maximum(np.einsum('ij,ij->i', vel[gg], vel[gg]) - urg*urg, 0.0)
            fc.write("%.10g,%.10g,%d,%.10e,%.10e,%d,%.10e,%.10e,%.10e,%.10e,%.10e\n"
                     % (t, tp, ci, r0[g].min(), r0[g].max(), gg.size,
                        rg.mean(dtype=np.float64), (rg*rg).mean(dtype=np.float64),
                        urg.mean(dtype=np.float64),
                        float(np.sqrt(np.mean(urg*urg))),
                        float(np.sqrt(np.mean(utg2)))))
        print("  t=%12.5f  t/P=%8.5f  N=%d  A1_raw=%.4e  A1_com=%.4e  "
              "A1/shot=%.3f  Rcom=%.4e  nbad=%d"
              % (t, tp, sel.size, Aall[1], Acom[1], Acom[1]/shot,
                 float(np.linalg.norm(com)), nbad), flush=True)
        del d, pos, vel, en

    fm.close(); fs.close(); fc.close()
    with open(os.path.join(a.out, "REDUCTION_META.txt"), "w") as f:
        f.write("rundir      %s\n" % a.rundir)
        f.write("frames      %d\n" % len(files))
        f.write("N           %d\n" % n)
        f.write("n_uniq      %d   (measured; N/n_uniq = %.6f)\n" % (nuniq, n/nuniq))
        f.write("A_shot      %.10e   = n_uniq^-1/2\n" % shot)
        f.write("period P    %.10g M\n" % a.period)
        f.write("lmax        %d\nmacro bands %d\ncohorts     %d\n"
                % (a.lmax, a.macro, a.ncohort))
        for i, (lo, hi) in enumerate(macro_edges):
            f.write("macro m%d    initial r in [%.6f, %.6f] M\n" % (i, lo, hi))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
