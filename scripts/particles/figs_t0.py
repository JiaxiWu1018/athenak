#!/usr/bin/env python3
"""t=0 validation figures for the Plummer campaign.

usage: figs_t0.py <shell_csv> <outdir> [field_csv]
Every panel compares the finite-N / gridded realization against the continuum
construction; the continuum is drawn in ink, never in a series colour.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plummer_style as ps
from plummer_style import SERIES, REFLINE, BAND, INK2, MUTED, label_end, mark_scales
from plummer_1d import PlummerModel
import matplotlib.pyplot as plt

ps.apply_style()
NPAIR = 1056768
N = 2*NPAIR


def _read_rows(path):
    """Read a pgen ledger CSV, de-duplicating restart overlap.

    A chained run re-appends rows for the interval the previous segment already
    covered, so the file is non-monotone in time. Keep the LAST occurrence of each
    (time, bin) key: that row was written by the state that actually continued.
    """
    hdr, rows = None, {}
    order = []
    for line in open(path):
        if line.startswith('#'):
            continue
        if hdr is None:
            hdr = line.strip().split(',')
            continue
        f = line.strip().split(',')
        if not f or f[0] == '':
            continue
        key = (f[0], f[2])
        if key not in rows:
            order.append(key)
        rows[key] = f
    arr = np.array([rows[k] for k in order], dtype=float)
    idx = np.lexsort((arr[:, 2], arr[:, 0]))
    return hdr, arr[idx]


def read_csv(path):
    hdr, arr = _read_rows(path)
    return {n: i for i, n in enumerate(hdr)}, arr


def main():
    shell_csv, outdir = sys.argv[1], sys.argv[2]
    field_csv = sys.argv[3] if len(sys.argv) > 3 else None
    os.makedirs(outdir, exist_ok=True)
    P = PlummerModel(npanel=20000, ngl=20)
    mu = P.M0/N

    c, A = read_csv(shell_csv)
    t0 = A[A[:, c['time']] == A[:, c['time']].min()]
    rlo, rhi = t0[:, c['r_lo']], t0[:, c['r_hi']]
    rc = np.sqrt(rlo*rhi)
    cnt, mass = t0[:, c['count']], t0[:, c['mass']]
    npair_bin = cnt/2.0
    ok = cnt > 200

    # ---------------- F1: rest-mass radial profile vs the continuum -------------
    # dM0/dr from the shells, compared with 4 pi r^2 eps B/W.  A count profile is a
    # DIFFERENT quantity and is not what is plotted here.
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    dm_dr = mass/(rhi - rlo)
    rr = np.geomspace(0.2, 400.0, 3000)
    ax.plot(rr, P.dM0dr(rr), color=REFLINE, lw=2.0, zorder=3,
            label=r"continuum  $4\pi r^2\epsilon B/W$")
    ax.plot(rc[ok], dm_dr[ok], 'o', color=SERIES[0], ms=4.5, zorder=4,
            label=r"particles, 48 shells")
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"areal radius $r\ [M]$")
    ax.set_ylabel(r"$\mathrm{d}M_0/\mathrm{d}r\ [M^{-1}\!\cdot\!M]$")
    ax.set_title("Rest-mass radial profile at $t=0$: particles vs the continuum measure")
    mark_scales(ax, seams=False, which=("b", "r_half", "r_t"))
    ax.legend(loc="lower left")
    label_end(ax, rc[ok][-1], dm_dr[ok][-1], "particles", SERIES[0])
    fig.savefig(os.path.join(outdir, "f1_restmass_profile_t0.png"))
    plt.close(fig)

    # ---------------- F2: shell-mass residual against the Poisson floor ----------
    Mc = P.M0*(P.F0_exact(np.minimum(rhi, 400.0)) - P.F0_exact(np.minimum(rlo, 400.0)))
    rel = np.where(Mc > 0, mass/np.maximum(Mc, 1e-300) - 1.0, np.nan)
    pois = 1.0/np.sqrt(np.maximum(npair_bin, 1))
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.fill_between(rc[ok], -pois[ok], pois[ok], color=BAND, zorder=1,
                    label=r"$\pm$ Poisson $n_{\rm pair}^{-1/2}$")
    ax.axhline(0.0, color=REFLINE, lw=1.2, zorder=2)
    ax.plot(rc[ok], rel[ok], 'o-', color=SERIES[0], ms=4.0, lw=1.4, zorder=4,
            label="measured shell mass")
    ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=1e-5)
    ax.set_xlabel(r"areal radius $r\ [M]$")
    ax.set_ylabel(r"$M_{\rm shell}/M_{\rm continuum}-1$")
    ax.set_title("Stratification beats Poisson: shell rest mass is exact to $\\sim\\!10^{-5}$")
    mark_scales(ax, seams=False, which=("b", "r_half", "r_t"))
    ax.legend(loc="upper left")
    label_end(ax, rc[ok][-1], rel[ok][-1], "measured", SERIES[0])
    fig.savefig(os.path.join(outdir, "f2_shellmass_residual_t0.png"))
    plt.close(fig)

    # ---------------- F3: A_l per band, normalised by each band's own shot -------
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    sel = npair_bin > 400
    for li, l in enumerate((1, 2, 3, 4)):
        s = np.zeros(len(rc))
        for m in range(-l, l + 1):
            q = l*l + (l + m)
            s += (t0[:, c['c%d' % q]]/np.maximum(mass, 1e-300))**2
        Al = np.sqrt(4.0*np.pi/(2.0*l + 1.0)*s)
        ratio = Al/np.maximum(npair_bin**-0.5, 1e-300)
        ax.plot(rc[sel], ratio[sel], 'o-', color=SERIES[li], ms=3.6, lw=1.4,
                label=r"$\ell=%d$" % l)
        label_end(ax, rc[sel][-1], ratio[sel][-1], r"$\ell=%d$" % l, SERIES[li])
    ax.axhline(1.0, color=REFLINE, lw=1.4, zorder=2)
    ax.annotate("shot floor", xy=(rc[sel][0], 1.0), xytext=(2, 5),
                textcoords="offset points", color=INK2, fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel(r"areal radius $r\ [M]$")
    ax.set_ylabel(r"$A_\ell\,/\,n_{\rm pair}^{-1/2}$ in the band")
    ax.set_title("Initial angular structure sits at the sampler's own noise floor")
    mark_scales(ax, seams=False, which=("b", "r_half", "r_t"))
    ax.legend(loc="upper left", ncol=4)
    fig.savefig(os.path.join(outdir, "f3_Al_bands_t0.png"))
    plt.close(fig)

    # ---------------- F4: velocity dispersions vs the continuum -----------------
    sr = np.sqrt(np.maximum(t0[:, c['m_vr2']]/np.maximum(mass, 1e-300), 0.0))
    st = np.sqrt(np.maximum(t0[:, c['m_vt2']]/np.maximum(mass, 1e-300), 0.0))
    rr = np.geomspace(0.3, 399.9, 3000)
    al = np.exp(P.Phi_exact(rr)); psi = np.exp(-0.5*P.j_exact(rr))
    vt_ref = al*psi**-2*np.sqrt(P.vc2(rr))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(rr, vt_ref, color=REFLINE, lw=2.0, zorder=3,
            label=r"continuum  $\alpha\psi^{-2}v_c$")
    ax.plot(rc[ok], st[ok], 'o', color=SERIES[0], ms=4.2, zorder=4,
            label=r"tangential  $\sigma_t$")
    ax.plot(rc[ok], np.maximum(sr[ok], 1e-20), 's', color=SERIES[1], ms=3.6, zorder=4,
            label=r"radial  $\sigma_r$")
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_ylim(1e-19, 0.3)
    ax.set_xlabel(r"areal radius $r\ [M]$")
    ax.set_ylabel(r"coordinate velocity dispersion")
    ax.set_title(r"Purely tangential start: $\sigma_r$ is zero to roundoff")
    mark_scales(ax, seams=False, which=("b", "r_half", "r_t"))
    ax.legend(loc="lower left")
    label_end(ax, rc[ok][-1], st[ok][-1], r"$\sigma_t$", SERIES[0])
    label_end(ax, rc[ok][-1], max(sr[ok][-1], 1e-19), r"$\sigma_r$", SERIES[1])
    fig.savefig(os.path.join(outdir, "f4_dispersions_t0.png"))
    plt.close(fig)

    # ---------------- F5/F6: constraint + field profile, if available -----------
    if field_csv and os.path.exists(field_csv):
        cf, F = read_csv(field_csv)
        f0 = F[F[:, cf['time']] == F[:, cf['time']].min()]
        Rlo, Rhi = f0[:, cf['r_lo']], f0[:, cf['r_hi']]
        Rc = np.sqrt(Rlo*Rhi)
        dV = f0[:, cf['dV']]
        good = dV > 0
        hrms = np.sqrt(np.where(good, f0[:, cf['H2_dV']]/np.maximum(dV, 1e-300), 0.0))
        mrms = np.sqrt(np.where(good, f0[:, cf['M2_dV']]/np.maximum(dV, 1e-300), 0.0))
        fig, ax = plt.subplots(figsize=(6.8, 4.3))
        ax.plot(Rc[good], np.maximum(hrms[good], 1e-22), 'o-', color=SERIES[0],
                ms=3.6, lw=1.4, label=r"$\langle H^2\rangle^{1/2}$")
        ax.plot(Rc[good], np.maximum(mrms[good], 1e-22), 's-', color=SERIES[1],
                ms=3.2, lw=1.4, label=r"$\langle M^2\rangle^{1/2}$")
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r"isotropic coordinate radius $R\ [M]$")
        ax.set_ylabel("volume-weighted RMS constraint violation")
        ax.set_title("Initial constraint profile; dashed lines are the refinement seams")
        mark_scales(ax, seams=True, which=("b", "R_half", "R_t"))
        ax.legend(loc="upper right")
        label_end(ax, Rc[good][-1], max(hrms[good][-1], 1e-22), "$H$", SERIES[0])
        label_end(ax, Rc[good][-1], max(mrms[good][-1], 1e-22), "$M$", SERIES[1])
        fig.savefig(os.path.join(outdir, "f5_constraint_profile_t0.png"))
        plt.close(fig)

        # deposited energy density vs the continuum eps(r)
        Ebar = np.where(good, f0[:, cf['E_dV']]/np.maximum(dV, 1e-300), np.nan)
        rbar = np.where(good, f0[:, cf['rareal_dV']]/np.maximum(dV, 1e-300), np.nan)
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        rr = np.geomspace(0.5, 500.0, 3000)
        ax.plot(rr, np.maximum(P.eps(rr), 1e-22), color=REFLINE, lw=2.0, zorder=3,
                label=r"continuum $\epsilon(r)$")
        m2 = good & np.isfinite(Ebar) & (Ebar > 0)
        ax.plot(rbar[m2], Ebar[m2], 'o', color=SERIES[0], ms=4.0, zorder=4,
                label="deposited $E$, cell average")
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(0.5, 600)
        ax.set_xlabel(r"areal radius $r\ [M]$  (from the evolved metric)")
        ax.set_ylabel(r"static-observer energy density")
        ax.set_title("Deposited source vs the prescribed Plummer energy density")
        mark_scales(ax, seams=False, which=("b", "r_half", "r_t"))
        ax.legend(loc="lower left")
        label_end(ax, rbar[m2][-1], Ebar[m2][-1], "deposited", SERIES[0])
        fig.savefig(os.path.join(outdir, "f6_deposited_E_t0.png"))
        plt.close(fig)

    # ---------------- F7: the 1D construction itself ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    rr = np.geomspace(0.05, 2000.0, 4000)
    Rr = P.R_of_r(rr)
    ax = axes[0]
    ax.plot(Rr, np.exp(P.Phi_exact(np.minimum(rr, 400.0))*(rr <= 400.0))
            * np.where(rr <= 400.0, 1.0, np.sqrt(np.maximum(1 - 2.0/np.maximum(rr, 1e-9), 0))),
            color=SERIES[0], label=r"$\alpha$")
    ax.plot(Rr, P.psi_of_R(Rr), color=SERIES[1], label=r"$\psi$")
    ax.set_xscale('log')
    ax.set_xlabel(r"isotropic radius $R\ [M]$")
    ax.set_ylabel("metric function")
    ax.set_title(r"Static lapse and conformal factor")
    mark_scales(ax, seams=True, which=("b", "R_half", "R_t"))
    ax.legend(loc="center left")
    ax = axes[1]
    ax.plot(rr, np.sqrt(P.vc2(np.minimum(rr, 400.0))), color=SERIES[0], label=r"$v_c$")
    ax.plot(rr, P.m(rr)/rr, color=SERIES[1], label=r"$m/r$")
    ax.axhline(1.0/3.0, color=REFLINE, lw=1.2, ls="--")
    ax.annotate(r"$m/r=1/3$ (no circular orbits above)", xy=(0.06, 1.0/3.0),
                xytext=(0, 5), textcoords="offset points", color=INK2, fontsize=8)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.05, 2000)
    ax.set_xlabel(r"areal radius $r\ [M]$")
    ax.set_ylabel("dimensionless")
    ax.set_title(r"Circular-orbit speed and compactness")
    mark_scales(ax, seams=False, which=("b", "r_vcmax", "r_t"))
    ax.legend(loc="lower left")
    fig.savefig(os.path.join(outdir, "f7_construction_1d.png"))
    plt.close(fig)
    print("wrote figures to", outdir)


if __name__ == "__main__":
    main()
