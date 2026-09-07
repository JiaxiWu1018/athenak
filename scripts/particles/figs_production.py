#!/usr/bin/env python3
"""Production time-series figures for the Plummer campaign.

usage: figs_production.py --hst FILE --out DIR [--modes modes.csv] [--scalars scalars.csv]
                          [--cohorts cohorts.csv] [--shells shells.csv] [--period P]
                          [--fit-lo 0.5 --fit-hi 2.0]

Every amplitude is plotted against ITS OWN finite-N floor. The floor for this sampler is
N_pair^-1/2 because co-located pairs give N/2 independent angular positions; per-band
floors use that band's own pair count. No panel uses two y-scales.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plummer_style as ps
from plummer_style import SERIES, REFLINE, BAND, INK2, MUTED, label_end
import matplotlib.pyplot as plt

ps.apply_style()
NPAIR = 1056768
SHOT = NPAIR**-0.5


def read_hst(path):
    names, rows = None, []
    for line in open(path):
        if line.startswith('#'):
            if '[1]=time' in line or '[1]=time' in line.replace(' ', ''):
                names = [x.split('=')[1] for x in line.replace('#', '').split()
                         if '=' in x]
            continue
        rows.append(line.split())
    a = np.array(rows, dtype=float)
    # a chained restart re-appends the overlap; keep the LAST row per time
    seen, keep = {}, []
    for i, t in enumerate(a[:, 0]):
        seen[t] = i
    idx = sorted(seen.values(), key=lambda i: a[i, 0])
    return {n: k for k, n in enumerate(names)}, a[idx]


def read_tidy(path):
    hdr, rows = None, {}
    order = []
    for line in open(path):
        if line.startswith('#'):
            continue
        if hdr is None:
            hdr = line.strip().split(',')
            continue
        f = line.strip().split(',')
        k = tuple(f[:4]) if len(f) > 3 else tuple(f[:2])
        if k not in rows:
            order.append(k)
        rows[k] = f
    return hdr, [rows[k] for k in order]


def fit_rate(t, y, lo, hi):
    """Least-squares e-folds per unit t over [lo, hi], with a 1-sigma slope error."""
    m = (t >= lo) & (t <= hi) & (y > 0) & np.isfinite(y)
    if m.sum() < 6:
        return None
    x, ly = t[m], np.log(y[m])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, ly, rcond=None)
    n = m.sum()
    resid = ly - A @ coef
    s2 = float(resid @ resid)/max(n - 2, 1)
    cov = s2*np.linalg.inv(A.T @ A)
    return coef[0], float(np.sqrt(cov[0, 0])), int(n), float(x.min()), float(x.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hst", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default=None)
    ap.add_argument("--scalars", default=None)
    ap.add_argument("--cohorts", default=None)
    ap.add_argument("--period", type=float, default=1192.496781)
    ap.add_argument("--fit-lo", type=float, default=0.5)
    ap.add_argument("--fit-hi", type=float, default=2.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    c, H = read_hst(a.hst)
    t = H[:, c['time']]
    tp = t/a.period
    print("history: %d rows, t/P from %.4f to %.4f" % (len(t), tp[0], tp[-1]))
    summary = {}

    # ---- P1: the headline. A_l against the sampler's own floor ----------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, (col, lab) in enumerate((('A1_com', r"$A_1$ (CoM removed)"),
                                    ('A1_raw', r"$A_1$ (raw)"),
                                    ('A2_com', r"$A_2$"),
                                    ('A3_com', r"$A_3$"))):
        if col not in c:
            continue
        y = H[:, c[col]]/SHOT
        ax.plot(tp, y, color=SERIES[i], lw=1.8, label=lab)
        label_end(ax, tp[-1], y[-1], lab.split()[0], SERIES[i])
    ax.axhline(1.0, color=REFLINE, lw=1.4)
    ax.annotate(r"finite-$N$ floor  $N_{\rm pair}^{-1/2}$", xy=(tp[0], 1.0),
                xytext=(3, 5), textcoords="offset points", color=INK2, fontsize=8)
    ax.set_yscale('log')
    ax.set_xlabel(r"$t/P_{1/2}$")
    ax.set_ylabel(r"$A_\ell\,/\,N_{\rm pair}^{-1/2}$")
    ax.set_title("Angular mode amplitudes relative to the sampler's own noise floor")
    ax.legend(loc="upper left", ncol=2)
    fig.savefig(os.path.join(a.out, "p1_modes_vs_shot.png"))
    plt.close(fig)

    fr = fit_rate(tp, H[:, c['A1_com']], a.fit_lo, a.fit_hi) if 'A1_com' in c else None
    if fr:
        summary['A1_com_rate'] = fr
        print("A1_com growth: %.4f +/- %.4f e-folds per P_1/2 over t/P in [%.3f, %.3f] "
              "(%d samples)" % (fr[0], fr[1], fr[3], fr[4], fr[2]))
    fr2 = fit_rate(tp, H[:, c['A1_raw']], a.fit_lo, a.fit_hi) if 'A1_raw' in c else None
    if fr2:
        summary['A1_raw_rate'] = fr2
        print("A1_raw growth: %.4f +/- %.4f e-folds per P_1/2 over t/P in [%.3f, %.3f]"
              % (fr2[0], fr2[1], fr2[3], fr2[4]))

    # ---- P2: bulk structure -- quantile radius, dispersions, centre of mass ---
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    ax = axes[0]
    y = H[:, c['r_q50']]/H[0, c['r_q50']]
    ax.plot(tp, y, color=SERIES[0], lw=1.8)
    ax.axhline(1.0, color=REFLINE, lw=1.2)
    ax.set_xlabel(r"$t/P_{1/2}$"); ax.set_ylabel(r"$r_{q50}(t)/r_{q50}(0)$")
    ax.set_title("Enclosed rest-mass median radius", loc="left")
    label_end(ax, tp[-1], y[-1], r"$r_{q50}$", SERIES[0])
    ax = axes[1]
    for i, col in enumerate(('sigma_r', 'sigma_t')):
        if col in c:
            yy = np.maximum(H[:, c[col]], 1e-20)
            ax.plot(tp, yy, color=SERIES[i], lw=1.8,
                    label=r"$\sigma_r$" if i == 0 else r"$\sigma_t$")
            label_end(ax, tp[-1], yy[-1], r"$\sigma_r$" if i == 0 else r"$\sigma_t$",
                      SERIES[i])
    ax.set_yscale('log')
    ax.set_xlabel(r"$t/P_{1/2}$"); ax.set_ylabel("velocity dispersion")
    ax.set_title("Radial vs tangential dispersion", loc="left")
    ax.legend(loc="center right")
    ax = axes[2]
    ax.plot(tp, H[:, c['Rcom']], color=SERIES[0], lw=1.8)
    ax.set_xlabel(r"$t/P_{1/2}$"); ax.set_ylabel(r"$|R_{\rm com}|\ [M]$")
    ax.set_title("Centre-of-mass excursion", loc="left")
    label_end(ax, tp[-1], H[-1, c['Rcom']], r"$|R_{\rm com}|$", SERIES[0])
    fig.savefig(os.path.join(a.out, "p2_bulk_structure.png"))
    plt.close(fig)

    # ---- P3: numerical health ------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    ax = axes[0]
    for i, (col, lab) in enumerate((('Ham_L2_mat', r"RMS $|H|$, $R \leq R_t$"),
                                    ('Ham_L2', r"RMS $|H|$, whole box"))):
        if col in c:
            yy = np.maximum(H[:, c[col]], 1e-24)
            ax.plot(tp, yy, color=SERIES[i], lw=1.8, label=lab)
            label_end(ax, tp[-1], yy[-1], "matter" if i == 0 else "box", SERIES[i])
    ax.set_yscale('log'); ax.set_xlabel(r"$t/P_{1/2}$")
    ax.set_ylabel("volume-weighted RMS $|H|$")
    ax.set_title("Hamiltonian constraint", loc="left"); ax.legend(loc="lower right")
    ax = axes[1]
    ax.plot(tp, H[:, c['alpha_min']], color=SERIES[0], lw=1.8)
    ax.set_xlabel(r"$t/P_{1/2}$"); ax.set_ylabel(r"$\min \alpha$")
    ax.set_title("Minimum lapse", loc="left")
    label_end(ax, tp[-1], H[-1, c['alpha_min']], r"$\min\alpha$", SERIES[0])
    ax = axes[2]
    n0 = H[0, c['N_alive']]
    ax.plot(tp, H[:, c['N_alive']]/n0, color=SERIES[0], lw=1.8, label="alive / initial")
    if 'boris_nfai' in c:
        ax.plot(tp, 1.0 + H[:, c['boris_nfai']]/max(n0, 1), color=SERIES[1], lw=1.8,
                label=r"$1+$ cumulative pusher fallbacks / $N$")
        label_end(ax, tp[-1], 1.0 + H[-1, c['boris_nfai']]/max(n0, 1), "fallbacks",
                  SERIES[1])
    ax.set_xlabel(r"$t/P_{1/2}$"); ax.set_ylabel("fraction of $N$")
    ax.set_title("Particle ledger", loc="left"); ax.legend(loc="center left")
    label_end(ax, tp[-1], H[-1, c['N_alive']]/n0, "alive", SERIES[0])
    fig.savefig(os.path.join(a.out, "p3_numerical_health.png"))
    plt.close(fig)

    # ---- P4: per-band A_1, each over its own floor ----------------------------
    if a.modes and os.path.exists(a.modes):
        hdr, rows = read_tidy(a.modes)
        k = {n: i for i, n in enumerate(hdr)}
        bands = sorted({r[k['band']] for r in rows if r[k['band']].startswith('m')})
        if bands:
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            for i, b in enumerate(bands):
                sel = [r for r in rows if r[k['band']] == b and int(r[k['l']]) == 1]
                if not sel:
                    continue
                x = np.array([float(r[k['t_over_P']]) for r in sel])
                y = np.array([float(r[k['A_l']])/max(float(r[k['A_shot']]), 1e-300)
                              for r in sel])
                o = np.argsort(x)
                ax.plot(x[o], y[o], color=SERIES[i % len(SERIES)], lw=1.7,
                        label="band %s" % b)
                label_end(ax, x[o][-1], y[o][-1], b, SERIES[i % len(SERIES)])
            ax.axhline(1.0, color=REFLINE, lw=1.4)
            ax.set_yscale('log'); ax.set_xlabel(r"$t/P_{1/2}$")
            ax.set_ylabel(r"$A_1\,/\,$ band's own $n_{\rm pair}^{-1/2}$")
            ax.set_title("Dipole by initial-radius band, each over its own floor")
            ax.legend(loc="upper left", ncol=2)
            fig.savefig(os.path.join(a.out, "p4_dipole_by_band.png"))
            plt.close(fig)

    # ---- P5: cohort radii -- bulk motion vs shell mixing ---------------------
    if a.cohorts and os.path.exists(a.cohorts):
        hdr, rows = read_tidy(a.cohorts)
        k = {n: i for i, n in enumerate(hdr)}
        cids = sorted({int(r[k['cohort']]) for r in rows})
        pick = cids[::max(1, len(cids)//6)][:6]
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for i, ci in enumerate(pick):
            sel = [r for r in rows if int(r[k['cohort']]) == ci]
            x = np.array([float(r[k['t_over_P']]) for r in sel])
            y = np.array([float(r[k['r_mean']]) for r in sel])
            o = np.argsort(x)
            if y[o][0] <= 0:
                continue
            ax.plot(x[o], y[o]/y[o][0], color=SERIES[i % len(SERIES)], lw=1.7,
                    label="cohort %d" % ci)
            label_end(ax, x[o][-1], y[o][-1]/y[o][0], "c%d" % ci,
                      SERIES[i % len(SERIES)])
        ax.axhline(1.0, color=REFLINE, lw=1.2)
        ax.set_xlabel(r"$t/P_{1/2}$")
        ax.set_ylabel(r"$\langle r\rangle_{\rm cohort}(t)\,/\,\langle r\rangle(0)$")
        ax.set_title("Lagrangian cohorts: bulk expansion or contraction, by initial radius")
        ax.legend(loc="upper left", ncol=2)
        fig.savefig(os.path.join(a.out, "p5_cohort_radii.png"))
        plt.close(fig)

    # ---- P6: orbit invariants ------------------------------------------------
    if a.scalars and os.path.exists(a.scalars):
        hdr, rows = read_tidy(a.scalars)
        k = {n: i for i, n in enumerate(hdr)}
        x = np.array([float(r[k['t_over_P']]) for r in rows])
        o = np.argsort(x)
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for i, (col, lab) in enumerate((('dE_rms', r"rms $\delta E/E$"),
                                        ('dL_rms', r"rms $\delta |L|/|L|$"),
                                        ('dr_rms', r"rms $\delta r/r$"))):
            y = np.array([float(r[k[col]]) for r in rows])[o]
            ax.plot(x[o], np.maximum(y, 1e-12), color=SERIES[i], lw=1.8, label=lab)
            label_end(ax, x[o][-1], max(y[-1], 1e-12), lab.split()[-1], SERIES[i])
        ax.set_yscale('log'); ax.set_xlabel(r"$t/P_{1/2}$")
        ax.set_ylabel("relative drift")
        ax.set_title(r"Orbit invariants $E=-u_t$ and $|L|=|x\times u|$, and the radius")
        ax.legend(loc="lower right")
        fig.savefig(os.path.join(a.out, "p6_orbit_invariants.png"))
        plt.close(fig)

    with open(os.path.join(a.out, "FIT_SUMMARY.txt"), "w") as f:
        f.write("history rows %d, t/P from %.6f to %.6f\n" % (len(t), tp[0], tp[-1]))
        f.write("A_shot = N_pair^-1/2 = %.10e  (N_pair = %d)\n" % (SHOT, NPAIR))
        for key, v in summary.items():
            f.write("%s: %.6f +/- %.6f e-folds per P_1/2 over t/P [%.4f, %.4f], "
                    "%d samples\n" % (key, v[0], v[1], v[3], v[4], v[2]))
        for col in ('A1_raw', 'A1_com', 'A2_com', 'A3_com', 'A4_com'):
            if col in c:
                f.write("%s: t=0 %.6e (%.4f shot) -> final %.6e (%.4f shot)\n"
                        % (col, H[0, c[col]], H[0, c[col]]/SHOT,
                           H[-1, c[col]], H[-1, c[col]]/SHOT))
        for col in ('r_q50', 'sigma_r', 'sigma_t', 'alpha_min', 'Ham_L2_mat', 'Rcom',
                    'N_alive'):
            if col in c:
                f.write("%s: t=0 %.6e -> final %.6e\n" % (col, H[0, c[col]],
                                                          H[-1, c[col]]))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
