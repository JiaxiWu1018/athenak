#!/usr/bin/env python3
"""Session-2 milestone figure suite.

Regenerated after EVERY completed P_1/2 into a milestone-specific directory, always
covering t = 0 through that milestone, so the evolving trend is visible rather than a
sequence of isolated snapshots, and no earlier milestone is overwritten.

Produces the set required by the specification's section 11, plus the extraction-radius
consistency panel required by section 9:

  f1_global_multipoles   A_1..A_4 about the CoM vs t/P_1/2
  f2_quartile_l1         l = 1 in the four equal-rest-mass quartiles, raw and in units
                         of each band's own finite-N null
  f3_dipole_direction    cos(D_q, D_q') for neighbouring quartiles, and the wander of
                         the dominant direction
  f4_adm_momentum        P_x, P_y, P_z, |P| (ADM) vs t/P_1/2, with the matter-side momenta
  f5_momentum_radii      the same |P| at every extraction radius: the radius-independence
                         test, and which radius is quoted
  f6_com_motion          X, Y, Z, R of the coordinate centre of mass
  f7_equilibrium         enclosed-rest-mass radii, Lagrangian cohort trajectories, and the
                         deposited density profile at successive times
  f8_velocity            sigma_r, sigma_t and their ratio
  f9_health              minimum lapse, constraint norms, particle count, non-finite
                         states, GR-Boris fallbacks

All amplitudes are CENTRE-OF-MASS referenced.  Session 1 established that an
origin-referenced band dipole is contaminated by a purely geometric (2/3)<1/r>|s| term
that is largest exactly where the signal was claimed -- the core -- so the
origin-referenced series is drawn only as a faint coordinate-drift diagnostic, never as
the measurement.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import s2_style as st                    # noqa: E402


def mag_axis(ax, values, floor=1e-16):
    """Axis for a NON-NEGATIVE magnitude such as |P|.

    Plain log, not symlog.  These runs span |P| from the 1e-12 quadrature floor to the
    1e-5 the outgoing constraint front produces as it crosses a sphere, and on a symlog
    axis whose linear threshold is set by the maximum the whole 1e-12 floor collapses
    onto the zero line -- which is precisely the part that carries the physics, because
    the radii the front has not reached are the ones that measure the momentum.

    The exception is an identically-zero series (t = 0, where K_ij vanishes in static
    initial data): log cannot show it, so use a linear axis and say so.
    """
    v = np.asarray([x for x in np.ravel(values) if np.isfinite(x)])
    if v.size == 0 or np.all(v == 0.0):
        ax.set_ylim(-1.0, 1.0)
        ax.annotate('identically zero at every stored time\n'
                    r'($K_{ij}\equiv 0$ in static initial data)',
                    xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center',
                    fontsize=8, color=st.MUTED)
        return
    pos = np.abs(v[v != 0.0])
    if pos.size == 0:
        return
    ax.set_yscale('log')
    ax.set_ylim(max(floor, 0.3 * pos.min()), 3.0 * pos.max())


def dedup(df, keys):
    """Drop duplicated rows.

    A segment that ends exactly on a milestone gets its outputs written twice: once by
    the regular cadence and once by Driver::Finalize.  The t = 0 preflights, with
    tlim = 0, duplicate every row for the same reason.  Deduplicating on the identifying
    keys is therefore mandatory, not defensive.
    """
    return df.drop_duplicates(subset=keys, keep='last').sort_values('time')


def read_hst(path):
    """AthenaK history: '# [1]=name [2]=name ...' then whitespace columns."""
    names = None
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') and '=' in line:
                names = [p.split('=', 1)[1] for p in line.replace('#', '').split()
                         if '=' in p]
            elif not line.startswith('#'):
                break
    df = pd.read_csv(path, comment='#', sep=r'\s+', header=None)
    if names and len(names) == df.shape[1]:
        df.columns = names
    return dedup(df, ['time'])


def f1_global(red, ref, ms, out):
    m = dedup(pd.read_csv(os.path.join(red, 'modes.csv')), ['time', 'band', 'l'])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    for ax, logy in zip(axes, (False, True)):
        for i, l in enumerate((1, 2, 3, 4)):
            s = m[(m.band == 'com') & (m.l == l)]
            ax.plot(s.t_over_P, s.A_l, label=r'$A_%d^{\rm CoM}$' % l, **st.style(i))
        st.mark_null(ax, ref['A_shot_global'],
                     r'$A^{\rm shot}=N_{\rm pair}^{-1/2}$')
        st.periods_axis(ax, ref, ms)
        ax.set_ylabel(r'$A_\ell$')
        if logy:
            ax.set_yscale('log')
            ax.set_title('same, logarithmic')
        else:
            ax.set_title('Global multipoles about the centre of mass')
            ax.legend(ncol=2, loc='upper left')
    st.stamp(fig, ref, ms)
    st.save(fig, out, 'f1_global_multipoles')
    plt.close(fig)


def f2_quartiles(red, ref, ms, out):
    m = dedup(pd.read_csv(os.path.join(red, 'modes.csv')), ['time', 'band', 'l'])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    labels = ['core (q0)', 'q1', 'q2', 'outer (q3)']
    for q in range(4):
        s = m[(m.band == 'm%d' % q) & (m.l == 1)]
        axes[0].plot(s.t_over_P, s.A_l, label=r'$A_{1,%s}$' % labels[q], **st.style(q))
        axes[1].plot(s.t_over_P, s.A_l / s.A_shot, label=labels[q], **st.style(q))
    o = m[(m.band == 'o0') & (m.l == 1)]
    if len(o):
        axes[0].plot(o.t_over_P, o.A_l, color=st.REFC, lw=0.9, ls=(0, (1, 2)),
                     label='core, origin-referenced (drift diagnostic only)')
    st.mark_null(axes[0], ref['A_shot_quartile'],
                 r'$(N_{\rm pair}/4)^{-1/2}$')
    st.mark_null(axes[1], 1.0, 'own null')
    for ax, t in zip(axes, ['$\\ell=1$ per equal-rest-mass quartile, about the CoM',
                            'in units of each band\'s own finite-N null']):
        st.periods_axis(ax, ref, ms)
        ax.set_title(t)
    axes[0].set_ylabel(r'$A_{1,q}$')
    axes[1].set_ylabel(r'$A_{1,q}/A^{\rm shot}_q$')
    axes[0].legend(loc='center left', fontsize=7.2)
    axes[1].legend(ncol=2, loc='upper left')
    axes[1].annotate('a ratio > 1 is NOT by itself evidence of instability:\n'
                     'CoM subtraction on a cusped profile lifts the core band\n'
                     'above its nominal Poisson floor already at $t=0$',
                     xy=(0.98, 0.02), xycoords='axes fraction', fontsize=7,
                     color=st.MUTED, va='bottom', ha='right',
                     bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='none',
                               alpha=0.88))
    st.stamp(fig, ref, ms)
    st.save(fig, out, 'f2_quartile_l1')
    plt.close(fig)


def f3_direction(red, ref, ms, out):
    p = os.path.join(red, 'dipoles.csv')
    if not os.path.exists(p):
        print('  (no dipoles.csv; skipping f3)')
        return
    d = dedup(pd.read_csv(p), ['time', 'band'])
    piv = {b: g.set_index('time')[['Dx', 'Dy', 'Dz']] for b, g in d.groupby('band')}
    t = sorted(set.intersection(*[set(v.index) for v in piv.values()]))
    t = np.array(t)
    tp = t / ref['P_half']
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    pairs = [('m0', 'm1'), ('m1', 'm2'), ('m2', 'm3'), ('m0', 'm3')]
    for i, (a, b) in enumerate(pairs):
        if a not in piv or b not in piv:
            continue
        A = piv[a].loc[t].to_numpy()
        B = piv[b].loc[t].to_numpy()
        c = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))
        axes[0].plot(tp, c, label=r'$\cos(\mathbf{D}_{%s},\mathbf{D}_{%s})$'
                     % (a[1], b[1]), **st.style(i))
    axes[0].axhline(0.0, color=st.REFC, lw=1.0, ls=(0, (4, 2)))
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_ylabel(r'$\cos\theta$')
    axes[0].set_title('Shell dipole alignment: aligned = translation, '
                      'anti-aligned = sloshing')
    axes[0].legend(ncol=2, loc='lower left', fontsize=7.4)
    # direction wander: angle of each band's dipole from its own time-mean direction
    for i, b in enumerate(['m0', 'm1', 'm2', 'm3']):
        if b not in piv:
            continue
        A = piv[b].loc[t].to_numpy()
        u = A / np.linalg.norm(A, axis=1)[:, None]
        mean = u.mean(0)
        mean /= np.linalg.norm(mean)
        ang = np.degrees(np.arccos(np.clip(u @ mean, -1, 1)))
        axes[1].plot(tp, ang, label='q%s' % b[1], **st.style(i))
    axes[1].axhline(90.0, color=st.REFC, lw=1.0, ls=(0, (4, 2)))
    axes[1].annotate('90$^\\circ$ = isotropic wander', xy=(0.99, 90),
                     xycoords=('axes fraction', 'data'), ha='right', va='bottom',
                     fontsize=7.5, color=st.MUTED)
    axes[1].set_ylabel(r'angle from own time-mean direction [deg]')
    axes[1].set_title('Does the direction lock or wander?')
    axes[1].legend(ncol=4, loc='upper left')
    for ax in axes:
        st.periods_axis(ax, ref, ms)
    st.stamp(fig, ref, ms)
    st.save(fig, out, 'f3_dipole_direction')
    plt.close(fig)


def load_admmom(rundir, ref):
    g = glob.glob(os.path.join(rundir, 'out', '*.plummer_admmom.csv'))
    if not g:
        return None
    d = dedup(pd.read_csv(g[0], comment='#'), ['time', 'R'])
    d['t_over_P'] = d.time / ref['P_half']
    return d


def f4_momentum(rundir, ref, ms, out):
    d = load_admmom(rundir, ref)
    if d is None or not len(d):
        print('  (no plummer_admmom.csv; skipping f4/f5)')
        return
    Rq = sorted(d.R.unique())[-1]          # outermost radius = best estimate
    s = d[d.R == Rq].sort_values('time')
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    for i, (c, lab) in enumerate([('Px_adm', '$P_x$'), ('Py_adm', '$P_y$'),
                                  ('Pz_adm', '$P_z$'), ('absP_adm', '$|P|$')]):
        axes[0].plot(s.t_over_P, s[c], label=lab, **st.style(i))
    axes[0].axhline(0.0, color=st.REFC, lw=1.0, ls=(0, (4, 2)))
    axes[0].set_ylabel(r'$P_i^{\rm ADM}$  [$M$]')
    axes[0].set_title(r'ADM linear momentum at $R = %.6g\,M$ '
                      r'($R/R_t = %.2f$)' % (Rq, Rq / ref['R_t']))
    axes[0].legend(ncol=4, loc='upper left')
    for i, (c, lab) in enumerate([('absP_adm', r'$|P^{\rm ADM}|$'),
                                  ('absP_matter', r'$|P^{\rm matter}| = |\sum_p m_p u_i|$')]):
        axes[1].plot(s.t_over_P, np.abs(s[c]), label=lab, **st.style(i))
    dep = np.sqrt(s.Px_dep**2 + s.Py_dep**2 + s.Pz_dep**2)
    axes[1].plot(s.t_over_P, dep, label=r'$|P^{\rm dep}| = |\int S_i \sqrt{\gamma}d^3x|$',
                 **st.style(2))
    mag_axis(axes[1], np.concatenate(
        [np.abs(s.absP_adm.to_numpy()), np.abs(s.absP_matter.to_numpy()),
         dep.to_numpy()]))
    axes[1].set_ylabel(r'$|P|$  [$M$]')
    axes[1].set_title('surface vs matter-side momenta')
    axes[1].legend(loc='center right', fontsize=7.4)
    # On a shared symlog axis a value nine orders below the others reads as a flat zero,
    # so state it.  The separation IS the result: the particle momentum sum and the
    # deposited source both drift (neither is a conserved quantity in curved spacetime),
    # while the spacetime's total momentum does not.
    axes[1].annotate(r'$|P^{\rm ADM}|$ stays at $%.1e$ to $%.1e\,M$;'
                     '\n'
                     r'$|P^{\rm matter}|$ reaches $%.1e\,M$ -- a factor $%.0e$ larger.'
                     '\n'
                     'Matter-side sums are coordinate quantities and are not conserved;\n'
                     'the ADM surface value is the total momentum of the spacetime.'
                     % (np.abs(s.absP_adm).min(), np.abs(s.absP_adm).max(),
                        np.abs(s.absP_matter).max(),
                        np.abs(s.absP_matter).max()
                        / max(np.abs(s.absP_adm).max(), 1e-300)),
                     xy=(0.02, 0.03), xycoords='axes fraction', fontsize=7,
                     color=st.MUTED, va='bottom')
    for ax in axes:
        st.periods_axis(ax, ref, ms)
    st.stamp(fig, ref, ms)
    st.save(fig, out, 'f4_adm_momentum')
    plt.close(fig)


def f5_radii(rundir, ref, ms, out):
    d = load_admmom(rundir, ref)
    if d is None or not len(d):
        return
    radii = sorted(d.R.unique())
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    cols = st.ramp(len(radii))
    for i, R in enumerate(radii):
        s = d[d.R == R].sort_values('time')
        axes[0].plot(s.t_over_P, s.absP_adm, color=cols[i], lw=1.3,
                     label=r'$R=%.4g$' % R)
    mag_axis(axes[0], d.absP_adm.to_numpy())
    axes[0].set_ylabel(r'$|P^{\rm ADM}|$  [$M$]')
    axes[0].set_title('Extraction-radius consistency: every sphere lies wholly\n'
                      'on one refinement level, in vacuum')
    axes[0].legend(ncol=2, loc='upper left', fontsize=7.2)
    st.periods_axis(axes[0], ref, ms)
    # radius dependence at the final available time
    tf = d.time.max()
    s = d[np.isclose(d.time, tf)].sort_values('R')
    axes[1].plot(s.R / ref['R_t'], s.absP_adm, marker='o', ms=5,
                 color=st.SERIES[0], lw=1.4, label=r'$|P^{\rm ADM}|$')
    axes[1].plot(s.R / ref['R_t'], np.abs(s.area_ratio - 1.0), marker='s', ms=5,
                 color=st.SERIES[1], lw=1.4, ls=st.DASH[1],
                 label=r'$|A/4\pi R^2 - 1|$ (area control)')
    if np.all(s.absP_adm.to_numpy() == 0.0):
        axes[1].set_yscale('log')
        axes[1].annotate(r'$|P^{\rm ADM}| \equiv 0$ (below the axis)',
                         xy=(0.03, 0.06), xycoords='axes fraction', fontsize=7.6,
                         color=st.MUTED)
    else:
        axes[1].set_yscale('log')
    axes[1].set_xlabel(r'$R / R_t$')
    axes[1].set_ylabel('value at the final stored time')
    axes[1].set_title(r'at $t = %.4g\,M$ ($t/P_{1/2} = %.3f$)'
                      % (tf, tf / ref['P_half']))
    clean = s.R[s.absP_adm.abs() < 1e-9].to_numpy()
    if clean.size and clean.size < len(s):
        axes[1].axvspan(clean.min() / ref['R_t'], s.R.max() / ref['R_t'],
                        color='#eef4fa', zorder=0)
        axes[1].annotate('uncrossed by the outgoing constraint front:\n'
                         'these radii measure the momentum',
                         xy=(clean.min() / ref['R_t'], 0.03),
                         xycoords=('data', 'axes fraction'), fontsize=7,
                         color=st.MUTED, va='bottom', ha='left')
    axes[1].legend(loc='best', fontsize=7.4)
    st.stamp(fig, ref, ms)
    st.save(fig, out, 'f5_momentum_radii')
    plt.close(fig)


def f6_com(red, ref, ms, out):
    s = dedup(pd.read_csv(os.path.join(red, 'scalars.csv')), ['time'])
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for i, (c, lab) in enumerate([('com_x', '$X_{\\rm CoM}$'),
                                  ('com_y', '$Y_{\\rm CoM}$'),
                                  ('com_z', '$Z_{\\rm CoM}$'),
                                  ('Rcom', '$R_{\\rm CoM}$')]):
        ax.plot(s.t_over_P, s[c], label=lab, **st.style(i))
    ax.axhline(0.0, color=st.REFC, lw=1.0, ls=(0, (4, 2)))
    st.periods_axis(ax, ref, ms)
    ax.set_ylabel(r'coordinate CoM  [$M$]')
    ax.set_title('Centre-of-mass motion (isotropic coordinates)')
    ax.legend(ncol=2, loc='upper left')
    st.stamp(fig, ref, ms)
    st.save(fig, out, 'f6_com_motion')
    plt.close(fig)


def f7_equilibrium(red, rundir, ref, ms, out):
    s = dedup(pd.read_csv(os.path.join(red, 'scalars.csv')), ['time'])
    co = dedup(pd.read_csv(os.path.join(red, 'cohorts.csv')), ['time', 'cohort'])
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.5))
    for i, q in enumerate((10, 25, 50, 75, 90)):
        c = 'r_q%02d' % q
        if c in s:
            axes[0].plot(s.t_over_P, s[c] / s[c].iloc[0], label='%d %%' % q,
                         **st.style(i))
    axes[0].axhline(1.0, color=st.REFC, lw=1.0, ls=(0, (4, 2)))
    axes[0].set_ylabel('enclosed-rest-mass radius / its $t=0$ value')
    axes[0].set_title('Radial equilibrium of the bulk')
    axes[0].legend(ncol=3, loc='upper left', fontsize=7.6)
    sel = [0, 8, 16, 24, 31]
    cols = st.ramp(len(sel))
    for i, ci in enumerate(sel):
        g = co[co.cohort == ci].sort_values('time')
        if len(g):
            axes[1].plot(g.t_over_P, g.r_mean / g.r_mean.iloc[0], color=cols[i],
                         lw=1.3, label='cohort %d' % ci)
    axes[1].axhline(1.0, color=st.REFC, lw=1.0, ls=(0, (4, 2)))
    axes[1].set_ylabel(r'$\langle r\rangle$ / its $t=0$ value')
    axes[1].set_title('Lagrangian cohort trajectories')
    axes[1].legend(ncol=2, loc='upper left', fontsize=7.2)
    # deposited density profile at successive times, from the in-code fields ledger
    g = glob.glob(os.path.join(rundir, 'out', '*.plummer_fields.csv'))
    if g:
        f = dedup(pd.read_csv(g[0], comment='#'), ['time', 'bin'])
        times = sorted(f.time.unique())
        pick = [times[int(k * (len(times) - 1) / 4)] for k in range(5)] if times else []
        cols = st.ramp(len(pick))
        for i, tt in enumerate(pick):
            h = f[f.time == tt]
            h = h[h.dV > 0]
            Rm = np.sqrt(h.r_lo * h.r_hi)
            axes[2].plot(Rm, h.E_dV / h.dV, color=cols[i], lw=1.3,
                         label=r'$t/P=%.2f$' % (tt / ref['P_half']))
        axes[2].set_xscale('log')
        axes[2].set_yscale('log')
        axes[2].axvline(ref['R_t'], color=st.REFC, lw=0.9, ls=(0, (4, 2)))
        axes[2].annotate('$R_t$', xy=(ref['R_t'], 0.02), xycoords=('data', 'axes fraction'),
                         fontsize=7.5, color=st.MUTED)
        axes[2].set_xlabel(r'isotropic radius $R$  [$M$]')
        axes[2].set_ylabel(r'deposited $E$')
        axes[2].set_title('Density profile (volume-weighted, on the mesh)')
        axes[2].legend(loc='lower left', fontsize=7.2)
    for ax in axes[:2]:
        st.periods_axis(ax, ref, ms)
    st.stamp(fig, ref, ms)
    st.save(fig, out, 'f7_equilibrium')
    plt.close(fig)


def f8_velocity(red, rundir, ref, ms, out):
    s = dedup(pd.read_csv(os.path.join(red, 'scalars.csv')), ['time'])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    axes[0].plot(s.t_over_P, s.sigma_r_u, label=r'$\sigma_r$', **st.style(0))
    axes[0].plot(s.t_over_P, s.sigma_t_u, label=r'$\sigma_t$', **st.style(1))
    axes[0].set_ylabel(r'dispersion of $u_i$')
    axes[0].set_title(r'Velocity dispersions ($\sigma_r = 0$ by construction at $t=0$)')
    axes[0].legend(loc='upper left')
    with np.errstate(divide='ignore', invalid='ignore'):
        axes[1].plot(s.t_over_P, s.sigma_r_u / s.sigma_t_u, **st.style(2))
    axes[1].set_ylabel(r'$\sigma_r/\sigma_t$')
    axes[1].set_title('Radial-to-tangential ratio (isotropy would be $1/\\sqrt{2}$)')
    axes[1].axhline(2.0**-0.5, color=st.REFC, lw=1.0, ls=(0, (4, 2)))
    for ax in axes:
        st.periods_axis(ax, ref, ms)
    st.stamp(fig, ref, ms)
    st.save(fig, out, 'f8_velocity')
    plt.close(fig)


def f9_health(red, rundir, ref, ms, out):
    g = glob.glob(os.path.join(rundir, 'out', '*[!c].user.hst'))
    gz = glob.glob(os.path.join(rundir, 'out', '*.z4c.user.hst'))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.5))
    if g:
        h = read_hst(sorted(g, key=len)[0])
        tp = h.time / ref['P_half']
        axes[0].plot(tp, h['alpha_min'], label=r'$\min\alpha$', **st.style(0))
        axes[0].axhline(ref['alpha_min'], color=st.REFC, lw=1.0, ls=(0, (4, 2)))
        axes[0].annotate(r'continuum $\alpha(0) = %.6f$' % ref['alpha_min'],
                         xy=(0.99, ref['alpha_min']),
                         xycoords=('axes fraction', 'data'), ha='right', va='bottom',
                         fontsize=7.5, color=st.MUTED)
        axes[0].set_ylabel(r'$\min \alpha$')
        axes[0].set_title('Minimum lapse')
        axes[0].legend(loc='lower left')
        axes[1].plot(tp, h['Ham_L2'], label=r'$\|H\|_2$ (whole box)', **st.style(0))
        axes[1].plot(tp, h['Ham_L2_mat'], label=r'$\|H\|_2$ (matter region)',
                     **st.style(1))
        if gz:
            z = read_hst(gz[0])
            axes[1].plot(z.time / ref['P_half'], np.sqrt(np.abs(z['M-norm2'])),
                         label=r'$\|M\|_2$ (momentum constraint)', **st.style(2))
        axes[1].set_yscale('log')
        axes[1].set_ylabel('constraint norm')
        axes[1].set_title('Constraints')
        axes[1].legend(loc='lower right', fontsize=7.4)
        axes[2].plot(tp, h['N_alive'] / ref['Npart'], label=r'$N_{\rm alive}/N$',
                     **st.style(0))
        axes[2].plot(tp, np.maximum(h['N_nonfinit'], 1e-1) / ref['Npart'],
                     label='non-finite / N', **st.style(1))
        axes[2].plot(tp, np.maximum(h['boris_nfai'], 1e-1) / ref['Npart'],
                     label='GR-Boris fallbacks / N', **st.style(2))
        axes[2].set_yscale('log')
        axes[2].set_ylabel('fraction of N')
        axes[2].set_title('Particle health')
        axes[2].legend(loc='center left', fontsize=7.4)
    for ax in axes:
        st.periods_axis(ax, ref, ms)
    st.stamp(fig, ref, ms)
    st.save(fig, out, 'f9_health')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--milestone', type=float, required=True)
    ap.add_argument('--reduced', required=True)
    ap.add_argument('--rundir', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    st.apply()
    ref = st.load_ref(a.ref)
    ms = a.milestone
    print('figs_session2: case %s through %g P_1/2 -> %s' % (a.case, ms, a.out))
    for fn, args in [(f1_global, (a.reduced, ref, ms, a.out)),
                     (f2_quartiles, (a.reduced, ref, ms, a.out)),
                     (f3_direction, (a.reduced, ref, ms, a.out)),
                     (f4_momentum, (a.rundir, ref, ms, a.out)),
                     (f5_radii, (a.rundir, ref, ms, a.out)),
                     (f6_com, (a.reduced, ref, ms, a.out)),
                     (f7_equilibrium, (a.reduced, a.rundir, ref, ms, a.out)),
                     (f8_velocity, (a.reduced, a.rundir, ref, ms, a.out)),
                     (f9_health, (a.reduced, a.rundir, ref, ms, a.out))]:
        try:
            fn(*args)
        except Exception as e:                       # a missing product must not lose the rest
            print('  %s FAILED: %s: %s' % (fn.__name__, type(e).__name__, e))
    print('figs_session2: done')


if __name__ == '__main__':
    main()
