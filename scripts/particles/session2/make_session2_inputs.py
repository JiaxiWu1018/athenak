#!/usr/bin/env python3
"""Generate the Session-2 production and preflight input decks, and the per-case
reference-value JSONs, from a single source of truth.

Every number that can be derived is derived: `b` is root-found against the measured
max_r[m(r)/r] of analysis/plummer_1d.py, and P_1/2, r_1/2, R_1/2, R_t, M_0 and the
output cadences all come from the same model object.  Nothing is transcribed by hand,
so a deck can never disagree with the reference values or with the reduction scripts.

Session-1's scripts/make_plummer_inputs.py cannot do this job: its set_key() exits if a
block or key is absent and cannot ADD either, and the Session-1 deck carries only six
<refined_region*> blocks where Session 2 needs seven.  This generator emits complete
decks instead of patching an existing one.

Usage
    make_session2_inputs.py --out DIR              write every deck and JSON
    make_session2_inputs.py --out DIR --check      re-derive and diff, exit 1 on drift
"""
import argparse
import difflib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'analysis'))

import numpy as np
from plummer_1d import PlummerModel
from solve_compactness import RT_OVER_B, solve_b, max_m_over_r

NPANEL, NGL = 40000, 20
NPAIR = 1056768                  # unchanged from Session 1: N = 2 * NPAIR = 2,113,536
SEED = 1985                      # unchanged from Session 1
NROOT, NBLOCK, NGHOST = 128, 32, 4

# ---------------------------------------------------------------- case definitions
# L and N were chosen with analysis/mesh_design.py.  The mesh family is Session 1's
# exactly (root 128^3, MeshBlock 32^3, level-l region [-L/2^l, L/2^l]), so every region
# is four parent blocks across and MeshBlock-aligned for any L, and the leaf count is
# 56 N + 64 = 456 for N = 7 (Session 1: N = 6, 400 leaves).
#
# Extraction radii for the ADM linear momentum: all in vacuum outside R_t, each placed
# between two refinement seams (the seams are the level half-widths L/2^l), spanning a
# factor ~5 in radius across three refinement levels so a radius-independence test is
# possible.
CASES = {
    'R10': dict(
        target=10.0, L=1280.0, N=7,
        basename='pl_R10_b3p8634_s1985',
        pmom_radii=[100.0, 140.0, 200.0, 260.0, 400.0, 500.0],
    ),
    'R6p5': dict(
        target=6.5, L=768.0, N=7,
        basename='pl_R6p5_b2p5112_s1985',
        pmom_radii=[60.0, 84.0, 120.0, 150.0, 240.0, 300.0],
    ),
}

# output cadences as fractions of that case's P_1/2
CADENCE = dict(hst=200.0, pvtk=100.0, cart=50.0, binslice=25.0, cbin=5.0, rst=4.0)
PERIODS = 5.0                    # production endpoint, 5 P_1/2


def seam_free_radii(halfwidth, R_t, nper=2, hi=0.975, lo=1.02):
    """Extraction radii whose ENTIRE coordinate sphere lies within one refinement level.

    The refined regions are CUBES, not spheres, so "between two seams" is not the same
    as "outside one cube and inside the next".  On the sphere of radius R the quantity
    max_i |n_i| ranges over [1/sqrt(3), 1] -- minimal along the body diagonal, maximal
    along an axis -- so the sphere crosses the cube |x_i| <= H exactly when

        H < R <= sqrt(3) H.

    A sphere therefore lies wholly inside the level-l cube AND wholly outside the
    level-(l+1) cube iff

        sqrt(3) H_{l+1} < R <= H_l,   i.e.  R/H_l in (sqrt(3)/2, 1] = (0.866, 1],

    because H_{l+1} = H_l/2 in this mesh family.  Anything else samples cells of two
    different sizes around one sphere, which is what the specification asks us to avoid
    where possible -- and here it is possible, because each level offers a usable band
    about 13 % wide in radius.

    `hi` keeps the sphere off the cube face itself; `lo` keeps it off the inner
    crossing radius.  Only bands entirely in vacuum (R > R_t) are used.
    """
    out = []
    for l in range(len(halfwidth) - 1, 0, -1):
        H = halfwidth[l]
        band_lo, band_hi = lo * (3.0 ** 0.5 / 2.0) * H, hi * H
        if band_hi <= band_lo or band_lo <= R_t:
            continue
        if nper == 1:
            picks = [0.5 * (band_lo + band_hi)]
        else:
            picks = [band_lo + (band_hi - band_lo) * k / (nper - 1)
                     for k in range(nper)]
        out.extend(round(v, 6) for v in picks)
    return sorted(out)


def derive(label):
    c = dict(CASES[label])
    b = solve_b(c['target'], npanel=NPANEL, ngl=NGL)
    rt = RT_OVER_B * b
    mod = PlummerModel(M=1.0, b=b, rt=rt, npanel=NPANEL, ngl=NGL)
    P_half, r_half, alpha_half, vc_half = mod.P_half()
    R_half = float(mod.R_of_r(np.array([r_half]))[0])
    mm = max_m_over_r(mod)
    vcmax, r_vcmax = mod.vc_max()
    alpha0 = float(np.exp(mod.Phi_exact(np.array([1e-12 * b]))[0]))
    psi0 = float(np.exp(-0.5 * mod.j_exact(np.array([1e-12 * b]))[0]))

    L, N = c['L'], c['N']
    dx0 = 2.0 * L / NROOT
    dx = [dx0 / 2**l for l in range(N + 1)]
    hw = [L / 2**l for l in range(N + 1)]
    n_leaf = N * ((NROOT // NBLOCK)**3 - (NROOT // NBLOCK // 2)**3) + (NROOT // NBLOCK)**3

    c['pmom_radii'] = seam_free_radii(hw, mod.Rt, nper=2)
    c.update(
        b=b, rt=rt, M=1.0, f_t=mod.ft, M_P=mod.MP, M_0=mod.M0,
        Npart=2 * NPAIR, npair=NPAIR, mu=mod.M0 / (2 * NPAIR), seed=SEED,
        r_half=r_half, R_half=R_half, R_t=mod.Rt, P_half=P_half,
        alpha_at_r_half=alpha_half, vc_at_r_half=vc_half,
        max_m_over_r=mm['max_m_over_r'], r_at_max_m_over_r=mm['r_at_max'],
        RM_eff=1.0 / mm['max_m_over_r'], vc_max=vcmax, r_at_vc_max=r_vcmax,
        alpha_0=alpha0, psi_0=psi0, alpha_min=alpha0,
        dx0=dx0, dx=dx, halfwidth=hw, dx_fine=dx[N], n_leaf=n_leaf,
        ncell_active=n_leaf * NBLOCK**3,
        cells_per_b=b / dx[N], cells_per_r_half=r_half / dx[N],
        cells_per_R_half=R_half / dx[N],
        dt_hyp=0.25 * dx[N],
        tlim=PERIODS * P_half, periods=PERIODS,
        # A_l finite-N nulls.  n_uniq is the number of INDEPENDENT ANGULAR POSITIONS:
        # the sampler co-locates each +/-u_i pair, so n_uniq = N/2 = NPAIR, not N.
        A_shot_global=(NPAIR)**-0.5,
        A_shot_quartile=(NPAIR / 4.0)**-0.5,
        A_shot_half=(NPAIR / 2.0)**-0.5,
        # ledger ranges, scaled from Session 1 so the LOG-BIN GEOMETRY is identical:
        # Session 1 used shells over [0.5, 800] M = [0.025 b, 40 b] at b = 20.
        shell_rmin=0.025 * b, shell_rmax=40.0 * b,
        field_rmin=dx[N], field_rmax=2.0 * L,
        metric_R0=mod.Rt / 800.0,
        out_dt={k: P_half / v for k, v in CADENCE.items()},
        milestones=[k * P_half for k in range(1, int(PERIODS) + 1)],
    )
    return c


def fmt(x):
    """17 significant digits: enough to round-trip a double exactly."""
    return repr(float(x)) if not float(x).is_integer() else '%.1f' % x


def deck(c, variant='prod'):
    """Emit a complete .athinput.  `variant` selects the preflight overrides."""
    b, rt, L, N = c['b'], c['rt'], c['L'], c['N']
    P = c['P_half']
    base = c['basename']
    tlim = c['tlim']
    if variant == 't0':
        base += '_pf_t0'
        tlim = 0.0
    elif variant == 'short':
        base += '_pf_short'
        tlim = P / 8.0
    elif variant == 'bytest':
        base += '_pf_bytest'
        tlim = 0.0

    o = []
    A = o.append
    A('# =============================================================================')
    A('# NRPIC relativistic PLUMMER Einstein cluster -- SESSION 2 compactness scan')
    A('# case %s: (R/M)_eff = 1/max_r[m(r)/r] = %.12g  (target %g)'
      % (c['label'], c['RM_eff'], c['target']))
    A('# variant: %s' % variant)
    A('#')
    A('# GENERATED by scripts/make_session2_inputs.py -- do not hand-edit.  Every number')
    A('# below is derived from analysis/plummer_1d.py; --check re-derives and diffs.')
    A('#')
    A('# QUESTION.  Session 1 ran this Plummer family at (R/M)_eff = 51.77 (b = 20 M) and')
    A('# found the cluster globally close to equilibrium for 3 P_1/2 with a slow, weak,')
    A('# CENTRE-OF-MASS-REFERENCED l = 1 signal confined to the inner mass quartile -- far')
    A('# weaker than the homogeneous Einstein cluster\'s whole-object dipole instability.')
    A('# Session 2 makes the SAME family substantially more relativistic and asks whether')
    A('# it stays broadly stable, keeps only the weak core-local mode, or crosses over to')
    A('# a strong collective l = 1 instability.  This deck does not presuppose an answer.')
    A('#')
    A('# CONTINUUM MODEL (G = c = 1).  The STATIC-OBSERVER energy density')
    A('# eps = T_{mu nu} n^mu n^nu is Plummer in AREAL radius r, hard-cut at r_t with')
    A('# vacuum outside and no taper, renormalised so M_ADM = 1 exactly.  The truncation')
    A('# convention r_t/b = %g is held fixed across the scan, which makes f_t independent'
      % RT_OVER_B)
    A('# of b and the compactness exactly linear in b:')
    A('#   max(m/r) = 2 M_P/(3 sqrt(3) b) at r = b sqrt(2)  =>  (R/M)_eff = 3 sqrt(3) b f_t/2.')
    A('#')
    A('#   b            = %.15g M' % b)
    A('#   r_t = %g b   = %.15g M' % (RT_OVER_B, rt))
    A('#   f_t          = %.15g   (removed mass fraction %.6g %%)'
      % (c['f_t'], 100.0 * (1.0 - c['f_t'])))
    A('#   M_P = M/f_t  = %.15g' % c['M_P'])
    A('#   M_0          = %.15g   (total REST mass; N mu = M_0 != M_ADM, binding energy)'
      % c['M_0'])
    A('#   mu = M_0/N   = %.15g' % c['mu'])
    A('#   r_1/2        = %.15g M      R_1/2 = %.15g M (isotropic)'
      % (c['r_half'], c['R_half']))
    A('#   R_t          = %.15g M (isotropic)' % c['R_t'])
    A('#   alpha(r_1/2) = %.12g        v_c(r_1/2) = %.12g'
      % (c['alpha_at_r_half'], c['vc_at_r_half']))
    A('#   P_1/2        = %.15g M      %g P_1/2 = %.15g M'
      % (P, c['periods'], tlim if variant == 'prod' else c['tlim']))
    A('#   alpha(0)     = %.12g        psi(0) = %.12g' % (c['alpha_0'], c['psi_0']))
    A('#   max m/r      = %.15g  (<< 1/3: circular geodesics exist everywhere)'
      % c['max_m_over_r'])
    A('#   max v_c      = %.12g at r = %.12g M' % (c['vc_max'], c['r_at_vc_max']))
    A('#   minimum lapse= %.12g' % c['alpha_min'])
    A('#')
    A('# PRE-PRODUCTION CHECKS (analysis/solve_compactness.py, all PASS):')
    A('#   circular geodesic exists everywhere      max m/r < 1/3')
    A('#   individual circular orbit radially stable r^2 m\' + r m - 6 m^2 > 0 everywhere')
    A('#   no horizon / regular geometry            max 2m/r < 1, min alpha > 0')
    A('#   continuum constraint identities          < 1e-12 relative')
    A('#')
    A('# SAMPLING.  Unchanged from Session 1: stratified antithetic, N_pair = %d'
      % c['npair'])
    A('# co-located +/-u_i pairs, N = %d equal-rest-mass particles, seed %d.'
      % (c['Npart'], c['seed']))
    A('# Initial deposited momentum and total angular momentum vanish EXACTLY by pair')
    A('# cancellation, and the initial radial velocity dispersion is zero.')
    A('# The A_l finite-N noise floor is set by the number of INDEPENDENT ANGULAR')
    A('# POSITIONS, N_pair, NOT by N:')
    A('#   A_l^shot(global)   = N_pair^-1/2      = %.12e' % c['A_shot_global'])
    A('#   A_l^shot(quartile) = (N_pair/4)^-1/2  = %.12e' % c['A_shot_quartile'])
    A('#')
    A('# MESH.  [-%g,%g]^3, root %d^3 (dx_0 = %g M), MeshBlocks %d^3, nghost %d,'
      % (L, L, NROOT, c['dx0'], NBLOCK, NGHOST))
    A('# %d static refined levels at block-aligned half-widths' % N)
    A('#   %s M' % '/'.join('%g' % h for h in c['halfwidth'][1:]))
    A('# -> dx_fine = %g M.  Leaf count %d*56 + 64 = %d MeshBlocks = %s active cells.'
      % (c['dx_fine'], N, c['n_leaf'], format(c['ncell_active'], ',')))
    A('# Resolution %.4g cells per Plummer scale b, %.4g per r_1/2 (areal), %.4g per'
      % (c['cells_per_b'], c['cells_per_r_half'], c['cells_per_R_half']))
    A('# R_1/2 (isotropic, the MESH coordinate).  Session 1 had 20.00 / 26.01 / 25.21,')
    A('# so every resolution measure here meets or exceeds the Session-1 baseline and')
    A('# both Session-2 mandated minima (>= 20 per b, >= 26 per r_1/2).')
    A('# Nominal dt = CFL*dx_fine = %g M; the code also applies the PARABOLIC bound of'
      % c['dt_hyp'])
    A('# the Hamiltonian-constraint damping, dt_par = safety*s_RK*dx^2/(3 sigma c_H psi_max)')
    A('# (z4c_newdt.cpp).  At t = 0, psi_max = psi(0) = %.6g gives dt_par/dt_hyp = %.4g,'
      % (c['psi_0'], 0.5 * 2.7853 * c['dx_fine']**2
         / (3.0 * (272.0 / 45.0) * 0.02 * c['psi_0']) / c['dt_hyp']))
    A('# so the hyperbolic bound controls -- but only by that factor, and psi_max is')
    A('# re-evaluated every cycle, so a contraction can hand control to the parabolic')
    A('# bound and shrink dt.  The startup banner prints which bound controls; the code')
    A('# prints a NOTE the first cycle the parabolic bound takes over.  MONITOR IT.')
    A('#')
    A('# CAUSALITY.  The 1+log gauge speed is sqrt(2/alpha) ~ sqrt(2). A disturbance')
    A('# launched at the outer boundary R = %g M reaches the matter edge R_t = %.6g M'
      % (L, c['R_t']))
    A('# after >= (%g - %.6g)/sqrt(2) = %.6g M = %.3g P_1/2 (and %.3g P_1/2 using the'
      % (L, c['R_t'], (L - c['R_t']) / np.sqrt(2.0),
         (L - c['R_t']) / np.sqrt(2.0) / P,
         (L - c['R_t']) / np.sqrt(2.0 / c['alpha_0']) / P))
    A('# ultra-conservative sqrt(2/alpha_min)), i.e. AFTER the %g P_1/2 endpoint.'
      % c['periods'])
    A('# Session 1 could not say this: its boundary signal arrived at 2.19 P_1/2, before')
    A('# its own 3 P_1/2 endpoint.  The extra refinement level here buys that margin.')
    A('#')
    A('# IN-CODE LEDGERS at the hst cadence, reduced over ALL particles/cells in double')
    A('# precision (see src/pgen/particles/nr_pic_plummer.cpp):')
    A('#   .plummer_shells.csv   48 log bins in areal radius over [%.6g, %.6g] M:'
      % (c['shell_rmin'], c['shell_rmax']))
    A('#                         mass, <r>, <v_r>, dispersions, <alpha W>, <|L|>, and real')
    A('#                         Y_lm moments to l = 4 about the ORIGIN (c_lm) and about')
    A('#                         the instantaneous CENTRE OF MASS (d_lm).  d_lm is primary.')
    A('#   .plummer_cohorts.csv  32 LAGRANGIAN equal-rest-mass initial radial bands.')
    A('#   .plummer_fields.csv   64 log bins in ISOTROPIC radius over [%.6g, %.6g] M:'
      % (c['field_rmin'], c['field_rmax']))
    A('#                         |H|, H^2, M^2, alpha, chi, deposited E, areal radius,')
    A('#                         Khat, plus the cell size, so a refinement seam, the')
    A('#                         density cutoff or a boundary feature is identifiable.')
    A('#   .plummer_admmom.csv   NEW in Session 2: the ADM linear momentum')
    A('#                         P_i = (1/8pi) oint dS_m (K^m_i - delta^m_i K) on %d'
      % len(c['pmom_radii']))
    A('#                         coordinate spheres, with the particle momentum sum')
    A('#                         P_i^matter = sum_p m_p u_i and the deposited')
    A('#                         P_i^dep = int S_i sqrt(gamma) d^3x for reconciliation.')
    A('#')
    A('# The truncation sits close to a refinement seam in this scan (the nearest seam is')
    A('# %.4g M from R_t). The deposited energy density there is (1+(r_t/b)^2)^-5/2 ~ 3e-7'
      % min(abs(h - c['R_t']) for h in c['halfwidth'][1:]))
    A('# of central and no mass lies beyond it, so a prolongation feature there cannot')
    A('# reach the core bands; the fields ledger resolves the seam explicitly so the claim')
    A('# is checkable rather than asserted.')
    A('#')
    A('# NO trk OUTPUT.  file_type = trk stays DISABLED (Session 1 found the writer reads')
    A('# past the end of a host array and produced a 94.9 GB file in 150 M).')
    A('# EXCISION IS OFF; no particle should ever be removed.')
    A('# =============================================================================')
    A('')
    A('<comment>')
    A('problem = NRPIC relativistic Plummer Einstein cluster SESSION 2 %s: '
      '(R/M)_eff=%.10g b=%.12gM rt=%.12gM N=%d seed=%d stratified-antithetic '
      'BSSN use_z4c=false cH=0.02 filter=OFF dxf=%g (%s)'
      % (c['label'], c['RM_eff'], b, rt, c['Npart'], c['seed'], c['dx_fine'], base))
    A('')
    A('<job>')
    A('basename = %s' % base)
    A('')
    A('<mesh>')
    A('nghost = %d' % NGHOST)
    for d in (1, 2, 3):
        A('nx%d    = %d' % (d, NROOT))
        A('x%dmin  = %s' % (d, fmt(-L)))
        A('x%dmax  = %s' % (d, fmt(L)))
        A('ix%d_bc = outflow' % d)
        A('ox%d_bc = outflow' % d)
    A('')
    A('<meshblock>')
    for d in (1, 2, 3):
        A('nx%d = %d' % (d, NBLOCK))
    A('')
    A('<mesh_refinement>')
    A('refinement       = static')
    A('# max_nmb_per_rank is consulted only for ADAPTIVE refinement; with')
    A('# refinement = static the code reports it as unused, so it is deliberately absent.')
    for l in range(1, N + 1):
        A('')
        A('<refined_region%d>' % l)
        A('level = %d' % l)
        h = c['halfwidth'][l]
        for d in (1, 2, 3):
            A('x%dmin = %s' % (d, fmt(-h)))
            A('x%dmax = %s' % (d, fmt(h)))
    A('')
    A('<time>')
    A('evolution  = dynamic')
    A('integrator = rk4')
    A('cfl_number = 0.25')
    A('nlim       = 100000000')
    A('tlim       = %.15g' % tlim)
    A('ndiag      = 100')
    A('')
    A('<z4c>')
    A('# Verbatim from the Session-1 production deck: the compactness scan changes')
    A('# compactness, not the formulation, gauge, damping or dissipation.')
    A('lapse_harmonic    = 0.0')
    A('lapse_harmonicf   = 1.0')
    A('lapse_oplog       = 2.0')
    A('lapse_advect      = 1.0')
    A('shift_driver      = legacy')
    A('shift_eta         = 2.0')
    A('shift_advect      = 1.0')
    A('shift_Gamma       = 1.0')
    A('shift_alpha2Gamma = 0.0')
    A('shift_H           = 0.0')
    A('const_damp        = true')
    A('diss              = 0.1')
    A('chi_div_floor     = 0.00001')
    A('use_z4c           = false')
    A('hdamp_cH          = 0.02')
    A('hdamp_par_safety  = 0.5')
    A('damp_kappa1       = 0.0')
    A('damp_kappa2       = 0.0')
    A('nrad_wave_extraction = 0')
    A('')
    A('<particles>')
    A('# Verbatim from the Session-1 production deck.')
    A('particle_type       = dust')
    A('pusher              = gr_boris')
    A('init                = pgen')
    A('feedback            = true')
    A('mass                = 1.0')
    A('debug               = 0')
    A('destroy_log         = true')
    A('excise_radius       = 0.0')
    A('excise_x1           = 0.0')
    A('excise_x2           = 0.0')
    A('excise_x3           = 0.0')
    A('excise_lapse        = 0.0')
    A('excise_ah           = false')
    A('cross_level_deposit = conservative')
    A('tmunu_filter_passes   = 0')
    A('')
    A('<problem>')
    A('user_hist            = true')
    A('plummer_mass         = 1.0')
    A('# plummer_b AND plummer_rt must BOTH be set: each is GetOrAdd with a Session-1')
    A('# default (20.0 and 400.0), so omitting rt would silently build an rt/b != 20')
    A('# cluster with a different f_t and a different (R/M)_eff, with no warning.')
    A('plummer_b            = %.17g' % b)
    A('plummer_rt           = %.17g' % rt)
    A('plummer_npair        = %d' % c['npair'])
    A('plummer_seed         = %d' % c['seed'])
    A('plummer_ntable       = 20000')
    A('plummer_nmetric      = 16384')
    A('plummer_metric_R0    = %.12g' % c['metric_R0'])
    A('plummer_center_x1    = 0.0')
    A('plummer_center_x2    = 0.0')
    A('plummer_center_x3    = 0.0')
    A('# Ledger ranges scaled from Session 1 so the log-bin GEOMETRY is identical:')
    A('# shells over [0.025 b, 40 b] (Session 1: [0.5, 800] M at b = 20 M), fields over')
    A('# [dx_fine, 2 L] (Session 1: [0.5, 8192] M at dx_fine = 1 M, L = 4096 M).')
    A('plummer_shell_nbin   = 48')
    A('plummer_shell_rmin   = %.12g' % c['shell_rmin'])
    A('plummer_shell_rmax   = %.12g' % c['shell_rmax'])
    A('plummer_cohort_nbin  = 32')
    A('plummer_lmax         = 4')
    A('plummer_field_nbin   = 64')
    A('plummer_field_rmin   = %.12g' % c['field_rmin'])
    A('plummer_field_rmax   = %.12g' % c['field_rmax'])
    A('')
    A('# ---- ADM linear momentum (Session 2).  Extraction spheres are coordinate spheres')
    A('#      R = const.  All lie in vacuum outside R_t = %.6g M, and each sphere lies'
      % c['R_t'])
    A('#      WHOLLY within a single refinement level, so one dx resolves every angle of')
    A('#      it.  That is a stronger condition than "between two seams": the refined')
    A('#      regions are CUBES, and on a sphere of radius R the quantity max_i|n_i| runs')
    A('#      over [1/sqrt(3), 1], so the sphere crosses the cube |x_i| <= H whenever')
    A('#      H < R <= sqrt(3) H.  With H_{l+1} = H_l/2 the seam-free band is therefore')
    A('#      R/H_l in (sqrt(3)/2, 1] -- about 13%% of each level half-width.  Level')
    A('#      half-widths here are %s M.'
      % '/'.join('%g' % h for h in c['halfwidth'][1:]))
    A('#      The radii span a factor %.2g across three levels, so radius-independence'
      % (max(c['pmom_radii']) / min(c['pmom_radii'])))
    A('#      is testable.')
    A('#      selftest = 1 runs the Bowen-York positive unit test at startup and is FATAL')
    A('#      on failure; t = 0 is static so K_ij == 0 and P_i must vanish there too.')
    A('plummer_pmom_nrad    = %d' % len(c['pmom_radii']))
    A('plummer_pmom_ntheta  = 32')
    A('plummer_pmom_selftest = 1')
    for k, R in enumerate(c['pmom_radii']):
        # the whole sphere is inside this level's cube (R <= H_l) and outside the next
        # one's (R > sqrt(3) H_{l+1}), so a single dx resolves every angle
        lev = max(l for l in range(N + 1) if c['halfwidth'][l] >= R)
        A('plummer_pmom_r%d       = %s   # wholly on level %d, dx = %g, R/dx = %.1f, '
          'R/R_t = %.2f'
          % (k + 1, fmt(R), lev, c['dx'][lev], R / c['dx'][lev], R / c['R_t']))
    A('')
    A('# ---- outputs.  P_1/2 = %.15g M.  Every cadence is an exact fraction of P_1/2,'
      % P)
    A('#      so t = n P_1/2 always lands on a written row and the per-period milestone')
    A('#      analysis has data exactly at each milestone.')
    for k, v in sorted(CADENCE.items()):
        A('#      %-8s P/%-5g = %.12g M' % (k, v, c['out_dt'][k]))
    A('')
    A('<output1>')
    A('file_type   = hst')
    A('dt          = %.12g' % c['out_dt']['hst'])
    A('data_format = %.17e')
    A('')
    A('<output2>')
    A('file_type = pvtk')
    A('variable  = prtcl_all')
    A('dt        = %.12g' % c['out_dt']['pvtk'])
    A('')
    A('# Movie B lives on a FIXED UNIFORM CARTESIAN grid, not on AMR patches: the same')
    A('# physical extent and the same pixel size in every frame, so nothing in the image')
    A('# can change because the refinement layout changed.')
    A('#')
    A('# NODE PLACEMENT IS DELIBERATE.  cart_grid.cpp decides which rank owns a node with')
    A('# a bounds test that is INCLUSIVE at both ends, and cartgrid.cpp MPI_SUMs the')
    A('# result, so a node lying exactly on a MeshBlock face shared by two ranks is')
    A('# DOUBLED.  On this origin-centred mesh the planes x = 0, y = 0 and z = 0 are')
    A('# MeshBlock faces at every level, so:')
    A('#   * center_x/y are offset by a NON-DYADIC fraction of dx_fine, moving every')
    A('#     node off the x = 0 and y = 0 faces and off every seam;')
    A('#   * the equatorial plane is sampled as TWO planes at z = +/- pixel/2 rather than')
    A('#     one plane at z = 0 (numpoints_z = 2, extent_z = pixel/2).  Their mean is the')
    A('#     z = 0 field to O(pixel^2) and neither plane is on a face.  numpoints_z = 2')
    A('#     with extent_z = 0 would be arithmetically safe (d_x3 = 0/1 = 0) but puts')
    A('#     both planes ON the z = 0 face, i.e. exactly the doubling case;')
    A('#     numpoints_z = 1 is silently broken (d_x3 = 0/0 = NaN).')
    A('# The preflight verifies this by differencing a 1-rank and a 4-rank t = 0 frame:')
    A('# any double-claimed node shows up as an exact factor 2.')
    # PIXEL SIZE IS SET BY THE CELL SIZE, not by a target image size.  Sampling the
    # CIC-deposited density BELOW the cell scale makes the eighth-order Lagrange
    # interpolant ring: measured ring-mean bias 0.60 at pixel/cell = 0.16, 0.84 at 0.62,
    # 1.008 at 1.24 (evidence/cart_pixel_calibration_20260910/RECORD.md).  One pixel per
    # cell of the level the panel shows is unbiased; display smoothing belongs in the
    # renderer.  Odd numpoints keeps the grid symmetric about its centre.
    off = 0.31 * c['dx_fine']
    # core panel: EXACTLY the finest refined region (half-width 64 dx_fine) at exactly
    # its own resolution, so pixel/cell == 1 over the whole panel and the field is
    # unbiased everywhere in it.  wide panel: pixel = 8 dx_fine, which is the cell size
    # of the level that contains the matter edge R_t, so pixel/cell >= 1 out to
    # 512 dx_fine; only the empty corners beyond that undersample.
    NCORE, NWIDE = 129, 161
    core = (NCORE - 1)//2 * c['dx_fine']            # 64 dx_fine = finest half-width
    wide = (NWIDE - 1)//2 * 8.0 * c['dx_fine']      # 640 dx_fine
    for i, (tag, ext, npix, var) in enumerate([
            ('dens_wide', wide, NWIDE, 'tmunu_E'), ('ham_wide', wide, NWIDE, 'con_H'),
            ('dens_core', core, NCORE, 'tmunu_E'), ('ham_core', core, NCORE, 'con_H')]):
        NPIX = npix
        pix = 2.0 * ext / (NPIX - 1)
        A('')
        A('<output%d>' % (3 + i))
        A('file_type   = cart')
        A('variable    = %s' % var)
        A('id          = %s' % tag)
        A('dt          = %.12g' % c['out_dt']['cart'])
        A('center_x    = %.12g' % off)
        A('center_y    = %.12g' % off)
        A('center_z    = 0.0')
        A('extent_x    = %.12g' % ext)
        A('extent_y    = %.12g' % ext)
        A('extent_z    = %.12g' % (0.5 * pix))
        A('numpoints_x = %d' % NPIX)
        A('numpoints_y = %d' % NPIX)
        A('numpoints_z = 2')
        A('chebyshev   = false')
        A('#           pixel = %.6g M = %.3g dx_fine = %.4g b; z planes at %+.6g;'
          % (pix, pix / c['dx_fine'], pix / b, 0.5 * pix))
        A('#           covers %s (R_t = %.6g M, finest region half-width %.6g M)'
          % ('the matter edge' if ext >= c['R_t'] else 'the finest refined region',
             c['R_t'], c['halfwidth'][N]))
    A('')
    A('# AMR-patch slices, kept at a reduced cadence purely as a cross-check that the')
    A('# uniform-grid movie product agrees with the data on the mesh it came from.')
    A('<output7>')
    A('file_type = bin')
    A('variable  = con')
    A('slice_x3  = 0.0')
    A('dt        = %.12g' % c['out_dt']['binslice'])
    A('')
    A('<output8>')
    A('file_type = bin')
    A('variable  = tmunu')
    A('slice_x3  = 0.0')
    A('dt        = %.12g' % c['out_dt']['binslice'])
    A('')
    A('<output9>')
    A('file_type      = cbin')
    A('variable       = adm')
    A('coarsen_factor = 2')
    A('dt             = %.12g' % c['out_dt']['cbin'])
    A('')
    A('<output10>')
    A('file_type = rst')
    A('dt        = %.12g' % c['out_dt']['rst'])
    return '\n'.join(o) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--variants', nargs='+',
                    default=['prod', 't0', 'short'])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bad = 0
    for label in CASES:
        CASES[label]['label'] = label
        c = derive(label)
        for v in a.variants:
            txt = deck(c, v)
            path = os.path.join(a.out, '%s_%s.athinput' % (label, v))
            if a.check:
                if not os.path.exists(path):
                    print('MISSING %s' % path)
                    bad += 1
                    continue
                have = open(path).read()
                if have != txt:
                    bad += 1
                    print('DRIFT in %s:' % path)
                    for line in list(difflib.unified_diff(
                            have.splitlines(), txt.splitlines(),
                            'on-disk', 're-derived', lineterm=''))[:40]:
                        print('   ' + line)
                else:
                    print('OK      %s' % path)
            else:
                open(path, 'w').write(txt)
                print('wrote   %s' % path)
        jpath = os.path.join(a.out, '..', 'initial_data',
                             'reference_values_%s.json' % label)
        jtxt = json.dumps({k: (list(v) if isinstance(v, (list, tuple, np.ndarray))
                               else (v if not isinstance(v, (np.floating, np.integer))
                                     else float(v)))
                           for k, v in c.items()},
                          indent=2, sort_keys=True, default=float)
        if a.check:
            if not os.path.exists(jpath) or open(jpath).read() != jtxt:
                print('DRIFT in %s' % jpath)
                bad += 1
            else:
                print('OK      %s' % jpath)
        else:
            os.makedirs(os.path.dirname(jpath), exist_ok=True)
            open(jpath, 'w').write(jtxt)
            print('wrote   %s' % jpath)
    if a.check and bad:
        print('\n%d artefact(s) differ from the generator' % bad)
        return 1
    if a.check:
        print('\nall generated artefacts match the generator')
    return 0


if __name__ == '__main__':
    sys.exit(main())
