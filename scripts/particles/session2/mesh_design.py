#!/usr/bin/env python3
"""Session 2: quantitative evaluator for a self-similar nested static-refinement
mesh, in the exact family Session 1 used.

Mesh family (identical topology to Session 1, only the scale and level count vary):
    root grid nx^3 over [-L, L]^3            -> dx_0 = 2L/nx
    MeshBlock nb^3, nghost = 4
    N static refined levels, level l covering [-L/2^l, L/2^l]^3, dx_l = dx_0/2^l

Because a level-l MeshBlock spans nb*dx_l = (nb/nx) * 2L/2^l and the level-l region
half-width is L/2^l, each region is exactly 2*nx/nb level-(l-1) blocks across; with
nx = 128 and nb = 32 that is 4 blocks across at every level, so every region is
automatically MeshBlock-aligned for ANY L.  Leaf count is then
    n_leaf = N*(4^3 - 2^3) + 4^3 = 56 N + 64.
The finest region half-width is fixed at (nx/2) * dx_fine = 64 dx_fine, independent
of L and N -- the one rigid coupling in this family.

Cost model, calibrated against Session 1 (see --calibrate).
"""
import argparse
import json
import sys
from math import sqrt

sys.path.insert(0, str(__file__).rsplit('/', 1)[0])
import numpy as np
from plummer_1d import PlummerModel

# --- Session-1 production mesh, for parity comparisons -----------------------
S1 = dict(L=4096.0, nx=128, nb=32, N=6, b=20.0, rt=400.0)


def model_for(b, rt_over_b=20.0, npanel=40000, ngl=20):
    return PlummerModel(M=1.0, b=b, rt=rt_over_b * b, npanel=npanel, ngl=ngl)


def evaluate(L, N, b, *, nx=128, nb=32, cfl=0.25, nghost=4,
             hdamp_cH=0.02, hdamp_par_safety=0.5, rt_over_b=20.0,
             periods=5.0, mod=None, nvar_bytes=None):
    """Full quantitative report for one (L, N, b) mesh."""
    mod = mod or model_for(b, rt_over_b)
    rt = mod.rt
    P_half, r_half, alpha_half, vc_half = mod.P_half()
    R_half = float(mod.R_of_r(np.array([r_half]))[0])
    R_t = mod.Rt
    psi0 = float(np.exp(-0.5 * mod.j_exact(np.array([1e-12 * b]))[0]))
    alpha0 = float(np.exp(mod.Phi_exact(np.array([1e-12 * b]))[0]))

    dx0 = 2.0 * L / nx
    dx = [dx0 / 2**l for l in range(N + 1)]
    hw = [L] + [L / 2**l for l in range(1, N + 1)]
    dxf = dx[N]
    blocks_across = 2 * nx // nb          # per level, in parent blocks
    n_leaf = N * ((nx // nb)**3 - (nx // nb // 2)**3) + (nx // nb)**3
    # generic form for nx/nb = 4: N*(64-8)+64
    ncell_active = n_leaf * nb**3
    ncell_with_ghost = n_leaf * (nb + 2 * nghost)**3

    # --- timestep: hyperbolic vs parabolic H-damping bound (z4c_newdt.cpp) ---
    dt_hyp = cfl * dxf
    s_rk = 2.7853
    sig1d = {2: 4.0, 3: 16.0 / 3.0, 4: 272.0 / 45.0}[nghost]
    psi_max = psi0
    dt_par = hdamp_par_safety * s_rk * dxf * dxf / (3.0 * sig1d * hdamp_cH * psi_max)
    dt = min(dt_hyp, dt_par)
    par_margin = dt_par / dt_hyp          # >1 means hyperbolic controls
    # how much psi_max may grow before the parabolic bound binds
    psi_growth_headroom = par_margin

    # --- resolution metrics --------------------------------------------------
    cells_per_b = b / dxf
    cells_per_rhalf = r_half / dxf
    cells_per_Rhalf = R_half / dxf
    finest_hw = hw[N]
    hw_over_Rhalf = finest_hw / R_half
    # which level's region contains the matter edge R_t (isotropic)
    lev_Rt = max([l for l in range(N + 1) if hw[l] >= R_t], default=0)
    dx_at_Rt = dx[lev_Rt]
    # distance from R_t to the nearest enclosing seam, in cells of that level
    seam_gap_cells = (hw[lev_Rt] - R_t) / dx_at_Rt
    # Distance from the matter edge to the NEAREST refinement seam, in cells of the
    # coarser side.  The enclosing-seam distance above can look comfortable while an
    # INNER seam sits right under the density cutoff, which is what actually risks a
    # prolongation feature at the truncation discontinuity.
    seams = [(hw[l], max(dx[l], dx[l - 1] if l > 0 else dx[l]))
             for l in range(1, N + 1)]
    near = min(seams, key=lambda sd: abs(sd[0] - R_t))
    seam_nearest_M = abs(near[0] - R_t)
    seam_nearest_cells = seam_nearest_M / near[1]
    seam_nearest_at = near[0]

    # --- causality: inward gauge signal from the outer boundary --------------
    # 1+log lapse gauge speed sqrt(2/alpha).  Two bounds: alpha ~ 1 in the vacuum
    # the signal must cross (nominal), and alpha = alpha_min (ultra-conservative).
    v_nom = sqrt(2.0 / 1.0)
    v_cons = sqrt(2.0 / alpha0)
    t_arr_nom = (L - R_t) / v_nom
    t_arr_cons = (L - R_t) / v_cons
    # also the light-crossing bound for a matter signal
    t_arr_light = (L - R_t) / 1.0

    # --- cost ---------------------------------------------------------------
    tlim = periods * P_half
    ncycle = tlim / dt
    blocksteps = n_leaf * ncycle

    s1mod = model_for(S1['b'], rt_over_b=S1['rt'] / S1['b'])
    s1P = s1mod.P_half()[0]
    s1_dxf = 2.0 * S1['L'] / S1['nx'] / 2**S1['N']
    s1_leaf = S1['N'] * ((S1['nx'] // S1['nb'])**3 - (S1['nx'] // S1['nb'] // 2)**3) \
        + (S1['nx'] // S1['nb'])**3
    s1_cycles = 3.0 * s1P / (cfl * s1_dxf)
    s1_blocksteps = s1_leaf * s1_cycles

    out = dict(
        # inputs
        L=L, N=N, nx=nx, nb=nb, nghost=nghost, cfl=cfl, b=b, rt=rt,
        rt_over_b=rt_over_b, periods=periods,
        # continuum
        r_half=r_half, R_half=R_half, R_t=R_t, P_half=P_half,
        alpha_0=alpha0, psi_0=psi0,
        # mesh
        dx0=dx0, dx_per_level=dx, halfwidth_per_level=hw, dx_fine=dxf,
        blocks_across_per_level=blocks_across,
        n_leaf=n_leaf, ncell_active=ncell_active, ncell_with_ghost=ncell_with_ghost,
        # resolution
        cells_per_b=cells_per_b, cells_per_r_half=cells_per_rhalf,
        cells_per_R_half=cells_per_Rhalf,
        finest_halfwidth=finest_hw, finest_hw_over_R_half=hw_over_Rhalf,
        level_containing_R_t=lev_Rt, dx_at_R_t=dx_at_Rt,
        R_t_to_enclosing_seam_cells=seam_gap_cells,
        R_t_nearest_seam_M=seam_nearest_M,
        R_t_nearest_seam_cells=seam_nearest_cells,
        R_t_nearest_seam_at=seam_nearest_at,
        # timestep
        dt_hyp=dt_hyp, dt_par=dt_par, dt=dt,
        parabolic_margin=par_margin, psi_growth_headroom=psi_growth_headroom,
        dt_bound=('hyperbolic' if dt_hyp <= dt_par else 'PARABOLIC'),
        # causality
        t_boundary_arrival_nominal=t_arr_nom,
        t_boundary_arrival_conservative=t_arr_cons,
        t_boundary_arrival_light=t_arr_light,
        P_boundary_arrival_nominal=t_arr_nom / P_half,
        P_boundary_arrival_conservative=t_arr_cons / P_half,
        # cost
        tlim=tlim, ncycle=ncycle, ncycle_per_P=P_half / dt,
        blocksteps=blocksteps,
        blocksteps_rel_session1_3P=blocksteps / s1_blocksteps,
        # session-1 parity
        s1_cells_per_b=S1['b'] / s1_dxf,
        s1_cells_per_r_half=s1mod.r_half() / s1_dxf,
        s1_cells_per_R_half=float(s1mod.R_of_r(np.array([s1mod.r_half()]))[0]) / s1_dxf,
        s1_finest_hw_over_R_half=(S1['L'] / 2**S1['N'])
        / float(s1mod.R_of_r(np.array([s1mod.r_half()]))[0]),
        s1_n_leaf=s1_leaf, s1_blocksteps_3P=s1_blocksteps,
        s1_P_boundary_arrival=(S1['L'] - s1mod.Rt) / sqrt(2.0) / s1P,
    )
    return out


def verdicts(e):
    """Pass/fail against the Session-2 design requirements."""
    v = {}
    v['cells_per_b >= 20'] = (e['cells_per_b'] >= 20.0, e['cells_per_b'])
    v['cells_per_r_half >= 26'] = (e['cells_per_r_half'] >= 26.0, e['cells_per_r_half'])
    v['cells_per_R_half >= session1 (25.21)'] = (
        e['cells_per_R_half'] >= e['s1_cells_per_R_half'], e['cells_per_R_half'])
    v['finest region hw/R_half >= 2.0'] = (
        e['finest_hw_over_R_half'] >= 2.0, e['finest_hw_over_R_half'])
    v['boundary arrival > 5 P_half (nominal)'] = (
        e['P_boundary_arrival_nominal'] > 5.0, e['P_boundary_arrival_nominal'])
    v['boundary arrival > 5 P_half (conservative)'] = (
        e['P_boundary_arrival_conservative'] > 5.0,
        e['P_boundary_arrival_conservative'])
    v['hyperbolic dt controls'] = (e['dt_bound'] == 'hyperbolic',
                                   e['parabolic_margin'])
    v['parabolic margin >= 1.3'] = (e['parabolic_margin'] >= 1.3,
                                    e['parabolic_margin'])
    v['R_t inside a refined region'] = (e['level_containing_R_t'] >= 1,
                                        e['level_containing_R_t'])
    v['R_t >= 2 cells inside its seam'] = (
        e['R_t_to_enclosing_seam_cells'] >= 2.0,
        e['R_t_to_enclosing_seam_cells'])
    v['cost <= 1.5x session1 3P'] = (
        e['blocksteps_rel_session1_3P'] <= 1.5, e['blocksteps_rel_session1_3P'])
    return v


def show(e, name=''):
    print('=' * 78)
    print('MESH %s : L = %g, N = %d levels, b = %.9g, r_t = %.9g'
          % (name, e['L'], e['N'], e['b'], e['rt']))
    print('  root %d^3, dx_0 = %g, MeshBlock %d^3, nghost %d'
          % (e['nx'], e['dx0'], e['nb'], e['nghost']))
    print('  matter edge R_t = %.6g: inside level %d (dx %.6g), %.2f cells inside its '
          'outer seam; NEAREST seam at %.6g is %.4g M = %.2f coarse cells away'
          % (e['R_t'], e['level_containing_R_t'], e['dx_at_R_t'],
             e['R_t_to_enclosing_seam_cells'], e['R_t_nearest_seam_at'],
             e['R_t_nearest_seam_M'], e['R_t_nearest_seam_cells']))
    print('  level  half-width        dx')
    for l in range(e['N'] + 1):
        tag = ''
        if l == e['level_containing_R_t']:
            tag = '   <- contains R_t = %.6g' % e['R_t']
        if l == e['N']:
            tag += '   <- FINEST'
        print('   %2d    %12.6f   %10.7f%s'
              % (l, e['halfwidth_per_level'][l], e['dx_per_level'][l], tag))
    print('  leaf MeshBlocks %d, active cells %s, cells+ghosts %s'
          % (e['n_leaf'], format(e['ncell_active'], ','),
             format(e['ncell_with_ghost'], ',')))
    print('  resolution: %.3f cells/b, %.3f cells/r_1/2, %.3f cells/R_1/2'
          % (e['cells_per_b'], e['cells_per_r_half'], e['cells_per_R_half']))
    print('              (session 1: %.3f, %.3f, %.3f)'
          % (e['s1_cells_per_b'], e['s1_cells_per_r_half'], e['s1_cells_per_R_half']))
    print('  finest region half-width %.4f = %.3f R_1/2  (session 1: %.3f R_1/2)'
          % (e['finest_halfwidth'], e['finest_hw_over_R_half'],
             e['s1_finest_hw_over_R_half']))
    print('  dt: hyp %.8g, par %.8g -> dt = %.8g (%s controls, margin %.3f)'
          % (e['dt_hyp'], e['dt_par'], e['dt'], e['dt_bound'],
             e['parabolic_margin']))
    print('  P_1/2 = %.6f M, cycles/P = %.0f, tlim(%gP) = %.6f, cycles = %.0f'
          % (e['P_half'], e['ncycle_per_P'], e['periods'], e['tlim'], e['ncycle']))
    print('  boundary gauge-signal arrival: %.1f M = %.2f P_1/2 (nominal), '
          '%.2f P_1/2 (conservative)'
          % (e['t_boundary_arrival_nominal'], e['P_boundary_arrival_nominal'],
             e['P_boundary_arrival_conservative']))
    print('              (session 1 was %.2f P_1/2 for a 3 P_1/2 run)'
          % e['s1_P_boundary_arrival'])
    print('  cost: %.3g block-steps = %.3f x session-1 3P production'
          % (e['blocksteps'], e['blocksteps_rel_session1_3P']))
    print('  VERDICTS')
    v = verdicts(e)
    nfail = 0
    for k, (ok, val) in v.items():
        if not ok:
            nfail += 1
        print('    [%s] %-42s %.6g' % ('PASS' if ok else 'FAIL', k, val))
    print('  -> %d FAIL' % nfail)
    return nfail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b', type=float, required=True)
    ap.add_argument('--L', type=float, nargs='+', required=True)
    ap.add_argument('--N', type=int, nargs='+', default=[7])
    ap.add_argument('--periods', type=float, default=5.0)
    ap.add_argument('--name', default='')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    mod = model_for(a.b)
    res = {}
    for L in a.L:
        for N in a.N:
            e = evaluate(L, N, a.b, periods=a.periods, mod=mod)
            nf = show(e, '%s L=%g N=%d' % (a.name, L, N))
            e['nfail'] = nf
            e['verdicts'] = {k: [bool(o), float(val)] for k, (o, val) in verdicts(e).items()}
            res['L%g_N%d' % (L, N)] = e
    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(res, fh, indent=2, sort_keys=True, default=float)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
