#!/usr/bin/env python3
"""t=0 validation of the deposited/particle radial structure against the continuum."""
import sys, numpy as np
sys.path.insert(0, '/data/jiaxiwu/NRPIC/Plummer-cluster/analysis')
from plummer_1d import PlummerModel

shell = sys.argv[1] if len(sys.argv) > 1 else \
    '/data/jiaxiwu/NRPIC/Plummer-cluster/runs/pf_t0_bis_a/bis_a_notrk.plummer_shells.csv'
cohort = shell.replace('_shells', '_cohorts')

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
    return _read_rows(path)

P = PlummerModel(npanel=20000, ngl=20)
mu = P.M0/2113536
NPAIR = 1056768

h, A = read_csv(shell)
c = {n: i for i, n in enumerate(h)}
t0 = A[A[:, c['time']] == 0.0]
print(f"shell ledger: {A.shape[0]} rows, {len(np.unique(A[:,c['time']]))} times, "
      f"{t0.shape[0]} bins at t=0")

rlo, rhi = t0[:, c['r_lo']], t0[:, c['r_hi']]
cnt, mass = t0[:, c['count']], t0[:, c['mass']]
mr, mvr, mvr2, mvt2 = t0[:, c['m_r']], t0[:, c['m_vr']], t0[:, c['m_vr2']], t0[:, c['m_vt2']]

print("\n--- global sums at t=0 ---")
print(f"  total count            = {cnt.sum():.0f}   (expect 2113536, match "
      f"{int(cnt.sum())==2113536})")
print(f"  total mass             = {mass.sum():.15f}  (M_0 = {P.M0:.15f}, rel "
      f"{abs(mass.sum()/P.M0-1):.2e})")
print(f"  sum m*v_r / M_0        = {mvr.sum()/mass.sum():.3e}   (must be ~0)")
print(f"  sqrt(sum m vr^2 / M_0) = {np.sqrt(max(mvr2.sum()/mass.sum(),0)):.3e}  (sigma_r, must be ~0)")
print(f"  sqrt(sum m vt^2 / M_0) = {np.sqrt(mvt2.sum()/mass.sum()):.6f}  (sigma_t)")

# continuum prediction for sigma_t: v_t = alpha psi^-2 v_c, mass-weighted by dM0
rr = np.geomspace(1e-3, 399.999, 40000)
w = P.dM0dr(rr)
al = np.exp(P.Phi_exact(rr)); ps = np.exp(-0.5*P.j_exact(rr))
vt = al*ps**-2*np.sqrt(P.vc2(rr))
sig_t_ref = np.sqrt(np.trapezoid(w*vt**2, rr)/np.trapezoid(w, rr))
print(f"  sigma_t continuum ref  = {sig_t_ref:.6f}   rel dev "
      f"{abs(np.sqrt(mvt2.sum()/mass.sum())/sig_t_ref-1):.2e}")

print("\n--- shell mass vs continuum rest mass in the same areal-radius band ---")
print(f"  {'r_lo':>9} {'r_hi':>9} {'count':>9} {'M_shell':>13} {'M_cont':>13} {'rel':>10} {'Poisson':>9}")
bad = 0
for i in range(len(rlo)):
    if cnt[i] < 200:
        continue
    Mc = P.M0*(P.F0_exact(np.array([min(rhi[i], 400.0)]))[0]
               - P.F0_exact(np.array([min(rlo[i], 400.0)]))[0])
    rel = mass[i]/Mc - 1.0 if Mc > 0 else float('nan')
    pois = 1.0/np.sqrt(cnt[i]/2.0)     # independent sites = pairs
    flag = '' if abs(rel) < 5*pois else '  <-- >5 sigma'
    if abs(rel) >= 5*pois:
        bad += 1
    print(f"  {rlo[i]:9.3f} {rhi[i]:9.3f} {cnt[i]:9.0f} {mass[i]:13.6e} {Mc:13.6e} "
          f"{rel:+10.3e} {pois:9.2e}{flag}")
print(f"  bins deviating by more than 5 Poisson sigma: {bad}")

print("\n--- A_l per radial band at t=0 (about the origin), vs the band's own shot floor ---")
nylm = 25
lmax = 4
print(f"  {'r_lo':>9} {'r_hi':>9} {'npair':>8} {'shot':>10}", end='')
for l in range(1, lmax+1):
    print(f" {'A'+str(l):>10}", end='')
print()
for i in range(len(rlo)):
    n = cnt[i]/2.0
    if n < 500:
        continue
    tot = mass[i]
    row = []
    for l in range(1, lmax+1):
        ssum = 0.0
        for m in range(-l, l+1):
            q = l*l + (l+m)
            coef = t0[i, c['c%d' % q]]/tot     # mass-weighted mean of Y_lm
            ssum += coef*coef
        row.append(np.sqrt(4*np.pi/(2*l+1)*ssum))
    print(f"  {rlo[i]:9.3f} {rhi[i]:9.3f} {n:8.0f} {n**-0.5:10.3e}", end='')
    for v in row:
        print(f" {v:10.3e}", end='')
    print()

print("\n--- cohort ledger at t=0 (initial radial groups) ---")
hc, C = read_csv(cohort)
cc = {n: i for i, n in enumerate(hc)}
c0 = C[C[:, cc['time']] == 0.0]
print(f"  {len(c0)} cohorts; total count {c0[:,cc['count']].sum():.0f}; "
      f"total mass {c0[:,cc['mass']].sum():.12f}")
print(f"  {'cohort':>7} {'count':>8} {'<r>':>10} {'r_expect':>10} {'rel':>10}")
ncoh = len(c0)
for i in range(0, ncoh, max(1, ncoh//12)):
    k = c0[i, cc['cohort']]
    n = c0[i, cc['count']]
    rbar = c0[i, cc['m_r']]/c0[i, cc['mass']]
    q0, q1 = k/ncoh, (k+1)/ncoh
    rr2 = P.invert_F0_exact(np.linspace(q0, q1, 4001))
    rexp = rr2.mean()
    print(f"  {k:7.0f} {n:8.0f} {rbar:10.4f} {rexp:10.4f} {rbar/rexp-1:+10.3e}")
