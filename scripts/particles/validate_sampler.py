#!/usr/bin/env python3
"""Validate the Plummer particle sampler against the continuum model."""
import sys, time, json
import numpy as np
from plummer_1d import PlummerModel
from plummer_sampler import build_table

NPAIR = 1056768
N = 2*NPAIR
SEED = 1985

t0 = time.time()
P = PlummerModel(npanel=20000, ngl=20)
X, U, r = build_table(P, NPAIR, SEED)
print(f"built {N} particles in {time.time()-t0:.1f} s")
mu = P.M0/N
print(f"mu = M_0/N = {mu:.17g}   (M_0 = {P.M0:.17g})")

Rp = np.linalg.norm(X, axis=1)          # per particle
psi_p = P.psi_of_R(Rp)                   # per particle
psi = psi_p[0::2]                        # per pair
W = P.W(r)
vc = np.sqrt(P.vc2(r))

print("\n--- 1. counts, masses ---")
print(f"  N               = {N}   (target 2113536, match {N==2113536})")
print(f"  N mu            = {N*mu:.15f}   = M_0 = {P.M0:.15f}")
print(f"  N mu / M_ADM    = {N*mu/P.M:.10f}   (must NOT be 1: binding energy)")

print("\n--- 2. velocity normalisation gamma^ij u_i u_j = W^2-1 ---")
g_inv = psi**-4                       # gamma^ij = psi^-4 delta^ij
u2_pair = g_inv*np.sum(U[0::2]**2, axis=1)
resid = u2_pair - (W**2 - 1.0)
print(f"  max |gamma^ij u_i u_j - (W^2-1)|              = {np.max(np.abs(resid)):.3e}")
print(f"  max rel                                       = {np.max(np.abs(resid)/np.maximum(W**2-1,1e-300)):.3e}")

print("\n--- 3. tangency x.u = 0 ---")
dot = np.sum(X[0::2]*U[0::2], axis=1)
scale = np.linalg.norm(X[0::2], axis=1)*np.linalg.norm(U[0::2], axis=1)
print(f"  max |x.u|/(|x||u|)                            = {np.max(np.abs(dot)/scale):.3e}")

print("\n--- 4. pair cancellation ---")
print(f"  max |u^+ + u^-| componentwise                 = {np.max(np.abs(U[0::2]+U[1::2])):.3e}")
print(f"  max |x^+ - x^-| componentwise                 = {np.max(np.abs(X[0::2]-X[1::2])):.3e}")
Ptot = mu*np.sum(U, axis=0, dtype=np.float64)
Jtot = mu*np.sum(np.cross(X, U), axis=0, dtype=np.float64)
uscale = mu*np.sum(np.abs(U[0::2]))
print(f"  |sum mu u_i|                                  = {np.linalg.norm(Ptot):.3e}  (scale {uscale:.3e})")
print(f"  |sum mu (x x u)|                              = {np.linalg.norm(Jtot):.3e}")

print("\n--- 5. radial CDF vs F0 (Kolmogorov-Smirnov style) ---")
rs = np.sort(r)
Fe = np.arange(1, NPAIR+1)/NPAIR
Fm = P.F0_exact(rs)
d = np.max(np.abs(Fm - Fe))
print(f"  max |F0(r_(k)) - k/Npair|                     = {d:.3e}   (stratified: <= 1/Npair = {1/NPAIR:.3e})")
print(f"  r range = [{r.min():.6f}, {r.max():.6f}]  (r_t = {P.rt})")
print(f"  monotone in pair index k                      = {bool(np.all(np.diff(r)>0))}")

print("\n--- 6. direction isotropy (per-pair n) ---")
nvec = X[0::2]/Rp[0::2][:, None]
mn = nvec.mean(axis=0)
print(f"  <n>            = {mn}   |<n>| = {np.linalg.norm(mn):.3e}  (1/sqrt(Npair) = {1/np.sqrt(NPAIR):.3e})")
Q = nvec.T @ nvec/NPAIR - np.eye(3)/3.0
print(f"  max |<n_i n_j> - delta_ij/3|                  = {np.max(np.abs(Q)):.3e}")

print("\n--- 7. deposited-source targets (continuum, exact, no grid) ---")
# ideal particle sums in shells: E = sum mu W /sqrt(gamma) ... compare shell-integrated
nb = 60
edges = np.geomspace(0.5, P.rt, nb+1)
idx = np.digitize(r, edges) - 1
ok = (idx >= 0) & (idx < nb)
cnt = np.bincount(idx[ok], minlength=nb)
# energy density: sum over particles in shell of mu*W divided by proper volume
Vshell = np.array([4*np.pi*np.trapezoid(
    (lambda s: s**2*P.B(s))(np.linspace(edges[i], edges[i+1], 400)),
    np.linspace(edges[i], edges[i+1], 400)) for i in range(nb)])
Esum = np.bincount(idx[ok], weights=2*mu*W[ok], minlength=nb)   # 2 per pair
rc = np.sqrt(edges[:-1]*edges[1:])
Enum = Esum/Vshell
Eref = P.eps(rc)
good = cnt > 400
rel = np.abs(Enum[good]/Eref[good] - 1.0)
print(f"  shells with >400 pairs: {good.sum()}/{nb}")
print(f"  max rel dev of shell-averaged E from eps(r)   = {rel.max():.3e}")
print(f"  median rel dev                                = {np.median(rel):.3e}")
print(f"  typical Poisson floor 1/sqrt(2*cnt)           = {np.median(1/np.sqrt(2*cnt[good])):.3e}")

print("\n--- 8. stress: radial vs tangential ---")
# S_ij = (1/sqrt(gamma)) sum mu u_i u_j / W ; radial component along n
un = np.sum(U*(X/np.maximum(np.linalg.norm(X,axis=1),1e-300)[:,None]), axis=1)
print(f"  max |u . nhat| (radial momentum, must be 0)   = {np.max(np.abs(un)):.3e}")
# tangential pressure check on a shell
sel = (r > 20) & (r < 30)
pair_sel = np.repeat(sel, 2)
# p_t = (1/2) * (1/proper volume) sum mu gamma^ij u_i u_j / W
loc = mu*np.sum(U[pair_sel]**2, axis=1)*(psi_p[pair_sel]**-4)/np.repeat(W[sel],2)
Vs = 4*np.pi*np.trapezoid((lambda s: s**2*P.B(s))(np.linspace(20,30,2000)), np.linspace(20,30,2000))
pt_num = 0.5*loc.sum()/Vs
rmid_w = np.trapezoid((lambda s: s**2*P.B(s)*P.pt(s))(np.linspace(20,30,2000)), np.linspace(20,30,2000))
pt_ref = 4*np.pi*rmid_w/Vs
print(f"  <p_t> particles (20<r<30)                     = {pt_num:.9e}")
print(f"  <p_t> continuum (20<r<30)                     = {pt_ref:.9e}")
print(f"  rel dev                                       = {abs(pt_num/pt_ref-1):.3e}")

print("\n--- 9. multipole sampling noise floor (mass-weighted, on the pair set) ---")
def Alm(pos, wts, lmax=4):
    x, y, z = pos[:,0], pos[:,1], pos[:,2]
    rr = np.sqrt(x*x+y*y+z*z); rr[rr==0]=1
    nx, ny, nz = x/rr, y/rr, z/rr
    Wtot = wts.sum()
    out = {}
    # l=1: dipole vector
    d = np.array([np.sum(wts*nx), np.sum(wts*ny), np.sum(wts*nz)])/Wtot
    out[1] = np.linalg.norm(d)
    # l=2..4: use real solid harmonics via power sums of direction cosines
    for l in (2,3,4):
        from scipy.special import sph_harm_y
        th = np.arccos(np.clip(nz,-1,1)); ph = np.arctan2(ny,nx)
        acc = 0.0
        for m in range(-l, l+1):
            c = np.sum(wts*np.conj(sph_harm_y(l, m, th, ph)))/Wtot
            acc += abs(c)**2
        out[l] = np.sqrt(4*np.pi/(2*l+1)*acc)
    return out
w = np.full(NPAIR, 2*mu)
A = Alm(X[0::2], w)
print("  whole cluster (all pairs):")
for l in (1,2,3,4):
    print(f"    A_{l} = {A[l]:.6e}    (1/sqrt(Npair) = {1/np.sqrt(NPAIR):.4e})")
inner = r < 26.007614229
Ai = Alm(X[0::2][inner], w[inner]); Ao = Alm(X[0::2][~inner], w[~inner])
print(f"  inner half (n={inner.sum()}):  A_1 = {Ai[1]:.4e}")
print(f"  outer half (n={(~inner).sum()}): A_1 = {Ao[1]:.4e}")
com = np.sum(w[:,None]*X[0::2], axis=0)/w.sum()
print(f"  centre of mass |R_com| = {np.linalg.norm(com):.6e} M")

json.dump(dict(N=int(N), mu=float(mu), M0=float(P.M0),
               A1_all=float(A[1]), A2_all=float(A[2]), A3_all=float(A[3]), A4_all=float(A[4]),
               A1_inner=float(Ai[1]), A1_outer=float(Ao[1]),
               com=[float(c) for c in com]),
          open('/data/jiaxiwu/NRPIC/Plummer-cluster/initial_data/sampler_validation.json','w'), indent=2)
print("\nDONE", time.time()-t0, "s")
