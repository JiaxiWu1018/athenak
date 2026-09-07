#!/usr/bin/env python3
"""Reference implementation and self-test for the relativistic Plummer Einstein-cluster
initial data used by src/pgen/particles/nr_pic_plummer.cpp.

This module is the INDEPENDENT double-precision cross-check of the C++ construction in
src/pgen/particles/plummer_profile.hpp and of the in-pgen particle sampler.  It shares no
code with them; agreement between the two is the validation.

Usage
    python3 plummer_id.py --selftest              # 1D construction vs published values
    python3 plummer_id.py --sampler [--npair N]   # sampler checks (needs ~1 GB for the
                                                  # production N)
    python3 plummer_id.py --table out.txt         # dump the 1D table for comparison with
                                                  # the pgen's startup banner
Run headless (MPLBACKEND=Agg); no plotting is done here.
"""
import argparse
import json
import sys
import numpy as np

from plummer_1d import PlummerModel, PlummerInfinite
from plummer_sampler import build_table, hash_unit_id, splitmix64

# published reference values (campaign prompt + NRPIC_Plummer_initial_data.pdf)
REF_TRUNC = {
    "M_P": 1.003752342774352,
    "M0_over_M": 1.00738,          # quoted as approximate
    "r_half": 26.00761423,
    "R_half": 25.20934281,
    "R_t": 398.99937343,
    "alpha_rhalf": 0.96911272,
    "vc_rhalf": 0.14139982,
    "P_half": 1192.496781,
}
REF_INF = {                        # infinite model at M_P/b = 0.05 (PDF p. 4)
    "alpha_0": 0.950003710,
    "psi_0": 1.025808423,
    "M0_over_MP": 1.007325025,
    "vc_max": 0.141475801,
}


def selftest(verbose=True):
    ok = True
    P = PlummerModel(M=1.0, b=20.0, rt=400.0, npanel=20000, ngl=20)
    Ph, rh, ah, vh = P.P_half()
    got = {
        "M_P": P.MP,
        "M0_over_M": P.M0/P.M,
        "r_half": rh,
        "R_half": float(P.R_of_r(np.array([rh]))[0]),
        "R_t": P.Rt,
        "alpha_rhalf": ah,
        "vc_rhalf": vh,
        "P_half": Ph,
    }
    tol = {"M_P": 1e-13, "M0_over_M": 1e-4, "r_half": 1e-8, "R_half": 1e-8,
           "R_t": 1e-10, "alpha_rhalf": 1e-7, "vc_rhalf": 1e-7, "P_half": 1e-8}
    if verbose:
        print("== truncated model  M=1, b=20, r_t=400 ==")
    for k, ref in REF_TRUNC.items():
        rel = abs(got[k] - ref)/abs(ref)
        good = rel < tol[k]
        ok &= good
        if verbose:
            print(f"  {k:12s} {got[k]:.12f}  ref {ref:.12f}  rel {rel:.2e}"
                  f"  {'OK' if good else '*** MISMATCH ***'}")

    # continuum constraint residuals
    rs = np.concatenate([np.geomspace(1e-8, 1.0, 80), np.geomspace(1.0, 399.999, 600)])
    rH, rL, sH, sL = P.constraint_residuals(rs)
    resH = float(np.max(np.abs(rH)/sH))
    resL = float(np.max(np.abs(rL)/sL))
    ok &= (resH < 1e-9 and resL < 1e-9)
    if verbose:
        print(f"  max rel |Lap psi + 2pi psi^5 eps|            = {resH:.2e}")
        print(f"  max rel |Lap(a psi) - 2pi a psi^5 (eps+2S)|  = {resL:.2e}")

    # occupancy / existence
    rr = P.r_tab[1:]
    mr = float(np.max(P.m(rr)/rr))
    st = float(np.min(P.radial_stability(rr)))
    vmax, rvmax = P.vc_max()
    ok &= (mr < 1.0/3.0 and st > -1e-12)
    if verbose:
        print(f"  max m/r = {mr:.9f} (< 1/3)   min(r^2 m'+rm-6m^2) = {st:.3e} (> 0)")
        print(f"  v_c,max = {vmax:.12f} at r = {rvmax:.9f} M  (= b sqrt(2) = "
              f"{20.0*np.sqrt(2):.9f})")

    if verbose:
        print("== infinite model  M_P/b = 0.05  (PDF p.4) ==")
    Q = PlummerInfinite(MP=1.0, b=20.0, rmax=1e8, npanel=60000, ngl=20)
    vq, _ = Q.vc_max()
    gotq = {"alpha_0": float(Q.alpha_tab[0]), "psi_0": float(Q.psi_tab[0]),
            "M0_over_MP": Q.M0/Q.MP, "vc_max": vq}
    for k, ref in REF_INF.items():
        rel = abs(gotq[k] - ref)/abs(ref)
        good = rel < 1e-8
        ok &= good
        if verbose:
            print(f"  {k:12s} {gotq[k]:.12f}  ref {ref:.12f}  rel {rel:.2e}"
                  f"  {'OK' if good else '*** MISMATCH ***'}")

    # RNG: canonical SplitMix64 finalizer values
    s0 = int(splitmix64(np.uint64(0)))
    s1 = int(splitmix64(np.uint64(1)))
    rng_ok = (s0 == 0xe220a8397b1dcdaf and s1 == 0x910a2dec89025cc1)
    ok &= rng_ok
    if verbose:
        print(f"  SplitMix64(0) = {s0:#018x}  SplitMix64(1) = {s1:#018x}"
              f"  {'OK' if rng_ok else '*** MISMATCH ***'}")
    return ok, P


def sampler_check(npair=1056768, seed=1985, verbose=True):
    P = PlummerModel(npanel=20000, ngl=20)
    X, U, r = build_table(P, npair, seed)
    N = 2*npair
    mu = P.M0/N
    Rp = np.linalg.norm(X, axis=1)
    psi_p = P.psi_of_R(Rp)
    W = P.W(r)
    out = {}
    out["N"] = int(N)
    out["mu"] = float(mu)
    out["N_mu"] = float(N*mu)
    u2 = (psi_p[0::2]**-4)*np.sum(U[0::2]**2, axis=1)
    out["max_rel_u_norm_err"] = float(np.max(np.abs(u2 - (W**2 - 1.0))
                                             /np.maximum(W**2 - 1.0, 1e-300)))
    dot = np.sum(X[0::2]*U[0::2], axis=1)
    sc = np.linalg.norm(X[0::2], axis=1)*np.linalg.norm(U[0::2], axis=1)
    out["max_tangency"] = float(np.max(np.abs(dot)/sc))
    out["max_pair_u_residual"] = float(np.max(np.abs(U[0::2] + U[1::2])))
    out["net_momentum"] = float(np.linalg.norm(mu*np.sum(U, axis=0, dtype=np.float64)))
    out["net_angmom"] = float(np.linalg.norm(
        mu*np.sum(np.cross(X, U), axis=0, dtype=np.float64)))
    rs = np.sort(r)
    out["max_cdf_dev"] = float(np.max(np.abs(P.F0_exact(rs)
                                             - np.arange(1, npair+1)/npair)))
    out["stratification_bound"] = 1.0/npair
    out["r_monotone_in_k"] = bool(np.all(np.diff(r) > 0))
    n = X[0::2]/Rp[0::2][:, None]
    out["A1_t0"] = float(np.linalg.norm(n.mean(axis=0)))
    out["A_shot"] = float(npair**-0.5)
    com = X[0::2].mean(axis=0)
    out["Rcom_t0"] = float(np.linalg.norm(com))
    if verbose:
        for k, v in out.items():
            print(f"  {k:24s} = {v!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sampler", action="store_true")
    ap.add_argument("--npair", type=int, default=1056768)
    ap.add_argument("--seed", type=int, default=1985)
    ap.add_argument("--table", type=str, default=None)
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()
    result = {}
    ok = True
    if a.selftest or not (a.sampler or a.table):
        ok, P = selftest()
        result["selftest_pass"] = bool(ok)
    if a.sampler:
        result["sampler"] = sampler_check(a.npair, a.seed)
    if a.table:
        P = PlummerModel(npanel=20000, ngl=20)
        rr = np.geomspace(1e-4, 400.0, 4001)
        arr = np.column_stack([rr, P.m(rr), P.eps(rr), P.B(rr), np.sqrt(P.vc2(rr)),
                               P.W(rr), np.exp(P.Phi_exact(rr)),
                               np.exp(-0.5*P.j_exact(rr)), P.R_of_r(rr),
                               P.F0_exact(rr)])
        np.savetxt(a.table, arr,
                   header="r m eps B vc W alpha psi R F0", fmt="%.17g")
        print("wrote", a.table)
    if a.json:
        json.dump(result, open(a.json, "w"), indent=2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
