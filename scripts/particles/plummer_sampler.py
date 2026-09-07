#!/usr/bin/env python3
"""Deterministic particle sampler for the relativistic Plummer Einstein cluster.

Sampler "stratified antithetic" (the Plummer analogue of the homogeneous campaign's
sampler G):

  * N = 2 N_pair particles in co-located +/- velocity pairs.
  * pair k gets a STRATIFIED radial quantile  q_k = (k + xi_k)/N_pair,  xi_k ~ U(0,1),
    inverted through the relativistic rest-mass CDF F0 (Eq. 13 of the derivation),
    so the radial ordering is monotone in k and the tag encodes the initial radial group.
  * an INDEPENDENT uniform direction n on S^2 and an independent uniform tangent
    angle zeta per pair -- no antipodal mirroring, no octant symmetry, so odd
    (l = 1) density modes keep their natural sampling seed.
  * both members of a pair sit at the SAME Cartesian position with u_i^(-) = -u_i^(+),
    so the initial deposited momentum and total angular momentum cancel to roundoff.

Every draw comes from a stateless counter-based hash keyed by (seed, pair id, stream),
bit-identical to the C++ HashUnitId in nr_pic_homogeneous_cluster.cpp, so the sample is
independent of MPI decomposition, construction order, and language.

Row order in the emitted table is  row 2k = '+', row 2k+1 = '-',  and the AthenaK HDF5
reader sets the particle tag to the row index; therefore
    pair index      = tag // 2
    velocity sign   = +1 if tag % 2 == 0 else -1
    initial radial group = monotone function of tag // 2.
"""
import numpy as np

_GOLD = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31, _S11 = (np.uint64(30), np.uint64(27), np.uint64(31), np.uint64(11))
_TWO53 = np.float64(9007199254740992.0)


def splitmix64(x):
    """SplitMix64 finalizer, bit-identical to the C++ helper (wrapping uint64)."""
    with np.errstate(over='ignore'):
        x = np.asarray(x, dtype=np.uint64) + _GOLD
        x = (x ^ (x >> _S30)) * _M1
        x = (x ^ (x >> _S27)) * _M2
        return x ^ (x >> _S31)


def hash_unit_id(seed, idx, stream):
    """Counter-based uniform on [0,1) keyed by (seed, id, stream).

    C++: key = SplitMix64(seed + GOLD*(id+1)); key = SplitMix64(key ^ (M1*(stream+1)));
         return (key >> 11) / 2^53
    """
    with np.errstate(over='ignore'):
        idx = np.asarray(idx, dtype=np.uint64)
        key = splitmix64(np.uint64(seed) + _GOLD*(idx + np.uint64(1)))
        key = splitmix64(key ^ (_M1*np.uint64(int(stream) + 1)))
        return (key >> _S11).astype(np.float64)/_TWO53


# stream ids (fixed for reproducibility -- do not renumber)
S_XI, S_Z, S_PHI, S_ZETA = 0, 1, 2, 3


def sample_pairs(model, npair, seed=1985, chunk=1 << 20):
    """Yield (k0, x, u_plus, r, extras) chunks of PAIRS (not particles).

    x        : (n,3) Cartesian isotropic positions of the pair
    u_plus   : (n,3) covariant specific spatial momentum u_i of the '+' member
    r        : (n,)  areal radius of the pair
    """
    for k0 in range(0, npair, chunk):
        k = np.arange(k0, min(k0 + chunk, npair), dtype=np.uint64)
        n = k.size
        xi = hash_unit_id(seed, k, S_XI)
        q = (k.astype(np.float64) + xi)/np.float64(npair)
        r = model.invert_F0_exact(q)

        z = 2.0*hash_unit_id(seed, k, S_Z) - 1.0
        phi = 2.0*np.pi*hash_unit_id(seed, k, S_PHI)
        zeta = 2.0*np.pi*hash_unit_id(seed, k, S_ZETA)
        st = np.sqrt(np.maximum(0.0, 1.0 - z*z))
        nvec = np.empty((n, 3))
        nvec[:, 0] = st*np.cos(phi)
        nvec[:, 1] = st*np.sin(phi)
        nvec[:, 2] = z

        # Cartesian axis least aligned with n
        axis = np.argmin(np.abs(nvec), axis=1)
        khat = np.zeros((n, 3))
        khat[np.arange(n), axis] = 1.0
        e1 = np.cross(khat, nvec)
        e1 /= np.linalg.norm(e1, axis=1)[:, None]
        e2 = np.cross(nvec, e1)
        t = np.cos(zeta)[:, None]*e1 + np.sin(zeta)[:, None]*e2

        Rr = model.R_of_r(r)
        psi = np.exp(-0.5*model.j_exact(r))
        W = model.W(r)
        vc = np.sqrt(model.vc2(r))
        amp = psi*psi*W*vc
        x = Rr[:, None]*nvec
        up = amp[:, None]*t
        yield int(k0), x, up, r, dict(nvec=nvec, t=t, psi=psi, W=W, vc=vc, R=Rr, q=q)


def build_table(model, npair, seed=1985, chunk=1 << 20):
    """Materialise the full particle table (2*npair rows) in memory."""
    N = 2*npair
    X = np.empty((N, 3))
    U = np.empty((N, 3))
    R_areal = np.empty(npair)
    for k0, x, up, r, _ in sample_pairs(model, npair, seed, chunk):
        n = x.shape[0]
        i0 = 2*k0
        X[i0:i0 + 2*n:2] = x
        X[i0 + 1:i0 + 2*n:2] = x
        U[i0:i0 + 2*n:2] = up
        U[i0 + 1:i0 + 2*n:2] = -up
        R_areal[k0:k0 + n] = r
    return X, U, R_areal
