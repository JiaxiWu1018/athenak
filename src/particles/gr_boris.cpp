//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_boris.cpp
//! \brief general-relativistic Boris particle pusher (Bacchini/Zou-style). One full-dt
//! update per cycle via a Strang split:
//!   (1) interpolate (B, v) and the metric at x^n; build the normal-observer tetrad; do a
//!       half-dt electromagnetic Boris kick in the local orthonormal frame;
//!   (2) implicitly advance the geodesic motion x^n -> x^{n+1}, u -> u^{+} with a
//!       fixed-point iteration over the time-and-space midpoint metric (forward-Euler
//!       fallback on failure; a substep whose interpolated geometry is invalid is
//!       rejected and not taken);
//!   (3) interpolate fields/metric at x^{n+1} and do the second half-dt EM kick.
//! The q=0 / no-MHD limit is the geodesic integrator (steps 1 and 3 are skipped). The
//! metric is read at two time levels: the *_last snapshots hold step n, the live arrays
//! hold step n+1 (for a static background the two coincide). Velocity slots
//! IPVX/IPVY/IPVZ store the covariant spatial 4-velocity u_i.
//!
//! MHD storage conventions (dyn_grmhd, see dyn_grmhd.cpp PrimToCon and the rsolvers):
//! bcc0 holds the DENSITIZED field B~^i = sqrt(gamma) B^i, and the w0 velocity slots
//! hold the PROJECTED 4-velocity utilde^i = W v^i (W = Lorentz factor, v = Valencia
//! velocity). The EM kicks below undensitize B and divide out W before forming the
//! normal-frame ideal-MHD field E_i = -sqrt(gamma) eps_ijk v^j B^k and the tetrad
//! components.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles.hpp"
#include "boris_utils.hpp"
#include "lagrange_interp.hpp"
#include "calc_tetrad.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/primitive-solver/geom_math.hpp"

namespace particles {

//----------------------------------------------------------------------------------------
//! \struct GeodesicPush
//! \brief functor evaluating one implicit geodesic substep.
//! operator()(xin,uin,xout,uout,Euler) returns the updated (x,u) given a trial (x,u): it
//! interpolates the metric and its spatial derivatives at the space-time midpoint and
//! applies the 3+1 geodesic equations. Euler=true evaluates at (x^n, step n) for the
//! forward-Euler fallback. It returns FALSE when the interpolated geometry is not a
//! valid Riemannian 3-metric, in which case (xout,uout) must not be used.
//!
//! gamma_ij is positive definite by construction, but the value used here is a 2*NG-node
//! Lagrange interpolation, which can overshoot into negative diagonal components where
//! gamma_ij varies steeply (inside a moving-puncture trumpet). Such a gamma yields
//! usq = gamma^{ij} u_i u_j < -1 and hence a NaN W, or a finite but meaningless W from a
//! garbage u^i. det gamma > 0 is NOT a sufficient test: two negative eigenvalues cancel
//! in the determinant. The test is therefore Sylvester's criterion on the three leading
//! principal minors, which is exactly positive-definiteness.
//!
//! TRI selects the interpolation operator and nothing else: with it set, every geometric
//! quantity below -- lapse, shift, gamma_ij, gamma^{ij} and all their spatial derivatives
//! -- comes from the two-node-per-direction interpolant over the eight cell centres
//! surrounding the point instead of the 2*NG-node stencil. The 3+1 formulae, the old/new
//! time averaging and the Euler branch are untouched, so the retry solves the same
//! discrete problem with a different interpolation operator. Trilinear weights lie in
//! [0,1] and sum to one, so the interpolated gamma_ij is a convex combination of the
//! eight corner matrices and is positive definite whenever they are; that is what the
//! wide stencil, whose weights alternate in sign, cannot promise.
//!
//! TRI is a TEMPLATE PARAMETER and the branches below are `if constexpr`, which is
//! load-bearing rather than stylistic. operator() is inlined wholesale into whichever
//! device kernel calls it, and it carries an 8-node stencil plus a per-node
//! inverse-metric stencil over the same nodes. If one kernel has to hold both operators
//! live the per-thread frame stops fitting in registers on gfx90a and spills to scratch;
//! measured on 4 x MI210, a version that selected the operator at runtime ran more than
//! three orders of magnitude slower than this one -- even on substeps that never took
//! the fallback. Each kernel must instantiate exactly one operator.

template <int NG, bool TRI>
struct GeodesicPush {
  const Real *x_old, *u_old, *mb_par;
  const int *ncell;
  const DvceArray5D<Real> adm_old, adm_new, z4c_old, z4c_new;
  const int mb;
  const Real dt;
  const bool use_z4c;

  KOKKOS_INLINE_FUNCTION
  GeodesicPush(const Real x_[3], const Real u_[3], const int mb_, const Real mb_par_[9],
               const int ncell_[3], const Real dt_,
               const DvceArray5D<Real>& adm_old_, const DvceArray5D<Real>& adm_new_,
               const bool use_z4c_,
               const DvceArray5D<Real>& z4c_old_, const DvceArray5D<Real>& z4c_new_)
    : x_old(x_), u_old(u_), mb_par(mb_par_), ncell(ncell_),
      adm_old(adm_old_), adm_new(adm_new_), z4c_old(z4c_old_), z4c_new(z4c_new_),
      mb(mb_), dt(dt_), use_z4c(use_z4c_) {}

  //! The only place the interpolation operator enters. The discarded branch is not
  //! instantiated, so <NG,false> carries no trilinear code and generates exactly what the
  //! pusher generated before the fallback existed.
  KOKKOS_INLINE_FUNCTION
  Real Interp(const DvceArray5D<Real>& u0, const int nvar, const int *idcs,
              const Real *Wx, const Real *Wy, const Real *Wz) const {
    if constexpr (TRI) {
      return TrilinearInterpolator<NG>(u0, nvar, idcs, Wx, Wy, Wz);
    } else {
      return LagrangeInterpolator<NG>(u0, nvar, idcs, Wx, Wy, Wz);
    }
  }
  KOKKOS_INLINE_FUNCTION
  void Interp(const DvceArray5D<Real>& u0, const int nvar, const int *idcs,
              const Real *Wx, const Real *Wy, const Real *Wz, Real *res) const {
    if constexpr (TRI) {
      TrilinearInterpolator<NG>(u0, nvar, idcs, Wx, Wy, Wz, res);
    } else {
      LagrangeInterpolator<NG>(u0, nvar, idcs, Wx, Wy, Wz, res);
    }
  }

  KOKKOS_INLINE_FUNCTION
  bool operator()(const Real xin[3], const Real uin[3], Real xout[3], Real uout[3],
                  bool Euler) const {
    Real x_mid[3] = {0.0}, u_mid[3] = {0.0};
    for (int i = 0; i < 3; ++i) {
      x_mid[i] = 0.5 * (xin[i] + x_old[i]);   // x_mid = x_old for the Euler step
      u_mid[i] = 0.5 * (uin[i] + u_old[i]);   // u_mid = u_old for the Euler step
    }
    int interp_indcs[4] = {mb, -1, -1, -1};
    SetInterpIndices(x_mid, mb_par, ncell, interp_indcs);
    Real Lx[8] = {0.0}, Ly[8] = {0.0}, Lz[8] = {0.0};
    Real dLx[8] = {0.0}, dLy[8] = {0.0}, dLz[8] = {0.0};
    if constexpr (TRI) {
      CalcTrilinearWghtAndDrv(x_mid, mb_par, ncell,
                              interp_indcs, Lx, Ly, Lz, dLx, dLy, dLz);
    } else {
      CalcInterpWghtAndDrv<NG>(x_mid, mb_par, ncell,
                               interp_indcs, Lx, Ly, Lz, dLx, dLy, dLz);
    }

    // ---- Step 1: update position ----
    // (i) interpolate lapse/shift/3-metric at the midpoint (time-averaged old/new)
    Real alp_old = 0.0, alp_new = 0.0, alp = 0.0;
    Real beta_old[3] = {0.0}, beta_new[3] = {0.0}, beta[3] = {0.0};
    Real g3d_old[6] = {0.0}, g3d_new[6] = {0.0}, g3d[6] = {0.0};
    if (use_z4c) {
      alp_old = Interp(z4c_old, z4c::Z4c::I_Z4C_ALPHA,
                       interp_indcs, Lx, Ly, Lz);
      alp_new = Interp(z4c_new, z4c::Z4c::I_Z4C_ALPHA,
                       interp_indcs, Lx, Ly, Lz);
      for (int i = 0; i < 3; ++i) {
        beta_old[i] = Interp(z4c_old, z4c::Z4c::I_Z4C_BETAX+i,
                             interp_indcs, Lx, Ly, Lz);
        beta_new[i] = Interp(z4c_new, z4c::Z4c::I_Z4C_BETAX+i,
                             interp_indcs, Lx, Ly, Lz);
      }
    } else {
      alp_old = Interp(adm_old, adm::ADM::I_ADM_ALPHA,
                       interp_indcs, Lx, Ly, Lz);
      alp_new = Interp(adm_new, adm::ADM::I_ADM_ALPHA,
                       interp_indcs, Lx, Ly, Lz);
      for (int i = 0; i < 3; ++i) {
        beta_old[i] = Interp(adm_old, adm::ADM::I_ADM_BETAX+i,
                             interp_indcs, Lx, Ly, Lz);
        beta_new[i] = Interp(adm_new, adm::ADM::I_ADM_BETAX+i,
                             interp_indcs, Lx, Ly, Lz);
      }
    }
    alp = 0.5 * (alp_old + alp_new);
    for (int i = 0; i < 3; ++i) { beta[i] = 0.5 * (beta_old[i] + beta_new[i]); }
    for (int i = 0; i < 6; ++i) {
      g3d_old[i] = Interp(adm_old, adm::ADM::I_ADM_GXX+i,
                          interp_indcs, Lx, Ly, Lz);
      g3d_new[i] = Interp(adm_new, adm::ADM::I_ADM_GXX+i,
                          interp_indcs, Lx, Ly, Lz);
      g3d[i] = 0.5 * (g3d_old[i] + g3d_new[i]);
    }
    if (Euler) {
      alp = alp_old;
      for (int i = 0; i < 3; ++i) { beta[i] = beta_old[i]; }
      for (int i = 0; i < 6; ++i) { g3d[i] = g3d_old[i]; }
    }
    // (ii) transport velocity v^i = alp u^i / W - beta^i
    // Reject a non-positive-definite interpolant by Sylvester's criterion on the
    // leading principal minors (see the struct docstring for why det > 0 is not
    // enough). gamma_ij storage is (xx,xy,xz,yy,yz,zz).
    Real det = Primitive::GetDeterminant(g3d);
    Real minor1 = g3d[0];
    Real minor2 = g3d[0]*g3d[3] - g3d[1]*g3d[1];
    if (!(minor1 > 0.0) || !(minor2 > 0.0) || !(det > 0.0)) {return false;}
    Real g3u[6] = {0.0};
    Primitive::InvertMatrix(g3u, g3d, det);
    Real u_mid_u[3] = {0.0};
    Primitive::RaiseForm(u_mid_u, u_mid, g3u);
    // usq >= 0 for a positive-definite gamma; belt-and-braces in case round-off in the
    // minors lets a marginal case through, so W stays real.
    Real usq = Primitive::Contract(u_mid_u, u_mid);
    if (!(usq >= 0.0)) {return false;}
    Real Lorentz = std::sqrt(1.0 + usq);
    Real v[3] = {0.0};
    for (int i = 0; i < 3; ++i) { v[i] = alp * u_mid_u[i] / Lorentz - beta[i]; }
    // (iii) advance position
    for (int i = 0; i < 3; ++i) { xout[i] = x_old[i] + dt * v[i]; }

    // ---- Step 2: update velocity ----
    // (i) interpolate spatial derivatives of lapse, shift, and inverse 3-metric at
    // the midpoint
    Real dalp_old[3] = {0.0}, dalp_new[3] = {0.0}, dalp[3] = {0.0};
    Real dbeta_old[3][3] = {0.0}, dbeta_new[3][3] = {0.0}, dbeta[3][3] = {0.0};
    Real dg3u_old[3][6] = {0.0}, dg3u_new[3][6] = {0.0}, dg3u[3][6] = {0.0};
    if (use_z4c) {
      dalp_old[0] = Interp(z4c_old, z4c::Z4c::I_Z4C_ALPHA,
                           interp_indcs, dLx, Ly, Lz);
      dalp_old[1] = Interp(z4c_old, z4c::Z4c::I_Z4C_ALPHA,
                           interp_indcs, Lx, dLy, Lz);
      dalp_old[2] = Interp(z4c_old, z4c::Z4c::I_Z4C_ALPHA,
                           interp_indcs, Lx, Ly, dLz);
      dalp_new[0] = Interp(z4c_new, z4c::Z4c::I_Z4C_ALPHA,
                           interp_indcs, dLx, Ly, Lz);
      dalp_new[1] = Interp(z4c_new, z4c::Z4c::I_Z4C_ALPHA,
                           interp_indcs, Lx, dLy, Lz);
      dalp_new[2] = Interp(z4c_new, z4c::Z4c::I_Z4C_ALPHA,
                           interp_indcs, Lx, Ly, dLz);
      for (int i = 0; i < 3; ++i) {
        dbeta_old[0][i] = Interp(z4c_old, z4c::Z4c::I_Z4C_BETAX+i,
                                 interp_indcs, dLx, Ly, Lz);
        dbeta_old[1][i] = Interp(z4c_old, z4c::Z4c::I_Z4C_BETAX+i,
                                 interp_indcs, Lx, dLy, Lz);
        dbeta_old[2][i] = Interp(z4c_old, z4c::Z4c::I_Z4C_BETAX+i,
                                 interp_indcs, Lx, Ly, dLz);
        dbeta_new[0][i] = Interp(z4c_new, z4c::Z4c::I_Z4C_BETAX+i,
                                 interp_indcs, dLx, Ly, Lz);
        dbeta_new[1][i] = Interp(z4c_new, z4c::Z4c::I_Z4C_BETAX+i,
                                 interp_indcs, Lx, dLy, Lz);
        dbeta_new[2][i] = Interp(z4c_new, z4c::Z4c::I_Z4C_BETAX+i,
                                 interp_indcs, Lx, Ly, dLz);
      }
    } else {
      dalp_old[0] = Interp(adm_old, adm::ADM::I_ADM_ALPHA,
                           interp_indcs, dLx, Ly, Lz);
      dalp_old[1] = Interp(adm_old, adm::ADM::I_ADM_ALPHA,
                           interp_indcs, Lx, dLy, Lz);
      dalp_old[2] = Interp(adm_old, adm::ADM::I_ADM_ALPHA,
                           interp_indcs, Lx, Ly, dLz);
      dalp_new[0] = Interp(adm_new, adm::ADM::I_ADM_ALPHA,
                           interp_indcs, dLx, Ly, Lz);
      dalp_new[1] = Interp(adm_new, adm::ADM::I_ADM_ALPHA,
                           interp_indcs, Lx, dLy, Lz);
      dalp_new[2] = Interp(adm_new, adm::ADM::I_ADM_ALPHA,
                           interp_indcs, Lx, Ly, dLz);
      for (int i = 0; i < 3; ++i) {
        dbeta_old[0][i] = Interp(adm_old, adm::ADM::I_ADM_BETAX+i,
                                 interp_indcs, dLx, Ly, Lz);
        dbeta_old[1][i] = Interp(adm_old, adm::ADM::I_ADM_BETAX+i,
                                 interp_indcs, Lx, dLy, Lz);
        dbeta_old[2][i] = Interp(adm_old, adm::ADM::I_ADM_BETAX+i,
                                 interp_indcs, Lx, Ly, dLz);
        dbeta_new[0][i] = Interp(adm_new, adm::ADM::I_ADM_BETAX+i,
                                 interp_indcs, dLx, Ly, Lz);
        dbeta_new[1][i] = Interp(adm_new, adm::ADM::I_ADM_BETAX+i,
                                 interp_indcs, Lx, dLy, Lz);
        dbeta_new[2][i] = Interp(adm_new, adm::ADM::I_ADM_BETAX+i,
                                 interp_indcs, Lx, Ly, dLz);
      }
    }
    for (int i = 0; i < 3; ++i) {
      dalp[i] = 0.5 * (dalp_old[i] + dalp_new[i]);
      for (int j = 0; j < 3; ++j) {
        dbeta[i][j] = 0.5 * (dbeta_old[i][j] + dbeta_new[i][j]);
      }
    }
    Interp(adm_old, adm::ADM::I_ADM_GXX,
           interp_indcs, dLx, Ly, Lz, dg3u_old[0]);
    Interp(adm_old, adm::ADM::I_ADM_GXX,
           interp_indcs, Lx, dLy, Lz, dg3u_old[1]);
    Interp(adm_old, adm::ADM::I_ADM_GXX,
           interp_indcs, Lx, Ly, dLz, dg3u_old[2]);
    Interp(adm_new, adm::ADM::I_ADM_GXX,
           interp_indcs, dLx, Ly, Lz, dg3u_new[0]);
    Interp(adm_new, adm::ADM::I_ADM_GXX,
           interp_indcs, Lx, dLy, Lz, dg3u_new[1]);
    Interp(adm_new, adm::ADM::I_ADM_GXX,
           interp_indcs, Lx, Ly, dLz, dg3u_new[2]);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 6; ++j) {
        dg3u[i][j] = 0.5 * (dg3u_old[i][j] + dg3u_new[i][j]);
      }
    }
    if (Euler) {
      for (int i = 0; i < 3; ++i) {
        dalp[i] = dalp_old[i];
        for (int j = 0; j < 3; ++j) { dbeta[i][j] = dbeta_old[i][j]; }
        for (int j = 0; j < 6; ++j) { dg3u[i][j] = dg3u_old[i][j]; }
      }
    }
    // (ii) geodesic force on the covariant velocity:
    //   du_i/dt = -W d_i alp + u_j d_i beta^j - alp/(2W) u_j u_k d_i gamma^{jk}
    Real g[3] = {0.0};
    for (int i = 0; i < 3; ++i) {
      g[i] = -1. * Lorentz * dalp[i] + u_mid[0] * dbeta[i][0] + u_mid[1] * dbeta[i][1] +
             u_mid[2] * dbeta[i][2] -
             0.5 * alp / Lorentz * (u_mid[0] * u_mid[0] * dg3u[i][0] +
                                    u_mid[1] * u_mid[1] * dg3u[i][3] +
                                    u_mid[2] * u_mid[2] * dg3u[i][5] +
                                    2. * u_mid[0] * u_mid[1] * dg3u[i][1] +
                                    2. * u_mid[0] * u_mid[2] * dg3u[i][2] +
                                    2. * u_mid[1] * u_mid[2] * dg3u[i][4]);
    }
    // (iii) advance velocity
    for (int i = 0; i < 3; ++i) { uout[i] = u_old[i] + dt * g[i]; }
    return true;
  }
};

//----------------------------------------------------------------------------------------
//! \enum GeodesicStatus
//! \brief outcome of one implicit geodesic substep.
//!   kConverged : the fixed point converged; (x,u) are the step-n+1 state
//!   kEuler     : the fixed point did not converge, but the forward-Euler fallback
//!                produced a VALID finite state -- a documented first-order step
//!   kRejected  : neither produced a usable state (invalid interpolated geometry or a
//!                non-finite result). (x,u) are meaningless and MUST NOT be written back.

enum GeodesicStatus {kConverged = 0, kEuler = 1, kRejected = 2};

//----------------------------------------------------------------------------------------
//! \fn int FixedPointIteration
//! \brief solve the implicit geodesic substep x=f(x) by fixed-point iteration. On
//! non-finite iterates or non-convergence it falls back to a forward-Euler step
//! f(x0,u0,...,Euler=true). The Euler result is validated too: an unchecked fallback is
//! how a NaN reaches the particle array.

template<class F>
KOKKOS_INLINE_FUNCTION
int FixedPointIteration(const F& f, const Real x0[3], const Real u0[3],
                        Real x[3], Real u[3], Real tol=1e-7, int maxIter=50) {
  Real x_new[3], u_new[3];
  for (int i = 0; i < 3; ++i) { x_new[i] = x0[i]; u_new[i] = u0[i]; }
  Real x_next[3], u_next[3];
  for (int iter = 0; iter < maxIter; ++iter) {
    if (!f(x_new, u_new, x_next, u_next, false)) { break; }
    bool to_break = false;
    for (int i = 0; i < 3; ++i) {
      if (!std::isfinite(x_next[i]) || !std::isfinite(u_next[i])) { to_break = true; }
    }
    if (to_break) { break; }
    Real err = 0.0;
    for (int i = 0; i < 3; ++i) {
      err = fmax(err, fabs(x_next[i] - x_new[i]));
      err = fmax(err, fabs(u_next[i] - u_new[i]));
    }
    if (err < tol) {
      for (int i = 0; i < 3; ++i) { x[i] = x_next[i]; u[i] = u_next[i]; }
      return kConverged;
    }
    for (int i = 0; i < 3; ++i) { x_new[i] = x_next[i]; u_new[i] = u_next[i]; }
  }
  if (!f(x0, u0, x, u, true)) { return kRejected; }   // forward-Euler fallback
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(x[i]) || !std::isfinite(u[i])) { return kRejected; }
  }
  return kEuler;
}

//----------------------------------------------------------------------------------------
//! \fn bool GridMetricValid
//! \brief Sylvester-test the eight stored gamma_ij the trilinear stencil reads at x, on
//! the step-n metric. Returns false if any of them is not positive definite or not
//! finite.
//!
//! This separates the two causes of a rejected substep: a high-order interpolant
//! overshooting between good grid values, which the fallback legitimately repairs, from
//! grid values that are already not a Riemannian 3-metric, where a "repair" would be
//! manufacturing a plausible number out of an invalid solution.
//!
//! It is a DIAGNOSTIC and deliberately does not gate the write-back. The gate is still
//! the Sylvester test inside GeodesicPush, applied to the interpolated metric the retry
//! actually uses; what this adds is the knowledge of whether the convexity argument
//! applied at all. One bad corner carrying a small weight can still leave the combination
//! positive definite, so rejecting on it would throw away sound repairs -- but a run in
//! which this count is not zero is a run whose stored solution needs looking at, not one
//! whose particle pusher needs looking at. Reachable because
//! gamma_ij = chi^(4/chi_psi_power) * gammat_ij (z4c_adm.cpp) with the default
//! chi_psi_power = -4, so one grid point with chi <= 0 flips the sign of the whole
//! matrix, and <z4c> floor_chi defaults to false.
//!
//! Scope: (adm_n, x_n) is the stencil the forward-Euler evaluation reads, and it is the
//! Euler evaluation failing that makes a substep kRejected, so this is the decisive
//! stencil. It is not the stencil of the fixed-point iterates, which sit at moving
//! midpoints. "One bad corner" also only voids the convexity guarantee; a bad corner
//! carrying a small weight can still leave the combination positive definite.

template <int NG>
KOKKOS_INLINE_FUNCTION
bool GridMetricValid(const DvceArray5D<Real>& adm, const Real x[3],
                     const Real *mb_par, const int *ncell, const int mb) {
  int idcs[4] = {mb, -1, -1, -1};
  SetInterpIndices(x, mb_par, ncell, idcs);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        Real g[6] = {0.0};
        for (int m = 0; m < 6; ++m) {
          g[m] = adm(idcs[0], adm::ADM::I_ADM_GXX+m,
                     idcs[3] + NG + k, idcs[2] + NG + j, idcs[1] + NG + i);
          if (!Kokkos::isfinite(g[m])) {return false;}
        }
        Real minor2 = g[0]*g[3] - g[1]*g[1];
        if (!(g[0] > 0.0) || !(minor2 > 0.0) ||
            !(Primitive::GetDeterminant(g) > 0.0)) {return false;}
      }
    }
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn void Particles::GR_BorisPush

void Particles::GR_BorisPush() {
  // GR Boris requires the ADM metric (the constructor enforces this; guard again so
  // misuse fails safe rather than dereferencing a null pointer).
  if (pmy_pack->padm == nullptr) {return;}

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ng = indcs.ng;
  auto &size = pmy_pack->pmb->mb_size;
  int gids = pmy_pack->gids;
  auto dt_ = pmy_pack->pmesh->dt;
  auto qom = q_over_m;

  auto nfail_ = boris_nfail;          // bounded non-convergence diagnostic counters
  const int ndetail_ = kBorisDetail;
  // The low-order retry is geodesic-only: it runs after the push, in its own kernel, and
  // cannot see the u^+ the first EM half-kick left in a register. With MHD that half-kick
  // has also already consumed its own interpolated gamma_ij through CalcTetrad, so the
  // state it would re-solve from is the poisoned one. An MHD run therefore keeps the
  // previous behaviour exactly: a rejected substep is rejected.
  const bool retry_on_ = (pmy_pack->pmhd == nullptr);
  if (retry_on_ && nprtcl_thispack > static_cast<int>(boris_retry.extent(0))) {
    Kokkos::realloc(boris_retry, nprtcl_thispack);
  }
  auto retry_ = boris_retry;

  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;

  // step-n snapshots (the *_last members) and step-(n+1) live arrays
  auto &w0_n = w0_last;
  auto &bcc0_n = bcc0_last;
  auto &adm_n = adm_last;
  auto &adm_np1 = pmy_pack->padm->u_adm;

  DvceArray5D<Real> w0_np1, bcc0_np1;
  bool use_mhd = false;
  if (pmy_pack->pmhd != nullptr) {
    use_mhd = true;
    w0_np1 = pmy_pack->pmhd->w0;
    bcc0_np1 = pmy_pack->pmhd->bcc0;
  }

  DvceArray5D<Real> z4c_n, z4c_np1;
  bool use_z4c = false;
  if (pmy_pack->pz4c != nullptr) {
    use_z4c = true;
    z4c_n = z4c_last;
    z4c_np1 = pmy_pack->pz4c->u0;
  }

  par_for("gr_boris_push", DevExeSpace(), 0, nprtcl_thispack - 1,
  KOKKOS_LAMBDA(const int p) {
    int mb = pi(PGID, p) - gids;
    const Real mb_par[9] = {size.d_view(mb).x1min, size.d_view(mb).x1max,
                            size.d_view(mb).dx1,
                            size.d_view(mb).x2min, size.d_view(mb).x2max,
                            size.d_view(mb).dx2,
                            size.d_view(mb).x3min, size.d_view(mb).x3max,
                            size.d_view(mb).dx3};
    int ncell[3] = {indcs.nx1, indcs.nx2, indcs.nx3};

    if (retry_on_) { retry_(p) = 0; }

    Real x_n[3] = {pr(IPX, p), pr(IPY, p), pr(IPZ, p)};
    Real u_n[3] = {pr(IPVX, p), pr(IPVY, p), pr(IPVZ, p)};

    // ---- Step 1+2: first half EM kick at x^n (skipped when no MHD) ----
    Real u_p[3] = {0.0};
    if (use_mhd) {
      Real B_interp_n[3] = {0.0}, v_interp_n[3] = {0.0};
      Real alp_n = 0.0, beta_n[3] = {0.0}, g3d_n[6] = {0.0};
      int interp_indcs[4] = {mb, -1, -1, -1};
      SetInterpIndices(x_n, mb_par, ncell, interp_indcs);
      Real Lx[8] = {0.0}, Ly[8] = {0.0}, Lz[8] = {0.0};
      switch (ng) {
      case 2: {
        CalcInterpWght<2>(x_n, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
        for (int idx = 0; idx < 3; ++idx) {
          B_interp_n[idx] = LagrangeInterpolator<2>(bcc0_n, idx,
                                                    interp_indcs, Lx, Ly, Lz);
          v_interp_n[idx] = LagrangeInterpolator<2>(w0_n, idx+IVX,
                                                    interp_indcs, Lx, Ly, Lz);
        }
        if (use_z4c) {
          alp_n = LagrangeInterpolator<2>(z4c_n, z4c::Z4c::I_Z4C_ALPHA,
                                          interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_n[idx] = LagrangeInterpolator<2>(z4c_n, idx+z4c::Z4c::I_Z4C_BETAX,
                                                  interp_indcs, Lx, Ly, Lz);
          }
        } else {
          alp_n = LagrangeInterpolator<2>(adm_n, adm::ADM::I_ADM_ALPHA,
                                          interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_n[idx] = LagrangeInterpolator<2>(adm_n, idx+adm::ADM::I_ADM_BETAX,
                                                  interp_indcs, Lx, Ly, Lz);
          }
        }
        for (int idx = 0; idx < 6; ++idx) {
          g3d_n[idx] = LagrangeInterpolator<2>(adm_n, idx+adm::ADM::I_ADM_GXX,
                                               interp_indcs, Lx, Ly, Lz);
        }
        break;
      }
      case 3: {
        CalcInterpWght<3>(x_n, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
        for (int idx = 0; idx < 3; ++idx) {
          B_interp_n[idx] = LagrangeInterpolator<3>(bcc0_n, idx,
                                                    interp_indcs, Lx, Ly, Lz);
          v_interp_n[idx] = LagrangeInterpolator<3>(w0_n, idx+IVX,
                                                    interp_indcs, Lx, Ly, Lz);
        }
        if (use_z4c) {
          alp_n = LagrangeInterpolator<3>(z4c_n, z4c::Z4c::I_Z4C_ALPHA,
                                          interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_n[idx] = LagrangeInterpolator<3>(z4c_n, idx+z4c::Z4c::I_Z4C_BETAX,
                                                  interp_indcs, Lx, Ly, Lz);
          }
        } else {
          alp_n = LagrangeInterpolator<3>(adm_n, adm::ADM::I_ADM_ALPHA,
                                          interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_n[idx] = LagrangeInterpolator<3>(adm_n, idx+adm::ADM::I_ADM_BETAX,
                                                  interp_indcs, Lx, Ly, Lz);
          }
        }
        for (int idx = 0; idx < 6; ++idx) {
          g3d_n[idx] = LagrangeInterpolator<3>(adm_n, idx+adm::ADM::I_ADM_GXX,
                                               interp_indcs, Lx, Ly, Lz);
        }
        break;
      }
      case 4: {
        CalcInterpWght<4>(x_n, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
        for (int idx = 0; idx < 3; ++idx) {
          B_interp_n[idx] = LagrangeInterpolator<4>(bcc0_n, idx,
                                                    interp_indcs, Lx, Ly, Lz);
          v_interp_n[idx] = LagrangeInterpolator<4>(w0_n, idx+IVX,
                                                    interp_indcs, Lx, Ly, Lz);
        }
        if (use_z4c) {
          alp_n = LagrangeInterpolator<4>(z4c_n, z4c::Z4c::I_Z4C_ALPHA,
                                          interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_n[idx] = LagrangeInterpolator<4>(z4c_n, idx+z4c::Z4c::I_Z4C_BETAX,
                                                  interp_indcs, Lx, Ly, Lz);
          }
        } else {
          alp_n = LagrangeInterpolator<4>(adm_n, adm::ADM::I_ADM_ALPHA,
                                          interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_n[idx] = LagrangeInterpolator<4>(adm_n, idx+adm::ADM::I_ADM_BETAX,
                                                  interp_indcs, Lx, Ly, Lz);
          }
        }
        for (int idx = 0; idx < 6; ++idx) {
          g3d_n[idx] = LagrangeInterpolator<4>(adm_n, idx+adm::ADM::I_ADM_GXX,
                                               interp_indcs, Lx, Ly, Lz);
        }
        break;
      }
      }
      // bcc0 stores the densitized B~^i = sqrt(gamma) B^i and w0 the projected 4-velocity
      // utilde^i = W v^i (dyn_grmhd conventions): recover physical B and Valencia v first
      Real sqrtdet = std::sqrt(Primitive::GetDeterminant(g3d_n));
      for (int idx = 0; idx < 3; ++idx) { B_interp_n[idx] /= sqrtdet; }
      Real ut_d_n[3] = {0.0};
      Primitive::LowerVector(ut_d_n, v_interp_n, g3d_n);
      Real Wf_n = std::sqrt(1.0 + Primitive::Contract(v_interp_n, ut_d_n));
      for (int idx = 0; idx < 3; ++idx) { v_interp_n[idx] /= Wf_n; }

      // normal-frame ideal-MHD E field: E_i = -sqrt(gamma) eps_ijk v^j B^k
      Real E_interp_n[3] = {0.0};
      E_interp_n[0] = sqrtdet * (B_interp_n[1]*v_interp_n[2] -
                                 B_interp_n[2]*v_interp_n[1]);
      E_interp_n[1] = sqrtdet * (B_interp_n[2]*v_interp_n[0] -
                                 B_interp_n[0]*v_interp_n[2]);
      E_interp_n[2] = sqrtdet * (B_interp_n[0]*v_interp_n[1] -
                                 B_interp_n[1]*v_interp_n[0]);

      // half EM kick in the local orthonormal tetrad
      Real tetrad[4][4] = {0.0}, inv_tetrad[4][4] = {0.0};
      CalcTetrad(alp_n, beta_n, g3d_n, tetrad, inv_tetrad);
      Real uhat_n[3] = {0.0}, Ehat_n[3] = {0.0}, Bhat_n[3] = {0.0};
      TetradCvrtL(uhat_n, u_n, inv_tetrad);
      TetradCvrtL(Ehat_n, E_interp_n, inv_tetrad);
      TetradCvrtU(Bhat_n, B_interp_n, tetrad);
      Real uhat_p[3] = {0.0};
      FlatBorisPush(uhat_p, uhat_n, Ehat_n, Bhat_n, qom, 0.5 * dt_);
      TetradCvrtL(u_p, uhat_p, tetrad);
    } else {
      for (int i = 0; i < 3; ++i) { u_p[i] = u_n[i]; }
    }

    // ---- Step 3: implicit geodesic substep x^n -> x^{n+1}, u_p -> u_pp ----
    Real x_np1[3] = {0.0}, u_pp[3] = {0.0};
    int gstat = kRejected;
    switch (ng) {
    case 2: {
      GeodesicPush<2, false> gp(x_n, u_p, mb, mb_par, ncell, dt_, adm_n, adm_np1,
                                use_z4c, z4c_n, z4c_np1);
      gstat = FixedPointIteration(gp, x_n, u_p, x_np1, u_pp);
      break;
    }
    case 3: {
      GeodesicPush<3, false> gp(x_n, u_p, mb, mb_par, ncell, dt_, adm_n, adm_np1,
                                use_z4c, z4c_n, z4c_np1);
      gstat = FixedPointIteration(gp, x_n, u_p, x_np1, u_pp);
      break;
    }
    case 4: {
      GeodesicPush<4, false> gp(x_n, u_p, mb, mb_par, ncell, dt_, adm_n, adm_np1,
                                use_z4c, z4c_n, z4c_np1);
      gstat = FixedPointIteration(gp, x_n, u_p, x_np1, u_pp);
      break;
    }
    }
    if (gstat == kRejected && retry_on_) {
      // Hand this particle to the retry kernel rather than rejecting it here. The
      // step-n state is left in place exactly as a rejection would leave it, so if the
      // retry also fails nothing has changed.
      retry_(p) = 1;
      Kokkos::atomic_fetch_add(&nfail_(4), 1);
      return;
    }
    if (gstat == kEuler) {
      // Count EVERY failure, but print at most ndetail_ detailed lines per cycle: the
      // host emits one summary line afterwards, so the log stays O(1) per rank per cycle
      // instead of O(N_particle). The atomic returns this failure's index, which both
      // accumulates the count and claims a detail slot.
      int islot = Kokkos::atomic_fetch_add(&nfail_(0), 1);
      if (islot < ndetail_) {
        Kokkos::printf("### WARNING gr_boris: fixed-point did not converge, "
                       "forward-Euler fallback used | tag=%d gid=%d "
                       "x=(% .6e,% .6e,% .6e) u_i=(% .6e,% .6e,% .6e) "
                       "dt=%.6e\n",
                       pi(PTAG, p), pi(PGID, p),
                       x_n[0], x_n[1], x_n[2], u_p[0], u_p[1], u_p[2], dt_);
      }
    }
    if (gstat == kRejected) {
      // Neither the fixed point nor the Euler fallback produced a usable state. The step
      // is NOT TAKEN: the step-n position and velocity are finite and still inside this
      // MeshBlock, so every downstream invariant holds and the ordinary excision
      // criterion can classify the particle on a later cycle. Writing the invalid result
      // instead lets a NaN into the particle array, where it is invisible to every
      // comparison-based predicate (migration, mesh-exit and excision all silently
      // decline to act on it).
      int islot = Kokkos::atomic_fetch_add(&nfail_(2), 1);
      if (islot < ndetail_) {
        Kokkos::printf("### WARNING gr_boris: geodesic substep REJECTED (interpolated "
                       "3-metric not positive definite or non-finite result); step not "
                       "taken | tag=%d gid=%d x=(% .6e,% .6e,% .6e) "
                       "u_i=(% .6e,% .6e,% .6e) dt=%.6e\n",
                       pi(PTAG, p), pi(PGID, p),
                       x_n[0], x_n[1], x_n[2], u_p[0], u_p[1], u_p[2], dt_);
      }
      return;   // leave pr(IPX..)/pr(IPVX..) at their step-n values
    }

    // ---- Step 4+5: second half EM kick at x^{n+1} (skipped when no MHD) ----
    Real u_np1[3] = {0.0};
    if (use_mhd) {
      Real B_interp_np1[3] = {0.0}, v_interp_np1[3] = {0.0};
      Real alp_np1 = 0.0, beta_np1[3] = {0.0}, g3d_np1[6] = {0.0};
      int interp_indcs[4] = {mb, -1, -1, -1};
      SetInterpIndices(x_np1, mb_par, ncell, interp_indcs);
      Real Lx[8] = {0.0}, Ly[8] = {0.0}, Lz[8] = {0.0};
      switch (ng) {
      case 2: {
        CalcInterpWght<2>(x_np1, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
        for (int idx = 0; idx < 3; ++idx) {
          B_interp_np1[idx] = LagrangeInterpolator<2>(bcc0_np1, idx,
                                                      interp_indcs, Lx, Ly, Lz);
          v_interp_np1[idx] = LagrangeInterpolator<2>(w0_np1, idx+IVX,
                                                      interp_indcs, Lx, Ly, Lz);
        }
        if (use_z4c) {
          alp_np1 = LagrangeInterpolator<2>(z4c_np1, z4c::Z4c::I_Z4C_ALPHA,
                                            interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_np1[idx] = LagrangeInterpolator<2>(z4c_np1, idx+z4c::Z4c::I_Z4C_BETAX,
                                                    interp_indcs, Lx, Ly, Lz);
          }
        } else {
          alp_np1 = LagrangeInterpolator<2>(adm_np1, adm::ADM::I_ADM_ALPHA,
                                            interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_np1[idx] = LagrangeInterpolator<2>(adm_np1, idx+adm::ADM::I_ADM_BETAX,
                                                    interp_indcs, Lx, Ly, Lz);
          }
        }
        for (int idx = 0; idx < 6; ++idx) {
          g3d_np1[idx] = LagrangeInterpolator<2>(adm_np1, idx+adm::ADM::I_ADM_GXX,
                                                 interp_indcs, Lx, Ly, Lz);
        }
        break;
      }
      case 3: {
        CalcInterpWght<3>(x_np1, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
        for (int idx = 0; idx < 3; ++idx) {
          B_interp_np1[idx] = LagrangeInterpolator<3>(bcc0_np1, idx,
                                                      interp_indcs, Lx, Ly, Lz);
          v_interp_np1[idx] = LagrangeInterpolator<3>(w0_np1, idx+IVX,
                                                      interp_indcs, Lx, Ly, Lz);
        }
        if (use_z4c) {
          alp_np1 = LagrangeInterpolator<3>(z4c_np1, z4c::Z4c::I_Z4C_ALPHA,
                                            interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_np1[idx] = LagrangeInterpolator<3>(z4c_np1, idx+z4c::Z4c::I_Z4C_BETAX,
                                                    interp_indcs, Lx, Ly, Lz);
          }
        } else {
          alp_np1 = LagrangeInterpolator<3>(adm_np1, adm::ADM::I_ADM_ALPHA,
                                            interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_np1[idx] = LagrangeInterpolator<3>(adm_np1, idx+adm::ADM::I_ADM_BETAX,
                                                    interp_indcs, Lx, Ly, Lz);
          }
        }
        for (int idx = 0; idx < 6; ++idx) {
          g3d_np1[idx] = LagrangeInterpolator<3>(adm_np1, idx+adm::ADM::I_ADM_GXX,
                                                 interp_indcs, Lx, Ly, Lz);
        }
        break;
      }
      case 4: {
        CalcInterpWght<4>(x_np1, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
        for (int idx = 0; idx < 3; ++idx) {
          B_interp_np1[idx] = LagrangeInterpolator<4>(bcc0_np1, idx,
                                                      interp_indcs, Lx, Ly, Lz);
          v_interp_np1[idx] = LagrangeInterpolator<4>(w0_np1, idx+IVX,
                                                      interp_indcs, Lx, Ly, Lz);
        }
        if (use_z4c) {
          alp_np1 = LagrangeInterpolator<4>(z4c_np1, z4c::Z4c::I_Z4C_ALPHA,
                                            interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_np1[idx] = LagrangeInterpolator<4>(z4c_np1, idx+z4c::Z4c::I_Z4C_BETAX,
                                                    interp_indcs, Lx, Ly, Lz);
          }
        } else {
          alp_np1 = LagrangeInterpolator<4>(adm_np1, adm::ADM::I_ADM_ALPHA,
                                            interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_np1[idx] = LagrangeInterpolator<4>(adm_np1, idx+adm::ADM::I_ADM_BETAX,
                                                    interp_indcs, Lx, Ly, Lz);
          }
        }
        for (int idx = 0; idx < 6; ++idx) {
          g3d_np1[idx] = LagrangeInterpolator<4>(adm_np1, idx+adm::ADM::I_ADM_GXX,
                                                 interp_indcs, Lx, Ly, Lz);
        }
        break;
      }
      }
      // undensitize B and recover Valencia v (same conventions as the step-n kick)
      Real sqrtdet = std::sqrt(Primitive::GetDeterminant(g3d_np1));
      for (int idx = 0; idx < 3; ++idx) { B_interp_np1[idx] /= sqrtdet; }
      Real ut_d_np1[3] = {0.0};
      Primitive::LowerVector(ut_d_np1, v_interp_np1, g3d_np1);
      Real Wf_np1 = std::sqrt(1.0 + Primitive::Contract(v_interp_np1, ut_d_np1));
      for (int idx = 0; idx < 3; ++idx) { v_interp_np1[idx] /= Wf_np1; }

      // normal-frame ideal-MHD E field: E_i = -sqrt(gamma) eps_ijk v^j B^k
      Real E_interp_np1[3] = {0.0};
      E_interp_np1[0] = sqrtdet * (B_interp_np1[1]*v_interp_np1[2] -
                                   B_interp_np1[2]*v_interp_np1[1]);
      E_interp_np1[1] = sqrtdet * (B_interp_np1[2]*v_interp_np1[0] -
                                   B_interp_np1[0]*v_interp_np1[2]);
      E_interp_np1[2] = sqrtdet * (B_interp_np1[0]*v_interp_np1[1] -
                                   B_interp_np1[1]*v_interp_np1[0]);

      Real tetrad[4][4] = {0.0}, inv_tetrad[4][4] = {0.0};
      CalcTetrad(alp_np1, beta_np1, g3d_np1, tetrad, inv_tetrad);
      Real uhat_pp[3] = {0.0}, Ehat_np1[3] = {0.0}, Bhat_np1[3] = {0.0};
      TetradCvrtL(uhat_pp, u_pp, inv_tetrad);
      TetradCvrtL(Ehat_np1, E_interp_np1, inv_tetrad);
      TetradCvrtU(Bhat_np1, B_interp_np1, tetrad);
      Real uhat_np1[3] = {0.0};
      FlatBorisPush(uhat_np1, uhat_pp, Ehat_np1, Bhat_np1, qom, 0.5 * dt_);
      TetradCvrtL(u_np1, uhat_np1, tetrad);
    } else {
      for (int i = 0; i < 3; ++i) { u_np1[i] = u_pp[i]; }
    }

    // ---- Step 6: write back -------------------------------------------------------
    // Backstop making "a non-finite state is never written" hold whatever produced it.
    // The geodesic substep is guarded at its source, but the two EM half-kicks are not:
    // they interpolate their own gamma_ij and feed it to CalcTetrad, whose
    // sqrt(gxx*gyy - gxy^2) and 1/sqrt(g3u[5]) are an unguarded Cholesky. Guarding here
    // rather than duplicating the Sylvester test into those branches keeps the invariant
    // total, including for sub-steps added later. Unreachable unless the old code would
    // have written a NaN, so it cannot change a finite result.
    bool finite_out = Kokkos::isfinite(x_np1[0]) && Kokkos::isfinite(x_np1[1])
                   && Kokkos::isfinite(x_np1[2]) && Kokkos::isfinite(u_np1[0])
                   && Kokkos::isfinite(u_np1[1]) && Kokkos::isfinite(u_np1[2]);
    if (!finite_out) {
      int islot = Kokkos::atomic_fetch_add(&nfail_(3), 1);
      if (islot < ndetail_) {
        Kokkos::printf("### WARNING gr_boris: non-finite state after the EM half-kicks; "
                       "step not taken | tag=%d gid=%d x=(% .6e,% .6e,% .6e) "
                       "u_i=(% .6e,% .6e,% .6e) dt=%.6e\n",
                       pi(PTAG, p), pi(PGID, p),
                       x_n[0], x_n[1], x_n[2], u_p[0], u_p[1], u_p[2], dt_);
      }
      return;   // leave pr(IPX..)/pr(IPVX..) at their step-n values
    }
    pr(IPX, p) = x_np1[0];
    pr(IPY, p) = x_np1[1];
    pr(IPZ, p) = x_np1[2];
    pr(IPVX, p) = u_np1[0];
    pr(IPVY, p) = u_np1[1];
    pr(IPVZ, p) = u_np1[2];
  });

  // ---- retry the rejected substeps, in a SEPARATE kernel ---------------------------
  // Separate because GeodesicPush is inlined into its caller: putting both interpolation
  // operators in one kernel overflows the register file on gfx90a and spills to scratch,
  // which costs orders of magnitude on every push whether or not the fallback fires.
  // Launched only when the push flagged something.
  // ---- bounded non-convergence summary: ONE line per rank per cycle ----------------
  // Read the device counter, report it, and reset for the next cycle. Failures are never
  // hidden: the count is exact and the per-rank running total is included.
  {
    auto hfail = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boris_nfail);
    // The retry needs the same counter, so it is launched from inside this block and the
    // mirror is re-read only when it actually ran: a cycle with nothing to retry costs
    // exactly the one device-to-host copy it cost before.
    if (retry_on_ && hfail(4) > 0) {
      GRBorisRetry();
      hfail = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boris_nfail);
    }
    int nfail = hfail(0);
    int nrej_geo = hfail(2);
    int nrej_out = hfail(3);
    int nrej = nrej_geo + nrej_out;
    int nretry = hfail(4);
    int nrescued = hfail(5);
    int nbadgrid = hfail(6);
    if (nretry > 0) {
      boris_nretry_cum += static_cast<std::int64_t>(nretry);
      boris_nrescued_cum += static_cast<std::int64_t>(nrescued);
      if (!boris_first_retry_seen) {
        boris_first_retry_seen = true;
        std::cout << "### WARNING in " << __FILE__ << ": a geodesic substep whose "
                  << "high-order geometry was invalid was re-solved with a TRILINEAR "
                  << "interpolant for the first time this run (rank "
                  << global_variable::my_rank << ", cycle " << pmy_pack->pmesh->ncycle
                  << ")." << std::endl
                  << "    That push is first order accurate in space, not " << 2*ng
                  << "th: it is a repair, not an improvement, and it means the metric is "
                  << "under-resolved where that particle sits -- refine, or excise "
                  << "earlier." << std::endl
                  << "    'invalid grid' below counts the retries whose eight "
                  << "surrounding STORED gamma_ij were themselves not a 3-metric. Those "
                  << "are not an interpolation problem and are not repaired."
                  << std::endl;
      }
      std::cout << "### trilinear retry: rank " << global_variable::my_rank
                << " cycle " << pmy_pack->pmesh->ncycle << ": " << nretry
                << " of " << nprtcl_thispack << " re-solved, " << nrescued
                << " rescued, " << nbadgrid << " with an invalid grid metric"
                << " (cumulative " << boris_nrescued_cum << "/" << boris_nretry_cum
                << ")" << std::endl;
    }
    if (nrej > 0) {
      boris_nreject_cum += static_cast<std::int64_t>(nrej);
      if (!boris_first_reject_seen) {
        boris_first_reject_seen = true;
        std::cout << "### WARNING in " << __FILE__ << ": gr_boris REJECTED a particle "
                  << "update for the first time this run (rank "
                  << global_variable::my_rank << ", cycle " << pmy_pack->pmesh->ncycle
                  << ", dt = " << pmy_pack->pmesh->dt << ")." << std::endl
                  << "    The interpolated 3-metric at the particle was not positive "
                  << "definite, or the resulting state was non-finite, so the step was "
                  << "NOT TAKEN and the particle kept its step-n position and momentum."
                  << std::endl
                  << "    This is a real loss of accuracy for that particle, not a "
                  << "crash: it happens where a high-order interpolation of gamma_ij "
                  << "overshoots, i.e. inside a moving-puncture trumpet." << std::endl
                  << "    In a geodesic configuration the trilinear retry has already "
                  << "been tried and also failed, so the eight surrounding grid values "
                  << "are themselves suspect -- see the 'invalid grid metric' count."
                  << std::endl
                  << "    If it affects more than a negligible fraction of the "
                  << "population, the matter there is under-resolved -- refine, or "
                  << "excise earlier (raise <particles> excise_lapse)." << std::endl;
      }
      std::cout << "### gr_boris update rejected: rank " << global_variable::my_rank
                << " cycle " << pmy_pack->pmesh->ncycle << ": " << nrej << " of "
                << nprtcl_thispack << " particles kept their step-n state"
                << " (geodesic " << nrej_geo << ", write-back " << nrej_out
                << "; rank total " << boris_nreject_cum << ")" << std::endl;
      if (fatal_boris_reject) {
        std::cout << "### FATAL ERROR: <particles> fatal_boris_reject=true and rank "
                  << global_variable::my_rank << " rejected " << nrej
                  << " GR-Boris update(s) at cycle " << pmy_pack->pmesh->ncycle
                  << ". The run cannot advance a trustworthy particle solution."
                  << std::endl;
#if MPI_PARALLEL_ENABLED
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        std::exit(EXIT_FAILURE);
#endif
      }
    }
    if (nfail > 0) {
      boris_nfail_cum += static_cast<std::int64_t>(nfail);
      if (!boris_first_fail_seen) {
        boris_first_fail_seen = true;
        std::cout << "### WARNING in " << __FILE__ << ": gr_boris implicit geodesic "
                  << "substep did not converge for the first time this run (rank "
                  << global_variable::my_rank << ", cycle " << pmy_pack->pmesh->ncycle
                  << ", dt = " << pmy_pack->pmesh->dt << ")." << std::endl
                  << "    The step falls back to forward Euler, which is a documented "
                  << "first-order fallback, not a crash." << std::endl
                  << "    It is expected at large CFL and disappears as dt is reduced; "
                  << "if it persists, reduce <time> cfl_number." << std::endl
                  << "    Per-particle detail is printed for at most "
                  << kBorisDetail << " particles per cycle; every failure is counted "
                  << "in the per-cycle summary below." << std::endl;
      }
      std::cout << "### gr_boris non-convergence: rank " << global_variable::my_rank
                << " cycle " << pmy_pack->pmesh->ncycle << ": " << nfail << " of "
                << nprtcl_thispack << " particles fell back to forward Euler"
                << " (rank total " << boris_nfail_cum << ")" << std::endl;
    }
    // slots 4-6 can be non-zero on a cycle where nfail and nrej are both zero (every
    // retry succeeded), and a counter that survives is double counted next cycle
    if (nfail > 0 || nrej > 0 || nretry > 0) {Kokkos::deep_copy(boris_nfail, 0);}
  }

  // snapshot the current fields/metric as step n for the next push (for a static
  // background these copies are a no-op; they carry the time level once the metric
  // evolves, Stage 4)
  if (use_mhd) {
    Kokkos::deep_copy(DevExeSpace(), w0_last, pmy_pack->pmhd->w0);
    Kokkos::deep_copy(DevExeSpace(), bcc0_last, pmy_pack->pmhd->bcc0);
  }
  Kokkos::deep_copy(DevExeSpace(), adm_last, pmy_pack->padm->u_adm);
  if (use_z4c) {
    Kokkos::deep_copy(DevExeSpace(), z4c_last, pmy_pack->pz4c->u0);
  }
  Kokkos::fence();
}


//----------------------------------------------------------------------------------------
//! \fn void Particles::GRBorisRetry
//! \brief re-solve, with a trilinear interpolant, the geodesic substeps GR_BorisPush
//! rejected.
//!
//! Runs only over the particles the push flagged, and only when it flagged at least one.
//! A particle reaches this kernel only if BOTH the high-order fixed point and the
//! high-order forward-Euler evaluation failed to produce a usable state, so an accepted
//! push is bit-for-bit unaffected by the existence of this pass.
//!
//! Geodesic-only, and the caller enforces that by not running it under MHD (see
//! GR_BorisPush). Every guard the push applies applies here too: the Sylvester and
//! usq >= 0 tests inside GeodesicPush, a finiteness check before write-back, and a
//! rejected retry that leaves the step-n state untouched.

void Particles::GRBorisRetry() {
  // correctness of this pass depends on u^+ == u^n, i.e. on there being no EM half-kick;
  // GR_BorisPush already only calls it then, but the entry point is public so guard it
  if (pmy_pack->pmhd != nullptr) {return;}
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ng = indcs.ng;
  auto &size = pmy_pack->pmb->mb_size;
  int gids = pmy_pack->gids;
  auto dt_ = pmy_pack->pmesh->dt;

  auto nfail_ = boris_nfail;
  auto retry_ = boris_retry;
  const int ndetail_ = kBorisDetail;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto &adm_n = adm_last;
  auto &adm_np1 = pmy_pack->padm->u_adm;

  DvceArray5D<Real> z4c_n, z4c_np1;
  bool use_z4c = false;
  if (pmy_pack->pz4c != nullptr) {
    use_z4c = true;
    z4c_n = z4c_last;
    z4c_np1 = pmy_pack->pz4c->u0;
  }

  par_for("gr_boris_retry", DevExeSpace(), 0, nprtcl_thispack - 1,
  KOKKOS_LAMBDA(const int p) {
    if (retry_(p) == 0) {return;}
    int mb = pi(PGID, p) - gids;
    const Real mb_par[9] = {size.d_view(mb).x1min, size.d_view(mb).x1max,
                            size.d_view(mb).dx1,
                            size.d_view(mb).x2min, size.d_view(mb).x2max,
                            size.d_view(mb).dx2,
                            size.d_view(mb).x3min, size.d_view(mb).x3max,
                            size.d_view(mb).dx3};
    int ncell[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
    Real x_n[3] = {pr(IPX, p), pr(IPY, p), pr(IPZ, p)};
    Real u_n[3] = {pr(IPVX, p), pr(IPVY, p), pr(IPVZ, p)};

    Real x_np1[3] = {0.0}, u_np1[3] = {0.0};
    int gstat = kRejected;
    bool grid_ok = false;
    switch (ng) {
    case 2: {
      GeodesicPush<2, true> gp(x_n, u_n, mb, mb_par, ncell, dt_, adm_n, adm_np1,
                               use_z4c, z4c_n, z4c_np1);
      gstat = FixedPointIteration(gp, x_n, u_n, x_np1, u_np1);
      grid_ok = GridMetricValid<2>(adm_n, x_n, mb_par, ncell, mb);
      break;
    }
    case 3: {
      GeodesicPush<3, true> gp(x_n, u_n, mb, mb_par, ncell, dt_, adm_n, adm_np1,
                               use_z4c, z4c_n, z4c_np1);
      gstat = FixedPointIteration(gp, x_n, u_n, x_np1, u_np1);
      grid_ok = GridMetricValid<3>(adm_n, x_n, mb_par, ncell, mb);
      break;
    }
    case 4: {
      GeodesicPush<4, true> gp(x_n, u_n, mb, mb_par, ncell, dt_, adm_n, adm_np1,
                               use_z4c, z4c_n, z4c_np1);
      gstat = FixedPointIteration(gp, x_n, u_n, x_np1, u_np1);
      grid_ok = GridMetricValid<4>(adm_n, x_n, mb_par, ncell, mb);
      break;
    }
    }
    if (!grid_ok) {Kokkos::atomic_fetch_add(&nfail_(6), 1);}

    bool finite_out = gstat != kRejected
                   && Kokkos::isfinite(x_np1[0]) && Kokkos::isfinite(x_np1[1])
                   && Kokkos::isfinite(x_np1[2]) && Kokkos::isfinite(u_np1[0])
                   && Kokkos::isfinite(u_np1[1]) && Kokkos::isfinite(u_np1[2]);
    if (!finite_out) {
      // the low-order retry failed too: reject exactly as the push would have
      int islot = Kokkos::atomic_fetch_add(&nfail_(2), 1);
      if (islot < ndetail_) {
        Kokkos::printf("### WARNING gr_boris: geodesic substep REJECTED (trilinear retry "
                       "also invalid or non-finite); step not taken | tag=%d gid=%d "
                       "x=(% .6e,% .6e,% .6e) u_i=(% .6e,% .6e,% .6e) "
                       "grid_metric_ok=%d\n",
                       pi(PTAG, p), pi(PGID, p),
                       x_n[0], x_n[1], x_n[2], u_n[0], u_n[1], u_n[2],
                       static_cast<int>(grid_ok));
      }
      return;   // leave pr(IPX..)/pr(IPVX..) at their step-n values
    }
    Kokkos::atomic_fetch_add(&nfail_(5), 1);
    // A retry that only converged through the Euler branch is still a first-order step,
    // so it belongs in the same accuracy ledger as a high-order Euler fallback rather
    // than disappearing into the rescue count.
    if (gstat == kEuler) {Kokkos::atomic_fetch_add(&nfail_(0), 1);}
    pr(IPX, p) = x_np1[0];
    pr(IPY, p) = x_np1[1];
    pr(IPZ, p) = x_np1[2];
    pr(IPVX, p) = u_np1[0];
    pr(IPVY, p) = u_np1[1];
    pr(IPVZ, p) = u_np1[2];
  });
  return;
}

} // namespace particles
