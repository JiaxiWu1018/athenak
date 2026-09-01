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
//!       fallback on failure);
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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles.hpp"
#include "boris_utils.hpp"
#include "gr_monopole.hpp"
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
//! forward-Euler fallback.
//!
//! IMETH selects the gather scheme at compile time (values of ParticleInterpMethod):
//! 0 = Lagrange (historical; the instantiation contains no trilinear code, preserving
//! the default path bitwise), 1 = trilinear (padded weights; every LagrangeInterpolator
//! contraction below then evaluates the genuine 2x2x2 linear interpolant and its exact
//! derivative -- see CalcTriWghtAndDrv).

template <int NG, int IMETH = 0>
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

  KOKKOS_INLINE_FUNCTION
  void operator()(const Real xin[3], const Real uin[3], Real xout[3], Real uout[3],
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
    if constexpr (IMETH == static_cast<int>(ParticleInterpMethod::trilinear)) {
      CalcTriWghtAndDrv<NG>(x_mid, mb_par, ncell,
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
      alp_old = LagrangeInterpolator<NG>(z4c_old, z4c::Z4c::I_Z4C_ALPHA,
                                         interp_indcs, Lx, Ly, Lz);
      alp_new = LagrangeInterpolator<NG>(z4c_new, z4c::Z4c::I_Z4C_ALPHA,
                                         interp_indcs, Lx, Ly, Lz);
      for (int i = 0; i < 3; ++i) {
        beta_old[i] = LagrangeInterpolator<NG>(z4c_old, z4c::Z4c::I_Z4C_BETAX+i,
                                               interp_indcs, Lx, Ly, Lz);
        beta_new[i] = LagrangeInterpolator<NG>(z4c_new, z4c::Z4c::I_Z4C_BETAX+i,
                                               interp_indcs, Lx, Ly, Lz);
      }
    } else {
      alp_old = LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_ALPHA,
                                         interp_indcs, Lx, Ly, Lz);
      alp_new = LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_ALPHA,
                                         interp_indcs, Lx, Ly, Lz);
      for (int i = 0; i < 3; ++i) {
        beta_old[i] = LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_BETAX+i,
                                               interp_indcs, Lx, Ly, Lz);
        beta_new[i] = LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_BETAX+i,
                                               interp_indcs, Lx, Ly, Lz);
      }
    }
    alp = 0.5 * (alp_old + alp_new);
    for (int i = 0; i < 3; ++i) { beta[i] = 0.5 * (beta_old[i] + beta_new[i]); }
    for (int i = 0; i < 6; ++i) {
      g3d_old[i] = LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_GXX+i,
                                            interp_indcs, Lx, Ly, Lz);
      g3d_new[i] = LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_GXX+i,
                                            interp_indcs, Lx, Ly, Lz);
      g3d[i] = 0.5 * (g3d_old[i] + g3d_new[i]);
    }
    if (Euler) {
      alp = alp_old;
      for (int i = 0; i < 3; ++i) { beta[i] = beta_old[i]; }
      for (int i = 0; i < 6; ++i) { g3d[i] = g3d_old[i]; }
    }
    // (ii) transport velocity v^i = alp u^i / W - beta^i
    Real g3u[6] = {0.0};
    Real det = Primitive::GetDeterminant(g3d);
    Primitive::InvertMatrix(g3u, g3d, det);
    Real u_mid_u[3] = {0.0};
    Primitive::RaiseForm(u_mid_u, u_mid, g3u);
    Real Lorentz = std::sqrt(1.0 + Primitive::Contract(u_mid_u, u_mid));
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
      dalp_old[0] = LagrangeInterpolator<NG>(z4c_old, z4c::Z4c::I_Z4C_ALPHA,
                                             interp_indcs, dLx, Ly, Lz);
      dalp_old[1] = LagrangeInterpolator<NG>(z4c_old, z4c::Z4c::I_Z4C_ALPHA,
                                             interp_indcs, Lx, dLy, Lz);
      dalp_old[2] = LagrangeInterpolator<NG>(z4c_old, z4c::Z4c::I_Z4C_ALPHA,
                                             interp_indcs, Lx, Ly, dLz);
      dalp_new[0] = LagrangeInterpolator<NG>(z4c_new, z4c::Z4c::I_Z4C_ALPHA,
                                             interp_indcs, dLx, Ly, Lz);
      dalp_new[1] = LagrangeInterpolator<NG>(z4c_new, z4c::Z4c::I_Z4C_ALPHA,
                                             interp_indcs, Lx, dLy, Lz);
      dalp_new[2] = LagrangeInterpolator<NG>(z4c_new, z4c::Z4c::I_Z4C_ALPHA,
                                             interp_indcs, Lx, Ly, dLz);
      for (int i = 0; i < 3; ++i) {
        dbeta_old[0][i] = LagrangeInterpolator<NG>(z4c_old, z4c::Z4c::I_Z4C_BETAX+i,
                                                   interp_indcs, dLx, Ly, Lz);
        dbeta_old[1][i] = LagrangeInterpolator<NG>(z4c_old, z4c::Z4c::I_Z4C_BETAX+i,
                                                   interp_indcs, Lx, dLy, Lz);
        dbeta_old[2][i] = LagrangeInterpolator<NG>(z4c_old, z4c::Z4c::I_Z4C_BETAX+i,
                                                   interp_indcs, Lx, Ly, dLz);
        dbeta_new[0][i] = LagrangeInterpolator<NG>(z4c_new, z4c::Z4c::I_Z4C_BETAX+i,
                                                   interp_indcs, dLx, Ly, Lz);
        dbeta_new[1][i] = LagrangeInterpolator<NG>(z4c_new, z4c::Z4c::I_Z4C_BETAX+i,
                                                   interp_indcs, Lx, dLy, Lz);
        dbeta_new[2][i] = LagrangeInterpolator<NG>(z4c_new, z4c::Z4c::I_Z4C_BETAX+i,
                                                   interp_indcs, Lx, Ly, dLz);
      }
    } else {
      dalp_old[0] = LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_ALPHA,
                                             interp_indcs, dLx, Ly, Lz);
      dalp_old[1] = LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_ALPHA,
                                             interp_indcs, Lx, dLy, Lz);
      dalp_old[2] = LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_ALPHA,
                                             interp_indcs, Lx, Ly, dLz);
      dalp_new[0] = LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_ALPHA,
                                             interp_indcs, dLx, Ly, Lz);
      dalp_new[1] = LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_ALPHA,
                                             interp_indcs, Lx, dLy, Lz);
      dalp_new[2] = LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_ALPHA,
                                             interp_indcs, Lx, Ly, dLz);
      for (int i = 0; i < 3; ++i) {
        dbeta_old[0][i] = LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_BETAX+i,
                                                   interp_indcs, dLx, Ly, Lz);
        dbeta_old[1][i] = LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_BETAX+i,
                                                   interp_indcs, Lx, dLy, Lz);
        dbeta_old[2][i] = LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_BETAX+i,
                                                   interp_indcs, Lx, Ly, dLz);
        dbeta_new[0][i] = LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_BETAX+i,
                                                   interp_indcs, dLx, Ly, Lz);
        dbeta_new[1][i] = LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_BETAX+i,
                                                   interp_indcs, Lx, dLy, Lz);
        dbeta_new[2][i] = LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_BETAX+i,
                                                   interp_indcs, Lx, Ly, dLz);
      }
    }
    for (int i = 0; i < 3; ++i) {
      dalp[i] = 0.5 * (dalp_old[i] + dalp_new[i]);
      for (int j = 0; j < 3; ++j) {
        dbeta[i][j] = 0.5 * (dbeta_old[i][j] + dbeta_new[i][j]);
      }
    }
    LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_GXX,
                             interp_indcs, dLx, Ly, Lz, dg3u_old[0]);
    LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_GXX,
                             interp_indcs, Lx, dLy, Lz, dg3u_old[1]);
    LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_GXX,
                             interp_indcs, Lx, Ly, dLz, dg3u_old[2]);
    LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_GXX,
                             interp_indcs, dLx, Ly, Lz, dg3u_new[0]);
    LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_GXX,
                             interp_indcs, Lx, dLy, Lz, dg3u_new[1]);
    LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_GXX,
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
  }
};

//----------------------------------------------------------------------------------------
//! \fn InterpolateMonopoleScalars
//! \brief Interpolate one numerical 3+1 metric and project it onto the four scalar
//! fields of a spherical metric at a sampled direction.

template <int NG>
KOKKOS_INLINE_FUNCTION
void InterpolateMonopoleScalars(
    const DvceArray5D<Real> &adm_metric, const bool use_z4c,
    const DvceArray5D<Real> &z4c_metric, const int interp_indcs[4],
    const Real Lx[2*NG], const Real Ly[2*NG], const Real Lz[2*NG],
    const Real n[3], Real out[4]) {
  Real beta[3] = {0.0};
  if (use_z4c) {
    out[0] = LagrangeInterpolator<NG>(
        z4c_metric, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, Lx, Ly, Lz);
    for (int i = 0; i < 3; ++i) {
      beta[i] = LagrangeInterpolator<NG>(
          z4c_metric, z4c::Z4c::I_Z4C_BETAX+i, interp_indcs, Lx, Ly, Lz);
    }
  } else {
    out[0] = LagrangeInterpolator<NG>(
        adm_metric, adm::ADM::I_ADM_ALPHA, interp_indcs, Lx, Ly, Lz);
    for (int i = 0; i < 3; ++i) {
      beta[i] = LagrangeInterpolator<NG>(
          adm_metric, adm::ADM::I_ADM_BETAX+i, interp_indcs, Lx, Ly, Lz);
    }
  }
  Real g[6];
  for (int i = 0; i < 6; ++i) {
    g[i] = LagrangeInterpolator<NG>(
        adm_metric, adm::ADM::I_ADM_GXX+i, interp_indcs, Lx, Ly, Lz);
  }
  out[1] = beta[0]*n[0] + beta[1]*n[1] + beta[2]*n[2];
  out[2] = g[0]*n[0]*n[0] + g[3]*n[1]*n[1] + g[5]*n[2]*n[2]
         + 2.0*(g[1]*n[0]*n[1] + g[2]*n[0]*n[2] + g[4]*n[1]*n[2]);
  out[3] = 0.5*(g[0] + g[3] + g[5] - out[2]);
}

template <int NG>
void AccumulateGRBorisMonopoleProfiles(
    Particles *pp, MeshBlockPack *ppack, const DvceArray5D<Real> &adm_old,
    const DvceArray5D<Real> &adm_new, const bool use_z4c,
    const DvceArray5D<Real> &z4c_old, const DvceArray5D<Real> &z4c_new) {
  auto &indcs = ppack->pmesh->mb_indcs;
  int ncell[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
  auto &size = ppack->pmb->mb_size;
  int gids = ppack->gids;
  auto &pi = pp->prtcl_idata;
  auto &pr = pp->prtcl_rdata;
  auto &accum = pp->gr_boris_monopole_accum;
  int nr = pp->gr_boris_monopole_nr;
  int sample_stride = pp->gr_boris_monopole_sample_stride;
  Real dr = pp->gr_boris_monopole_dr;
  Real rmax = pp->gr_boris_monopole_rmax;
  Real c0 = pp->gr_boris_monopole_center[0];
  Real c1 = pp->gr_boris_monopole_center[1];
  Real c2 = pp->gr_boris_monopole_center[2];
  const bool tri = (pp->interp_method == ParticleInterpMethod::trilinear);

  par_for("gr_boris_monopole_average", DevExeSpace(), 0, pp->nprtcl_thispack - 1,
  KOKKOS_LAMBDA(const int p) {
    if ((pi(PTAG,p) % sample_stride) != 0) {return;}
    Real x[3] = {pr(IPX,p), pr(IPY,p), pr(IPZ,p)};
    Real xr[3] = {x[0]-c0, x[1]-c1, x[2]-c2};
    Real r = sqrt(xr[0]*xr[0] + xr[1]*xr[1] + xr[2]*xr[2]);
    if (!(r > 0.0) || r >= rmax) {return;}
    int b = static_cast<int>(floor(r/dr));
    if (b < 0 || b >= nr) {return;}
    Real n[3] = {xr[0]/r, xr[1]/r, xr[2]/r};
    int mb = pi(PGID,p) - gids;
    const Real mb_par[9] = {size.d_view(mb).x1min, size.d_view(mb).x1max,
                            size.d_view(mb).dx1,
                            size.d_view(mb).x2min, size.d_view(mb).x2max,
                            size.d_view(mb).dx2,
                            size.d_view(mb).x3min, size.d_view(mb).x3max,
                            size.d_view(mb).dx3};
    int interp_indcs[4] = {mb, -1, -1, -1};
    SetInterpIndices(x, mb_par, ncell, interp_indcs);
    Real Lx[2*NG] = {0.0}, Ly[2*NG] = {0.0}, Lz[2*NG] = {0.0};
    if (tri) {
      CalcTriWght<NG>(x, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
    } else {
      CalcInterpWght<NG>(x, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
    }
    Real old_scalar[4], new_scalar[4];
    InterpolateMonopoleScalars<NG>(
        adm_old, use_z4c, z4c_old, interp_indcs, Lx, Ly, Lz, n, old_scalar);
    InterpolateMonopoleScalars<NG>(
        adm_new, use_z4c, z4c_new, interp_indcs, Lx, Ly, Lz, n, new_scalar);
    for (int a = 0; a < 4; ++a) {
      Kokkos::atomic_add(&accum(MONO_AVG_ALPHA_OLD+a,b), old_scalar[a]);
      Kokkos::atomic_add(&accum(MONO_AVG_ALPHA_NEW+a,b), new_scalar[a]);
    }
    Kokkos::atomic_add(&accum(MONO_AVG_COUNT,b), static_cast<Real>(1.0));
  });
}

//----------------------------------------------------------------------------------------
//! \fn bool FixedPointIteration
//! \brief solve the implicit geodesic substep x=f(x) by fixed-point iteration. Returns
//! true on convergence (writes x,u); on non-finite iterates or non-convergence it falls
//! back to a forward-Euler step f(x0,u0,...,Euler=true) and returns false.

template<class F>
KOKKOS_INLINE_FUNCTION
bool FixedPointIteration(const F& f, const Real x0[3], const Real u0[3],
                         Real x[3], Real u[3], Real tol=1e-7, int maxIter=50) {
  Real x_new[3], u_new[3];
  for (int i = 0; i < 3; ++i) { x_new[i] = x0[i]; u_new[i] = u0[i]; }
  Real x_next[3], u_next[3];
  for (int iter = 0; iter < maxIter; ++iter) {
    f(x_new, u_new, x_next, u_next, false);
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
      return true;
    }
    for (int i = 0; i < 3; ++i) { x_new[i] = x_next[i]; u_new[i] = u_next[i]; }
  }
  f(x0, u0, x, u, true);   // forward-Euler fallback
  return false;
}

//----------------------------------------------------------------------------------------
//! \fn void Particles::BuildGRBorisMonopoleProfiles
//! \brief Angularly average the old/new live numerical metric over particle sample
//! positions, fill unsampled radial bins by interpolation, and construct inverse-metric
//! scalar profiles plus radial derivatives.

void Particles::BuildGRBorisMonopoleProfiles(
    const DvceArray5D<Real> &adm_old, const DvceArray5D<Real> &adm_new,
    bool use_z4c, const DvceArray5D<Real> &z4c_old,
    const DvceArray5D<Real> &z4c_new, bool equal_time) {
  if (!gr_boris_live_monopole) {return;}

  Kokkos::deep_copy(gr_boris_monopole_accum, static_cast<Real>(0.0));
  int ng = pmy_pack->pmesh->mb_indcs.ng;
  switch (ng) {
    case 2:
      AccumulateGRBorisMonopoleProfiles<2>(
          this, pmy_pack, adm_old, adm_new, use_z4c, z4c_old, z4c_new);
      break;
    case 3:
      AccumulateGRBorisMonopoleProfiles<3>(
          this, pmy_pack, adm_old, adm_new, use_z4c, z4c_old, z4c_new);
      break;
    case 4:
      AccumulateGRBorisMonopoleProfiles<4>(
          this, pmy_pack, adm_old, adm_new, use_z4c, z4c_old, z4c_new);
      break;
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "live-monopole profile supports NGHOST=2,3,4 only"
                << std::endl;
      std::exit(EXIT_FAILURE);
  }
  Kokkos::fence();

  auto havg = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), gr_boris_monopole_accum);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, havg.data(),
                N_GR_MONO_AVERAGES*gr_boris_monopole_nr,
                MPI_ATHENA_REAL, MPI_SUM, pbval_part->mpi_comm_part);
#endif

  const int nr = gr_boris_monopole_nr;
  std::vector<Real> count(nr);
  std::vector<Real> old_raw(4*nr), new_raw(4*nr);
  bool any_sample = false;
  for (int b = 0; b < nr; ++b) {
    count[b] = havg(MONO_AVG_COUNT,b);
    if (count[b] > 0.0) {
      any_sample = true;
      for (int a = 0; a < 4; ++a) {
        old_raw[a*nr+b] = havg(MONO_AVG_ALPHA_OLD+a,b)/count[b];
        new_raw[a*nr+b] = havg(MONO_AVG_ALPHA_NEW+a,b)/count[b];
      }
    }
  }
  if (!any_sample) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "live-monopole profile has no particle samples inside rmax"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Empty bins are expected below the innermost shell and can occur between the discrete
  // Lagrangian shells.  Interpolate between the nearest sampled bins; initially extend
  // the nearest valid value outside the sampled radial span.
  for (int b = 0; b < nr; ++b) {
    if (count[b] > 0.0) {continue;}
    int lo = b - 1;
    while (lo >= 0 && count[lo] <= 0.0) {--lo;}
    int hi = b + 1;
    while (hi < nr && count[hi] <= 0.0) {++hi;}
    for (int a = 0; a < 4; ++a) {
      if (lo >= 0 && hi < nr) {
        Real f = static_cast<Real>(b-lo)/static_cast<Real>(hi-lo);
        old_raw[a*nr+b] = (1.0-f)*old_raw[a*nr+lo] + f*old_raw[a*nr+hi];
        new_raw[a*nr+b] = (1.0-f)*new_raw[a*nr+lo] + f*new_raw[a*nr+hi];
      } else if (lo >= 0) {
        old_raw[a*nr+b] = old_raw[a*nr+lo];
        new_raw[a*nr+b] = new_raw[a*nr+lo];
      } else {
        old_raw[a*nr+b] = old_raw[a*nr+hi];
        new_raw[a*nr+b] = new_raw[a*nr+hi];
      }
    }
  }

  // A constant extension makes the centered derivative at the first/last sampled bin
  // half of its interior one-sided value.  Those are precisely the bins occupied by the
  // innermost/outermost particles.  Extrapolate only the adjacent guard bin from the
  // nearest two sampled bins so the derivative stencil remains supported; bins farther
  // outside the particle-supported span retain the bounded constant extension.
  int first_sample = -1;
  int second_sample = -1;
  int penultimate_sample = -1;
  int last_sample = -1;
  for (int b = 0; b < nr; ++b) {
    if (count[b] <= 0.0) {continue;}
    if (first_sample < 0) {
      first_sample = b;
    } else if (second_sample < 0) {
      second_sample = b;
    }
    penultimate_sample = last_sample;
    last_sample = b;
  }
  if (second_sample >= 0) {
    for (int a = 0; a < 4; ++a) {
      if (first_sample > 0) {
        Real old_slope = (old_raw[a*nr+second_sample] -
                          old_raw[a*nr+first_sample])/
                         static_cast<Real>(second_sample-first_sample);
        Real new_slope = (new_raw[a*nr+second_sample] -
                          new_raw[a*nr+first_sample])/
                         static_cast<Real>(second_sample-first_sample);
        old_raw[a*nr+first_sample-1] = old_raw[a*nr+first_sample] - old_slope;
        new_raw[a*nr+first_sample-1] = new_raw[a*nr+first_sample] - new_slope;
      }
      if (last_sample < nr-1 && penultimate_sample >= 0) {
        Real old_slope = (old_raw[a*nr+last_sample] -
                          old_raw[a*nr+penultimate_sample])/
                         static_cast<Real>(last_sample-penultimate_sample);
        Real new_slope = (new_raw[a*nr+last_sample] -
                          new_raw[a*nr+penultimate_sample])/
                         static_cast<Real>(last_sample-penultimate_sample);
        old_raw[a*nr+last_sample+1] = old_raw[a*nr+last_sample] + old_slope;
        new_raw[a*nr+last_sample+1] = new_raw[a*nr+last_sample] + new_slope;
      }
    }
  }

  HostArray2D<Real> hold("gr_boris_monopole_profile_old_host",
                         N_GR_MONO_PROFILE, nr);
  HostArray2D<Real> hnew("gr_boris_monopole_profile_new_host",
                         N_GR_MONO_PROFILE, nr);
  for (int b = 0; b < nr; ++b) {
    if (!(old_raw[0*nr+b] > 0.0) || !(new_raw[0*nr+b] > 0.0) ||
        !(old_raw[2*nr+b] > 0.0) || !(new_raw[2*nr+b] > 0.0) ||
        !(old_raw[3*nr+b] > 0.0) || !(new_raw[3*nr+b] > 0.0)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "non-positive lapse/spatial metric in live-monopole bin "
                << b << std::endl;
      std::exit(EXIT_FAILURE);
    }
    hold(MONO_ALPHA,b) = old_raw[0*nr+b];
    hold(MONO_BETA_R,b) = old_raw[1*nr+b];
    hold(MONO_GAMMA_R,b) = 1.0/old_raw[2*nr+b];
    hold(MONO_GAMMA_T,b) = 1.0/old_raw[3*nr+b];
    hnew(MONO_ALPHA,b) = new_raw[0*nr+b];
    hnew(MONO_BETA_R,b) = new_raw[1*nr+b];
    hnew(MONO_GAMMA_R,b) = 1.0/new_raw[2*nr+b];
    hnew(MONO_GAMMA_T,b) = 1.0/new_raw[3*nr+b];
  }

  const Real dr = gr_boris_monopole_dr;
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < nr; ++b) {
      int bm = (b > 0) ? b-1 : b;
      int bp = (b < nr-1) ? b+1 : b;
      Real denom = static_cast<Real>(bp-bm)*dr;
      hold(a+4,b) = (hold(a,bp)-hold(a,bm))/denom;
      hnew(a+4,b) = (hnew(a,bp)-hnew(a,bm))/denom;
    }
  }
  Kokkos::deep_copy(gr_boris_monopole_profile_old, hold);
  Kokkos::deep_copy(gr_boris_monopole_profile_new, hnew);
  Kokkos::fence();

  bool was_valid = gr_boris_monopole_profile_valid;
  gr_boris_monopole_profile_valid = true;
  int cycle = pmy_pack->pmesh->ncycle + (equal_time ? 0 : 1);
  Real profile_time = pmy_pack->pmesh->time
                    + (equal_time ? 0.0 : pmy_pack->pmesh->dt);
  if (gr_boris_monopole_profile_interval > 0 &&
      (!was_valid || cycle % gr_boris_monopole_profile_interval == 0) &&
      global_variable::my_rank == 0) {
    FILE *pfile = std::fopen(gr_boris_monopole_profile_fname.c_str(), "a");
    if (pfile == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "could not open live-monopole profile CSV '"
                << gr_boris_monopole_profile_fname << "'" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::fseek(pfile, 0, SEEK_END);
    if (std::ftell(pfile) == 0) {
      std::fprintf(pfile,
          "# time,cycle,bin,r_center,sample_count,"
          "alpha_old,beta_r_old,gamma_rr_cov_old,gamma_tt_cov_old,"
          "alpha_new,beta_r_new,gamma_rr_cov_new,gamma_tt_cov_new,"
          "gamma_rr_inv_new,gamma_tt_inv_new,dalpha_dr_new,dbeta_r_dr_new,"
          "dgamma_rr_inv_dr_new,dgamma_tt_inv_dr_new\n");
    }
    for (int b = 0; b < nr; ++b) {
      std::fprintf(pfile,
          "%.17g,%d,%d,%.17g,%.17g,"
          "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
          "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
          profile_time, cycle, b, (static_cast<Real>(b)+0.5)*dr, count[b],
          old_raw[0*nr+b], old_raw[1*nr+b], old_raw[2*nr+b], old_raw[3*nr+b],
          new_raw[0*nr+b], new_raw[1*nr+b], new_raw[2*nr+b], new_raw[3*nr+b],
          hnew(MONO_GAMMA_R,b), hnew(MONO_GAMMA_T,b),
          hnew(MONO_DALPHA_DR,b), hnew(MONO_DBETA_R_DR,b),
          hnew(MONO_DGAMMA_R_DR,b), hnew(MONO_DGAMMA_T_DR,b));
    }
    std::fclose(pfile);
  }
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
  // gather-scheme selector for the geodesic substep; the EM half-kicks below are
  // Lagrange-only (trilinear+MHD is rejected at construction in particles.cpp)
  const bool tri_gather = (interp_method == ParticleInterpMethod::trilinear);

  auto nfail_ = boris_nfail;          // bounded non-convergence diagnostic counters
  const int ndetail_ = kBorisDetail;

  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;

  // step-n snapshots (the *_last members) and step-(n+1) live arrays
  auto &w0_n = w0_last;
  auto &bcc0_n = bcc0_last;
  auto &adm_n = adm_last;
  DvceArray5D<Real> adm_np1 = gr_boris_freeze_metric
      ? adm_last : pmy_pack->padm->u_adm;

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
    z4c_np1 = gr_boris_freeze_metric ? z4c_last : pmy_pack->pz4c->u0;
  }

  if (gr_boris_live_monopole) {
    BuildGRBorisMonopoleProfiles(
        adm_n, adm_np1, use_z4c, z4c_n, z4c_np1, false);
  }
  auto mono_old = gr_boris_monopole_profile_old;
  auto mono_new = gr_boris_monopole_profile_new;
  int mono_nr = gr_boris_monopole_nr;
  Real mono_dr = gr_boris_monopole_dr;
  Real mono_c0 = gr_boris_monopole_center[0];
  Real mono_c1 = gr_boris_monopole_center[1];
  Real mono_c2 = gr_boris_monopole_center[2];
  bool use_monopole = gr_boris_live_monopole;

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

    Real x_n[3] = {pr(IPX, p), pr(IPY, p), pr(IPZ, p)};
    Real u_n[3] = {pr(IPVX, p), pr(IPVY, p), pr(IPVZ, p)};

    // ---- Step 1+2: first half EM kick at x^n (skipped when no MHD) ----
    // NOTE: the EM half-kick gathers below are Lagrange-only; interpolation=trilinear
    // with MHD present is rejected at construction (particles.cpp), so no run mixes
    // schemes here.
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
    bool find_root = false;
    if (use_monopole) {
      Real center[3] = {mono_c0, mono_c1, mono_c2};
      MonopoleGeodesicPush gp(
          x_n, u_p, mono_old, mono_new, mono_nr, mono_dr, dt_, center);
      find_root = FixedPointIteration(gp, x_n, u_p, x_np1, u_pp);
    } else if (tri_gather) {
      switch (ng) {
      case 2: {
        GeodesicPush<2,1> gp(x_n, u_p, mb, mb_par, ncell, dt_, adm_n, adm_np1, use_z4c,
                             z4c_n, z4c_np1);
        find_root = FixedPointIteration(gp, x_n, u_p, x_np1, u_pp);
        break;
      }
      case 3: {
        GeodesicPush<3,1> gp(x_n, u_p, mb, mb_par, ncell, dt_, adm_n, adm_np1, use_z4c,
                             z4c_n, z4c_np1);
        find_root = FixedPointIteration(gp, x_n, u_p, x_np1, u_pp);
        break;
      }
      case 4: {
        GeodesicPush<4,1> gp(x_n, u_p, mb, mb_par, ncell, dt_, adm_n, adm_np1, use_z4c,
                             z4c_n, z4c_np1);
        find_root = FixedPointIteration(gp, x_n, u_p, x_np1, u_pp);
        break;
      }
      }
    } else {
      switch (ng) {
      case 2: {
        GeodesicPush<2> gp(x_n, u_p, mb, mb_par, ncell, dt_, adm_n, adm_np1, use_z4c,
                           z4c_n, z4c_np1);
        find_root = FixedPointIteration(gp, x_n, u_p, x_np1, u_pp);
        break;
      }
      case 3: {
        GeodesicPush<3> gp(x_n, u_p, mb, mb_par, ncell, dt_, adm_n, adm_np1, use_z4c,
                           z4c_n, z4c_np1);
        find_root = FixedPointIteration(gp, x_n, u_p, x_np1, u_pp);
        break;
      }
      case 4: {
        GeodesicPush<4> gp(x_n, u_p, mb, mb_par, ncell, dt_, adm_n, adm_np1, use_z4c,
                           z4c_n, z4c_np1);
        find_root = FixedPointIteration(gp, x_n, u_p, x_np1, u_pp);
        break;
      }
      }
    }
    if (!find_root) {
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

    // ---- Step 6: write back ----
    pr(IPX, p) = x_np1[0];
    pr(IPY, p) = x_np1[1];
    pr(IPZ, p) = x_np1[2];
    pr(IPVX, p) = u_np1[0];
    pr(IPVY, p) = u_np1[1];
    pr(IPVZ, p) = u_np1[2];
  });

  // ---- bounded non-convergence summary: ONE line per rank per cycle ----------------
  // Read the device counter, report it, and reset for the next cycle. Failures are never
  // hidden: the count is exact and the per-rank running total is included.
  {
    auto hfail = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boris_nfail);
    int nfail = hfail(0);
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
      Kokkos::deep_copy(boris_nfail, 0);
    }
  }

  // snapshot the current fields/metric as step n for the next push (for a static
  // background these copies are a no-op; they carry the time level once the metric
  // evolves, Stage 4)
  if (use_mhd) {
    Kokkos::deep_copy(DevExeSpace(), w0_last, pmy_pack->pmhd->w0);
    Kokkos::deep_copy(DevExeSpace(), bcc0_last, pmy_pack->pmhd->bcc0);
  }
  if (!gr_boris_freeze_metric) {
    Kokkos::deep_copy(DevExeSpace(), adm_last, pmy_pack->padm->u_adm);
    if (use_z4c) {
      Kokkos::deep_copy(DevExeSpace(), z4c_last, pmy_pack->pz4c->u0);
    }
  }
  Kokkos::fence();

}

//----------------------------------------------------------------------------------------
//! \fn void Particles::GRBorisDiagnostics
//! \brief Evaluate instantaneous du_i/dt and d(x cross u)/dt at the current particle
//! state using the same metric interpolation and derivatives as the geodesic pusher.

void Particles::GRBorisDiagnostics() {
  if (!gr_boris_diagnostics || pmy_pack->padm == nullptr) {return;}
  if (nprtcl_thispack == 0) {return;}

  if (gr_boris_du_dt.extent_int(1) != nprtcl_thispack) {
    Kokkos::realloc(gr_boris_du_dt, 3, nprtcl_thispack);
    Kokkos::realloc(gr_boris_dL_dt, 3, nprtcl_thispack);
  }
  if (gr_boris_live_monopole &&
      gr_boris_raw_du_dt.extent_int(1) != nprtcl_thispack) {
    Kokkos::realloc(gr_boris_raw_du_dt, 3, nprtcl_thispack);
    Kokkos::realloc(gr_boris_raw_dL_dt, 3, nprtcl_thispack);
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ng = indcs.ng;
  auto &size = pmy_pack->pmb->mb_size;
  int gids = pmy_pack->gids;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto &du_dt = gr_boris_du_dt;
  auto &dL_dt = gr_boris_dL_dt;
  DvceArray5D<Real> adm_metric = gr_boris_freeze_metric
      ? adm_last : pmy_pack->padm->u_adm;
  DvceArray5D<Real> z4c_metric;
  bool use_z4c = (pmy_pack->pz4c != nullptr);
  if (use_z4c) {
    z4c_metric = gr_boris_freeze_metric ? z4c_last : pmy_pack->pz4c->u0;
  }
  if (gr_boris_live_monopole && !gr_boris_monopole_profile_valid) {
    BuildGRBorisMonopoleProfiles(
        adm_metric, adm_metric, use_z4c, z4c_metric, z4c_metric, true);
  }
  auto mono_profile = gr_boris_monopole_profile_new;
  int mono_nr = gr_boris_monopole_nr;
  Real mono_dr = gr_boris_monopole_dr;
  Real mono_c0 = gr_boris_monopole_center[0];
  Real mono_c1 = gr_boris_monopole_center[1];
  Real mono_c2 = gr_boris_monopole_center[2];
  bool use_monopole = gr_boris_live_monopole;
  auto &raw_du_dt = gr_boris_raw_du_dt;
  auto &raw_dL_dt = gr_boris_raw_dL_dt;
  const bool tri_gather = (interp_method == ParticleInterpMethod::trilinear);

  par_for("gr_boris_diagnostics", DevExeSpace(), 0, nprtcl_thispack - 1,
  KOKKOS_LAMBDA(const int p) {
    int mb = pi(PGID, p) - gids;
    const Real mb_par[9] = {size.d_view(mb).x1min, size.d_view(mb).x1max,
                            size.d_view(mb).dx1,
                            size.d_view(mb).x2min, size.d_view(mb).x2max,
                            size.d_view(mb).dx2,
                            size.d_view(mb).x3min, size.d_view(mb).x3max,
                            size.d_view(mb).dx3};
    int ncell[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
    Real x[3] = {pr(IPX,p), pr(IPY,p), pr(IPZ,p)};
    Real u[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
    Real x_plus_rate[3] = {0.0};
    Real u_plus_rate[3] = {0.0};

    // Equal old/new arrays remove temporal averaging. With dt=1 and Euler=true the
    // returned increments are the instantaneous coordinate velocity and covariant force.
    if (use_monopole) {
      Real center[3] = {mono_c0, mono_c1, mono_c2};
      MonopoleGeodesicPush evaluate(
          x, u, mono_profile, mono_profile, mono_nr, mono_dr, 1.0, center);
      evaluate(x, u, x_plus_rate, u_plus_rate, true);
    } else if (tri_gather) {
      if (ng == 2) {
        GeodesicPush<2,1> evaluate(
            x, u, mb, mb_par, ncell, 1.0, adm_metric, adm_metric,
            use_z4c, z4c_metric, z4c_metric);
        evaluate(x, u, x_plus_rate, u_plus_rate, true);
      } else if (ng == 3) {
        GeodesicPush<3,1> evaluate(
            x, u, mb, mb_par, ncell, 1.0, adm_metric, adm_metric,
            use_z4c, z4c_metric, z4c_metric);
        evaluate(x, u, x_plus_rate, u_plus_rate, true);
      } else {
        GeodesicPush<4,1> evaluate(
            x, u, mb, mb_par, ncell, 1.0, adm_metric, adm_metric,
            use_z4c, z4c_metric, z4c_metric);
        evaluate(x, u, x_plus_rate, u_plus_rate, true);
      }
    } else {
      if (ng == 2) {
        GeodesicPush<2> evaluate(
            x, u, mb, mb_par, ncell, 1.0, adm_metric, adm_metric,
            use_z4c, z4c_metric, z4c_metric);
        evaluate(x, u, x_plus_rate, u_plus_rate, true);
      } else if (ng == 3) {
        GeodesicPush<3> evaluate(
            x, u, mb, mb_par, ncell, 1.0, adm_metric, adm_metric,
            use_z4c, z4c_metric, z4c_metric);
        evaluate(x, u, x_plus_rate, u_plus_rate, true);
      } else {
        GeodesicPush<4> evaluate(
            x, u, mb, mb_par, ncell, 1.0, adm_metric, adm_metric,
            use_z4c, z4c_metric, z4c_metric);
        evaluate(x, u, x_plus_rate, u_plus_rate, true);
      }
    }

    Real v[3], g[3];
    for (int a = 0; a < 3; ++a) {
      v[a] = x_plus_rate[a] - x[a];
      g[a] = u_plus_rate[a] - u[a];
      du_dt(a,p) = g[a];
    }
    Real xr[3] = {x[0]-mono_c0, x[1]-mono_c1, x[2]-mono_c2};
    // d(x cross u)/dt = (dx/dt) cross u + x cross (du/dt).
    dL_dt(0,p) = v[1]*u[2] - v[2]*u[1] + xr[1]*g[2] - xr[2]*g[1];
    dL_dt(1,p) = v[2]*u[0] - v[0]*u[2] + xr[2]*g[0] - xr[0]*g[2];
    dL_dt(2,p) = v[0]*u[1] - v[1]*u[0] + xr[0]*g[1] - xr[1]*g[0];

    if (use_monopole) {
      Real x_raw[3] = {0.0}, u_raw[3] = {0.0};
      if (tri_gather) {
        if (ng == 2) {
          GeodesicPush<2,1> evaluate(
              x, u, mb, mb_par, ncell, 1.0, adm_metric, adm_metric,
              use_z4c, z4c_metric, z4c_metric);
          evaluate(x, u, x_raw, u_raw, true);
        } else if (ng == 3) {
          GeodesicPush<3,1> evaluate(
              x, u, mb, mb_par, ncell, 1.0, adm_metric, adm_metric,
              use_z4c, z4c_metric, z4c_metric);
          evaluate(x, u, x_raw, u_raw, true);
        } else {
          GeodesicPush<4,1> evaluate(
              x, u, mb, mb_par, ncell, 1.0, adm_metric, adm_metric,
              use_z4c, z4c_metric, z4c_metric);
          evaluate(x, u, x_raw, u_raw, true);
        }
      } else if (ng == 2) {
        GeodesicPush<2> evaluate(
            x, u, mb, mb_par, ncell, 1.0, adm_metric, adm_metric,
            use_z4c, z4c_metric, z4c_metric);
        evaluate(x, u, x_raw, u_raw, true);
      } else if (ng == 3) {
        GeodesicPush<3> evaluate(
            x, u, mb, mb_par, ncell, 1.0, adm_metric, adm_metric,
            use_z4c, z4c_metric, z4c_metric);
        evaluate(x, u, x_raw, u_raw, true);
      } else {
        GeodesicPush<4> evaluate(
            x, u, mb, mb_par, ncell, 1.0, adm_metric, adm_metric,
            use_z4c, z4c_metric, z4c_metric);
        evaluate(x, u, x_raw, u_raw, true);
      }
      Real vr[3], gr[3];
      for (int a = 0; a < 3; ++a) {
        vr[a] = x_raw[a] - x[a];
        gr[a] = u_raw[a] - u[a];
        raw_du_dt(a,p) = gr[a];
      }
      raw_dL_dt(0,p) = vr[1]*u[2] - vr[2]*u[1] + xr[1]*gr[2] - xr[2]*gr[1];
      raw_dL_dt(1,p) = vr[2]*u[0] - vr[0]*u[2] + xr[2]*gr[0] - xr[0]*gr[2];
      raw_dL_dt(2,p) = vr[0]*u[1] - vr[1]*u[0] + xr[0]*gr[1] - xr[1]*gr[0];
    }
  });
  Kokkos::fence();
}

} // namespace particles
