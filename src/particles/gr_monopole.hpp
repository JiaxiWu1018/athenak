//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_monopole.hpp
//! \brief Rotationally invariant, spherically averaged 3+1 metric used by the optional
//! live-monopole particle diagnostic.

#ifndef PARTICLES_GR_MONOPOLE_HPP_
#define PARTICLES_GR_MONOPOLE_HPP_

#include <cmath>

#include "athena.hpp"

namespace particles {

// The profile stores the four independent scalars of a spherical 3+1 metric and their
// radial derivatives.  GAMMA_R and GAMMA_T are inverse spatial-metric eigenvalues:
// gamma^{ij} = GAMMA_T delta^{ij} + (GAMMA_R-GAMMA_T) n^i n^j.
enum GRMonopoleProfileIndex {
  MONO_ALPHA = 0,
  MONO_BETA_R = 1,
  MONO_GAMMA_R = 2,
  MONO_GAMMA_T = 3,
  MONO_DALPHA_DR = 4,
  MONO_DBETA_R_DR = 5,
  MONO_DGAMMA_R_DR = 6,
  MONO_DGAMMA_T_DR = 7,
  N_GR_MONO_PROFILE = 8
};

// Raw angular averages accumulated before inversion/differentiation.
enum GRMonopoleAverageIndex {
  MONO_AVG_ALPHA_OLD = 0,
  MONO_AVG_BETA_R_OLD = 1,
  MONO_AVG_GAMMA_RR_OLD = 2,
  MONO_AVG_GAMMA_TT_OLD = 3,
  MONO_AVG_ALPHA_NEW = 4,
  MONO_AVG_BETA_R_NEW = 5,
  MONO_AVG_GAMMA_RR_NEW = 6,
  MONO_AVG_GAMMA_TT_NEW = 7,
  MONO_AVG_COUNT = 8,
  N_GR_MONO_AVERAGES = 9
};

KOKKOS_INLINE_FUNCTION
void InterpolateGRMonopoleProfile(const DvceArray2D<Real> &profile, const int nr,
                                  const Real dr, const Real r,
                                  Real value[N_GR_MONO_PROFILE]) {
  Real q = r/dr - 0.5;
  int i0 = static_cast<int>(floor(q));
  Real f = q - static_cast<Real>(i0);
  if (i0 < 0) {
    i0 = 0;
    f = 0.0;
  } else if (i0 >= nr - 1) {
    i0 = nr - 1;
    f = 0.0;
  }
  int i1 = (i0 < nr - 1) ? i0 + 1 : i0;
  for (int n = 0; n < N_GR_MONO_PROFILE; ++n) {
    value[n] = (1.0 - f)*profile(n, i0) + f*profile(n, i1);
  }
}

//----------------------------------------------------------------------------------------
//! \struct MonopoleGeodesicPush
//! \brief Implicit geodesic substep in a time-dependent spherical 3+1 metric.
//!
//! For gamma^{ij}=B delta^{ij}+(A-B)n^i n^j and beta^i=beta_r n^i, the angular
//! derivatives of n^i generate tangential terms in both dx^i/dt and du_i/dt.  They
//! cancel exactly in d(x cross u)/dt.  Keeping these terms is essential: projecting an
//! otherwise nonspherical Cartesian force onto the radial direction is not the same
//! spherical Hamiltonian and need not preserve angular momentum in the full update.

struct MonopoleGeodesicPush {
  const Real *x_old, *u_old;
  const DvceArray2D<Real> profile_old, profile_new;
  const int nr;
  const Real dr, dt;
  const Real *center;

  KOKKOS_INLINE_FUNCTION
  MonopoleGeodesicPush(const Real x_[3], const Real u_[3],
                       const DvceArray2D<Real> &profile_old_,
                       const DvceArray2D<Real> &profile_new_,
                       const int nr_, const Real dr_, const Real dt_,
                       const Real center_[3])
      : x_old(x_), u_old(u_), profile_old(profile_old_), profile_new(profile_new_),
        nr(nr_), dr(dr_), dt(dt_), center(center_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Real xin[3], const Real uin[3], Real xout[3], Real uout[3],
                  bool Euler) const {
    Real x_mid[3], u_mid[3], xr[3];
    Real r2 = 0.0;
    for (int i = 0; i < 3; ++i) {
      x_mid[i] = 0.5*(xin[i] + x_old[i]);
      u_mid[i] = 0.5*(uin[i] + u_old[i]);
      xr[i] = x_mid[i] - center[i];
      r2 += xr[i]*xr[i];
    }
    const Real tiny = 1.0e-14;
    Real r = sqrt(r2);
    Real rsafe = (r > tiny) ? r : tiny;
    Real n[3] = {xr[0]/rsafe, xr[1]/rsafe, xr[2]/rsafe};

    Real po[N_GR_MONO_PROFILE], pn[N_GR_MONO_PROFILE];
    InterpolateGRMonopoleProfile(profile_old, nr, dr, r, po);
    InterpolateGRMonopoleProfile(profile_new, nr, dr, r, pn);
    Real p[N_GR_MONO_PROFILE];
    for (int a = 0; a < N_GR_MONO_PROFILE; ++a) {
      p[a] = Euler ? po[a] : 0.5*(po[a] + pn[a]);
    }

    const Real alpha = p[MONO_ALPHA];
    const Real beta_r = p[MONO_BETA_R];
    const Real A = p[MONO_GAMMA_R];
    const Real B = p[MONO_GAMMA_T];
    const Real dA = p[MONO_DGAMMA_R_DR];
    const Real dB = p[MONO_DGAMMA_T_DR];
    const Real C = A - B;
    const Real dC = dA - dB;

    Real q = 0.0, usq = 0.0;
    for (int i = 0; i < 3; ++i) {
      q += n[i]*u_mid[i];
      usq += u_mid[i]*u_mid[i];
    }
    Real u_con[3];
    for (int i = 0; i < 3; ++i) {u_con[i] = B*u_mid[i] + C*q*n[i];}
    Real W = sqrt(1.0 + B*usq + C*q*q);

    // Coordinate transport velocity.
    for (int i = 0; i < 3; ++i) {
      Real v = alpha*u_con[i]/W - beta_r*n[i];
      xout[i] = x_old[i] + dt*v;
    }

    // Covariant geodesic force.  The last terms are the angular derivatives of the
    // radial shift and inverse spatial metric, written after contraction with u_j u_k.
    Real radial_metric_derivative = dB*usq + dC*q*q;
    for (int i = 0; i < 3; ++i) {
      Real u_tangent_i = u_mid[i] - q*n[i];
      Real u_dbeta = p[MONO_DBETA_R_DR]*q*n[i]
                   + beta_r*u_tangent_i/rsafe;
      Real u_dgamma_u = n[i]*radial_metric_derivative
                      + 2.0*C*q*u_tangent_i/rsafe;
      Real g = -W*p[MONO_DALPHA_DR]*n[i] + u_dbeta
             - 0.5*alpha*u_dgamma_u/W;
      uout[i] = u_old[i] + dt*g;
    }
  }
};

} // namespace particles

#endif // PARTICLES_GR_MONOPOLE_HPP_
