#ifndef PARTICLES_BORIS_UTILS_HPP_
#define PARTICLES_BORIS_UTILS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file boris_utils.hpp
//! \brief Shared utility functions for Boris particle pushers.

#include <cmath>

#include <Kokkos_Core.hpp>

#include "athena.hpp"

namespace particles {

KOKKOS_INLINE_FUNCTION
int ClampInterpIndex(const int idx, const int ncell) {
  return (idx < -1) ? -1 : ((idx > ncell - 1) ? ncell - 1 : idx);
}

KOKKOS_INLINE_FUNCTION
void SetInterpIndices(const Real x[3], const Real mb_par[9], const int ncell[3],
                      int interp_indcs[4]) {
  const int raw_i = static_cast<int>(std::floor((x[0] - (mb_par[0] + 0.5*mb_par[2])) /
                                                mb_par[2]));
  const int raw_j = static_cast<int>(std::floor((x[1] - (mb_par[3] + 0.5*mb_par[5])) /
                                                mb_par[5]));
  const int raw_k = static_cast<int>(std::floor((x[2] - (mb_par[6] + 0.5*mb_par[8])) /
                                                mb_par[8]));
  interp_indcs[1] = ClampInterpIndex(raw_i, ncell[0]);
  interp_indcs[2] = ClampInterpIndex(raw_j, ncell[1]);
  interp_indcs[3] = ClampInterpIndex(raw_k, ncell[2]);
  if ((raw_i != interp_indcs[1]) || (raw_j != interp_indcs[2]) ||
      (raw_k != interp_indcs[3])) {
    Kokkos::printf("WARNING: particle interpolation index clipped: "
                   "x=(%.17g, %.17g, %.17g), raw=(%d, %d, %d), "
                   "clipped=(%d, %d, %d), ncell=(%d, %d, %d)\n",
                   x[0], x[1], x[2], raw_i, raw_j, raw_k,
                   interp_indcs[1], interp_indcs[2], interp_indcs[3],
                   ncell[0], ncell[1], ncell[2]);
  }
}

KOKKOS_INLINE_FUNCTION
void FlatBorisPush(Real u_pushed[3], const Real u[3], const Real E[3],
                   const Real B[3], const Real qom, const Real dt) {
  Real u_minus[3] = {0.0};
  for (int i = 0; i < 3; ++i) {
    u_minus[i] = u[i] + 0.5*qom*dt*E[i];
  }

  Real gamma_minus = std::sqrt(1.0 + u_minus[0]*u_minus[0] +
                               u_minus[1]*u_minus[1] + u_minus[2]*u_minus[2]);
  Real t[3] = {0.0};
  for (int i = 0; i < 3; ++i) {
    t[i] = 0.5*qom*dt/gamma_minus*B[i];
  }

  Real tsqr = t[0]*t[0] + t[1]*t[1] + t[2]*t[2];
  Real s[3] = {0.0};
  for (int i = 0; i < 3; ++i) {
    s[i] = 2.0/(1.0 + tsqr)*t[i];
  }

  Real s_dot_t = s[0]*t[0] + s[1]*t[1] + s[2]*t[2];
  Real s_dot_u_minus = s[0]*u_minus[0] + s[1]*u_minus[1] + s[2]*u_minus[2];
  Real u_plus[3] = {0.0};
  u_plus[0] = u_minus[0] + u_minus[1]*s[2] - u_minus[2]*s[1] -
              s_dot_t*u_minus[0] + s_dot_u_minus*t[0];
  u_plus[1] = u_minus[1] + u_minus[2]*s[0] - u_minus[0]*s[2] -
              s_dot_t*u_minus[1] + s_dot_u_minus*t[1];
  u_plus[2] = u_minus[2] + u_minus[0]*s[1] - u_minus[1]*s[0] -
              s_dot_t*u_minus[2] + s_dot_u_minus*t[2];

  for (int i = 0; i < 3; ++i) {
    u_pushed[i] = u_plus[i] + 0.5*qom*dt*E[i];
  }
}

} // namespace particles

#endif // PARTICLES_BORIS_UTILS_HPP_
