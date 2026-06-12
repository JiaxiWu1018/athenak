#ifndef PARTICLES_BORIS_UTILS_HPP_
#define PARTICLES_BORIS_UTILS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file boris_utils.hpp
//! \brief Shared kernel for the relativistic Boris rotation, used by both the SR pusher
//! (boris_pusher.cpp) and the EM half-kicks of the GR pusher (gr_boris.cpp, in the local
//! orthonormal tetrad frame). The interpolation-index helpers (ClampInterpIndex /
//! SetInterpIndices) live in lagrange_interp.hpp and are NOT duplicated here.

#include <cmath>

#include <Kokkos_Core.hpp>

#include "athena.hpp"

namespace particles {

//----------------------------------------------------------------------------------------
//! \fn void FlatBorisPush
//! \brief One Boris update of the (special-relativistic) spatial 4-velocity u over a step
//! dt under fields E,B with charge-to-mass ratio qom: half electric kick -> magnetic
//! rotation -> half electric kick. u is the contravariant spatial 4-velocity (in flat
//! space u^i==u_i).
//! NOTE: tsqr = t0^2 + t1^2 + t2^2 (the prototype had a `*` typo here).

KOKKOS_INLINE_FUNCTION
void FlatBorisPush(Real u_pushed[3], const Real u[3], const Real E[3],
                   const Real B[3], const Real qom, const Real dt) {
  // half electric kick
  Real u_minus[3] = {0.0};
  for (int i = 0; i < 3; ++i) {
    u_minus[i] = u[i] + 0.5*qom*dt*E[i];
  }

  // magnetic rotation about t = (qom dt / 2 gamma_minus) B
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

  // half electric kick
  for (int i = 0; i < 3; ++i) {
    u_pushed[i] = u_plus[i] + 0.5*qom*dt*E[i];
  }
}

} // namespace particles

#endif // PARTICLES_BORIS_UTILS_HPP_
