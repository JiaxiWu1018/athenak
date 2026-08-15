#ifndef PARTICLES_LAGRANGE_INTERP_HPP_
#define PARTICLES_LAGRANGE_INTERP_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lagrange_interp.hpp
//! \brief device-side templated Lagrange interpolation of grid fields to a particle
//! position. Header-only, all KOKKOS_INLINE_FUNCTION so it can be called from per-particle
//! device kernels (the host/per-point utils/lagrange_interpolator.hpp is unsuitable there).
//!
//! ORDER is tied to NGHOST: a centred stencil of width N=2*ORDER cell centres straddles the
//! particle. Callers dispatch on the active NGHOST (2,3,4) via a switch.
//!
//! STAGE 2 will add CalcInterpWghtAndDrv<ORDER> (derivative weights for the geodesic /
//! Christoffel force) and a per-node inverse-metric vector overload. Stage 1 only needs to
//! interpolate scalar metric components (it inverts the 3-metric once, at the particle).

#include <math.h>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"

namespace particles {

//----------------------------------------------------------------------------------------
//! \fn int ClampInterpIndex
//! \brief Clamp the base (cell-centre-relative) stencil index so the 2*ORDER-wide window
//! [idx+1, idx+2*ORDER] stays inside the allocated array extent [0, ncell+2*ORDER-1]
//! (active cells + ghosts, with NGHOST==ORDER). Valid base range is [-1, ncell-1]. For a
//! particle inside its owning MeshBlock this is a no-op; it is a safety net against an
//! out-of-block position (floating-point edge cases / pre-migration drift).

KOKKOS_INLINE_FUNCTION
int ClampInterpIndex(const int idx, const int ncell) {
  const int lo = -1;
  const int hi = ncell - 1;
  return (idx < lo) ? lo : ((idx > hi) ? hi : idx);
}

//----------------------------------------------------------------------------------------
//! \fn void SetInterpIndices
//! \brief Compute the clamped base stencil index in each direction for a particle at x0.
//! interp_indcs[0] is the MeshBlock index (set by the caller); [1],[2],[3] are the i,j,k
//! cell indices of the cell centre just below the particle, relative to the first active
//! cell centre (xmin + dx/2). grid[9] = {x1min,x1max,dx1, x2min,x2max,dx2, x3min,x3max,dx3}.

KOKKOS_INLINE_FUNCTION
void SetInterpIndices(const Real *x0, const Real *grid, const int *ncell, int *interp_indcs) {
  interp_indcs[1] = ClampInterpIndex(
      static_cast<int>(floor((x0[0] - (grid[0] + 0.5*grid[2])) / grid[2])), ncell[0]);
  interp_indcs[2] = ClampInterpIndex(
      static_cast<int>(floor((x0[1] - (grid[3] + 0.5*grid[5])) / grid[5])), ncell[1]);
  interp_indcs[3] = ClampInterpIndex(
      static_cast<int>(floor((x0[2] - (grid[6] + 0.5*grid[8])) / grid[8])), ncell[2]);
}

//----------------------------------------------------------------------------------------
//! \fn void CalcInterpWght
//! \brief Build the 1-D Lagrange basis weights L_a(x0) = prod_{b!=a} (x0 - x_b)/(x_a - x_b)
//! over the N=2*ORDER cell-centre nodes straddling the particle, in each direction.
//! Outputs Lx[2*ORDER], Ly[2*ORDER], Lz[2*ORDER].

template <int ORDER>
KOKKOS_INLINE_FUNCTION
void CalcInterpWght(const Real *x0, const Real *grid, const int *ncell,
                    const int *interp_indcs, Real *Lx, Real *Ly, Real *Lz) {
  constexpr int N = 2 * ORDER;
  Real xmin = grid[0], xmax = grid[1];
  Real ymin = grid[3], ymax = grid[4];
  Real zmin = grid[6], zmax = grid[7];
  Real x[N] = {0.0}, y[N] = {0.0}, z[N] = {0.0};
  for (int i = 0; i < N; ++i) {
    x[i] = CellCenterX(interp_indcs[1] - ORDER + 1 + i, ncell[0], xmin, xmax);
    y[i] = CellCenterX(interp_indcs[2] - ORDER + 1 + i, ncell[1], ymin, ymax);
    z[i] = CellCenterX(interp_indcs[3] - ORDER + 1 + i, ncell[2], zmin, zmax);
  }
  for (int i = 0; i < N; ++i) {
    Lx[i] = 1.0; Ly[i] = 1.0; Lz[i] = 1.0;
    for (int j = 0; j < N; ++j) {
      if (i == j) continue;
      Lx[i] *= (x0[0] - x[j]) / (x[i] - x[j]);
      Ly[i] *= (x0[1] - y[j]) / (y[i] - y[j]);
      Lz[i] *= (x0[2] - z[j]) / (z[i] - z[j]);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real LagrangeInterpolator
//! \brief Interpolate scalar field variable nvar of the 5-D array u0 to the particle:
//! sum over the 2*ORDER^3 stencil of Lx_i*Ly_j*Lz_k * u0(m,nvar,k,j,i). The allocated-index
//! identity (interp_indcs[.] + idx + 1) holds because the first active cell is at array
//! index is == NGHOST == ORDER, so cell-centre-relative index a maps to array index a+ORDER,
//! and the stencil starts at a-ORDER+1 -> array index a+1.

template <int ORDER>
KOKKOS_INLINE_FUNCTION
Real LagrangeInterpolator(const DvceArray5D<Real> &u0, const int nvar,
                          const int *interp_indcs, const Real *Lx, const Real *Ly,
                          const Real *Lz) {
  constexpr int N = 2 * ORDER;
  Real result = 0.0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        Real weight = Lx[i] * Ly[j] * Lz[k];
        result += weight * u0(interp_indcs[0], nvar,
                              interp_indcs[3] + k + 1,
                              interp_indcs[2] + j + 1,
                              interp_indcs[1] + i + 1);
      }
    }
  }
  return result;
}

} // namespace particles
#endif // PARTICLES_LAGRANGE_INTERP_HPP_
