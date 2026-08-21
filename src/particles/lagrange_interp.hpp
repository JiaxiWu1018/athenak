#ifndef PARTICLES_LAGRANGE_INTERP_HPP_
#define PARTICLES_LAGRANGE_INTERP_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lagrange_interp.hpp
//! \brief device-side templated Lagrange interpolation of grid fields to a particle
//! position. Header-only, all KOKKOS_INLINE_FUNCTION so it can be called from
//! per-particle device kernels (the host/per-point utils/lagrange_interpolator.hpp is
//! unsuitable there).
//!
//! ORDER is tied to NGHOST: a centred stencil of width N=2*ORDER cell centres straddles
//! the particle. Callers dispatch on the active NGHOST (2,3,4) via a switch.
//!
//! Stage 2 added CalcInterpWghtAndDrv<ORDER> (basis weights AND their first derivatives,
//! for the geodesic / Christoffel force in gr_boris) and a per-node inverse-metric
//! vector overload of LagrangeInterpolator (interpolate gamma^{ij} by inverting the
//! 3-metric at each stencil node). The scalar overload (Stage 1) is unchanged.
//!
//! The CalcTrilinearWghtAndDrv / TrilinearInterpolator trio at the end is the ORDER=1
//! member of the same family, added for the gr_boris low-order fallback (branch
//! experiment/ST-trilinear-fallback). It is a separate two-node loop rather than the
//! ORDER template with zeroed weights on purpose: the point of the fallback is that a
//! bad value further out CANNOT enter the result, and a zero weight times a non-finite
//! nodal value is NaN, not zero.

#include <math.h>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/primitive-solver/geom_math.hpp"

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
//! cell centre (xmin + dx/2). grid[9] = {x1min,x1max,dx1, x2min,x2max,dx2,
//! x3min,x3max,dx3}.

KOKKOS_INLINE_FUNCTION
void SetInterpIndices(const Real *x0, const Real *grid, const int *ncell,
                      int *interp_indcs) {
  interp_indcs[1] = ClampInterpIndex(
      static_cast<int>(floor((x0[0] - (grid[0] + 0.5*grid[2])) / grid[2])), ncell[0]);
  interp_indcs[2] = ClampInterpIndex(
      static_cast<int>(floor((x0[1] - (grid[3] + 0.5*grid[5])) / grid[5])), ncell[1]);
  interp_indcs[3] = ClampInterpIndex(
      static_cast<int>(floor((x0[2] - (grid[6] + 0.5*grid[8])) / grid[8])), ncell[2]);
}

//----------------------------------------------------------------------------------------
//! \fn void CalcInterpWght
//! \brief Build the 1-D Lagrange basis weights L_a(x0) = prod_{b!=a} (x0 - x_b)/(x_a -
//! x_b) over the N=2*ORDER cell-centre nodes straddling the particle, in each direction.
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
//! \fn void CalcInterpWghtAndDrv
//! \brief Like CalcInterpWght, but also returns the first derivative of each 1-D Lagrange
//! basis weight w.r.t. its coordinate: dL_a/dx = sum_{j!=a} 1/(x_a-x_j) * prod_{k!=a,j}
//! (x0-x_k)/(x_a-x_k). Feeding dLx (in place of Lx) into LagrangeInterpolator then yields
//! the x-derivative of the interpolated field, which gr_boris uses to build the geodesic
//! force (gradients of alpha, beta^i, gamma^{ij}). Outputs L*[2*ORDER] and dL*[2*ORDER].

template <int ORDER>
KOKKOS_INLINE_FUNCTION
void CalcInterpWghtAndDrv(const Real *x0, const Real *grid, const int *ncell,
                          const int *interp_indcs, Real *Lx, Real *Ly, Real *Lz,
                          Real *dLx, Real *dLy, Real *dLz) {
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
    dLx[i] = 0.0; dLy[i] = 0.0; dLz[i] = 0.0;
    for (int j = 0; j < N; ++j) {
      if (j == i) continue;
      Lx[i] *= (x0[0] - x[j]) / (x[i] - x[j]);
      Ly[i] *= (x0[1] - y[j]) / (y[i] - y[j]);
      Lz[i] *= (x0[2] - z[j]) / (z[i] - z[j]);
      Real xterm = 1.0 / (x[i] - x[j]);
      Real yterm = 1.0 / (y[i] - y[j]);
      Real zterm = 1.0 / (z[i] - z[j]);
      for (int k = 0; k < N; ++k) {
        if (k == i || k == j) continue;
        xterm *= (x0[0] - x[k]) / (x[i] - x[k]);
        yterm *= (x0[1] - y[k]) / (y[i] - y[k]);
        zterm *= (x0[2] - z[k]) / (z[i] - z[k]);
      }
      dLx[i] += xterm;
      dLy[i] += yterm;
      dLz[i] += zterm;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real LagrangeInterpolator
//! \brief Interpolate scalar field variable nvar of the 5-D array u0 to the particle:
//! sum over the 2*ORDER^3 stencil of Lx_i*Ly_j*Lz_k * u0(m,nvar,k,j,i). The
//! allocated-index identity (interp_indcs[.] + idx + 1) holds because the first active
//! cell is at array index is == NGHOST == ORDER, so cell-centre-relative index a maps to
//! array index a+ORDER, and the stencil starts at a-ORDER+1 -> array index a+1.

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

//----------------------------------------------------------------------------------------
//! \fn void LagrangeInterpolator (vector / inverse-metric overload)
//! \brief Interpolate the inverse 3-metric gamma^{ij} to the weights L*. The 6 symmetric
//! components of the (covariant) 3-metric are stored contiguously starting at variable
//! nvar (= I_ADM_GXX). At EACH stencil node the stored gamma_{ij} is inverted to
//! gamma^{ij}, and the result is accumulated with the weights. Passing derivative weights
//! (dLx,...) gives the corresponding derivative of the interpolated gamma^{ij} field.
//! Output results[6] in the geom_math symmetric order {S11,S12,S13,S22,S23,S33}.

template <int ORDER>
KOKKOS_INLINE_FUNCTION
void LagrangeInterpolator(const DvceArray5D<Real> &u0, const int nvar,
                          const int *interp_indcs, const Real *Lx, const Real *Ly,
                          const Real *Lz, Real *results) {
  for (int m = 0; m < 6; ++m) { results[m] = 0.0; }
  constexpr int N = 2 * ORDER;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        Real weight = Lx[i] * Ly[j] * Lz[k];
        Real g3d[6] = {0.0};
        for (int m = 0; m < 6; ++m) {
          g3d[m] = u0(interp_indcs[0], nvar+m,
                      interp_indcs[3] + k + 1,
                      interp_indcs[2] + j + 1,
                      interp_indcs[1] + i + 1);
        }
        Real g3u[6] = {0.0};
        Primitive::InvertMatrix(g3u, g3d, Primitive::GetDeterminant(g3d));
        for (int m = 0; m < 6; ++m) { results[m] += weight * g3u[m]; }
      }
    }
  }
}


//----------------------------------------------------------------------------------------
//! \fn void CalcTrilinearWghtAndDrv
//! \brief The ORDER=1 member of the same Lagrange family: two-node (linear) basis weights
//! over the pair of cell centres bracketing the particle in each direction, together with
//! the exact first derivative of that same interpolant. Outputs L*[0..1] and dL*[0..1];
//! the caller's arrays are sized for the high-order stencil, and entries >= 2 are not
//! written and must not be read.
//!
//! Two properties are the reason this exists (see the gr_boris fallback):
//!   (a) the weights are in [0,1] and sum to 1 for any particle inside its owning
//!       MeshBlock, so the interpolated 3-metric is a CONVEX COMBINATION of the eight
//!       corner values and inherits their positive-definiteness;
//!   (b) the stencil reads only the eight immediately surrounding cell centres, so a bad
//!       grid value further away cannot contaminate the result.
//! By contrast a 2*ORDER-node Lagrange interpolant has weights of both signs and can
//! overshoot outside the range of its nodal values.
//!
//! The nodes are taken from CellCenterX, exactly as CalcInterpWght(AndDrv) does, so the
//! spacing (and hence the derivative) is the same quantity the high-order path uses.

KOKKOS_INLINE_FUNCTION
void CalcTrilinearWghtAndDrv(const Real *x0, const Real *grid, const int *ncell,
                             const int *interp_indcs, Real *Lx, Real *Ly, Real *Lz,
                             Real *dLx, Real *dLy, Real *dLz) {
  const Real xlo = CellCenterX(interp_indcs[1],     ncell[0], grid[0], grid[1]);
  const Real xhi = CellCenterX(interp_indcs[1] + 1, ncell[0], grid[0], grid[1]);
  const Real ylo = CellCenterX(interp_indcs[2],     ncell[1], grid[3], grid[4]);
  const Real yhi = CellCenterX(interp_indcs[2] + 1, ncell[1], grid[3], grid[4]);
  const Real zlo = CellCenterX(interp_indcs[3],     ncell[2], grid[6], grid[7]);
  const Real zhi = CellCenterX(interp_indcs[3] + 1, ncell[2], grid[6], grid[7]);
  const Real hx = xhi - xlo, hy = yhi - ylo, hz = zhi - zlo;
  Lx[0] = (xhi - x0[0]) / hx;  Lx[1] = (x0[0] - xlo) / hx;
  Ly[0] = (yhi - x0[1]) / hy;  Ly[1] = (x0[1] - ylo) / hy;
  Lz[0] = (zhi - x0[2]) / hz;  Lz[1] = (x0[2] - zlo) / hz;
  dLx[0] = -1.0 / hx;  dLx[1] = 1.0 / hx;
  dLy[0] = -1.0 / hy;  dLy[1] = 1.0 / hy;
  dLz[0] = -1.0 / hz;  dLz[1] = 1.0 / hz;
}

//----------------------------------------------------------------------------------------
//! \fn Real TrilinearInterpolator
//! \brief Scalar counterpart of LagrangeInterpolator for the two-node stencil built by
//! CalcTrilinearWghtAndDrv. The base index from SetInterpIndices is the cell-centre-
//! relative index of the node just below the particle, so the two nodes are at relative
//! indices idx and idx+1, i.e. ALLOCATED indices idx+NGHOST and idx+NGHOST+1 (the
//! high-order path's "+1" is the same expression with NGHOST-ORDER+1 and ORDER==NGHOST).
//! With SetInterpIndices clamping idx to [-1, ncell-1] the reads stay inside
//! [0, ncell+2*NGHOST-1] for any NGHOST >= 1: at most one ghost layer is touched.

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real TrilinearInterpolator(const DvceArray5D<Real> &u0, const int nvar,
                           const int *interp_indcs, const Real *Lx, const Real *Ly,
                           const Real *Lz) {
  Real result = 0.0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        Real weight = Lx[i] * Ly[j] * Lz[k];
        result += weight * u0(interp_indcs[0], nvar,
                              interp_indcs[3] + NGHOST + k,
                              interp_indcs[2] + NGHOST + j,
                              interp_indcs[1] + NGHOST + i);
      }
    }
  }
  return result;
}

//----------------------------------------------------------------------------------------
//! \fn void TrilinearInterpolator (vector / inverse-metric overload)
//! \brief Two-node counterpart of the LagrangeInterpolator inverse-metric overload: the
//! stored gamma_{ij} is inverted at each of the eight corners and the inverses are
//! combined with the weights. Passing the derivative weights gives the derivative of that
//! same interpolated gamma^{ij} field, which is what makes the fallback RHS consistent:
//! value and gradient come from one interpolant. Output in the geom_math symmetric order.

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
void TrilinearInterpolator(const DvceArray5D<Real> &u0, const int nvar,
                           const int *interp_indcs, const Real *Lx, const Real *Ly,
                           const Real *Lz, Real *results) {
  for (int m = 0; m < 6; ++m) { results[m] = 0.0; }
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        Real weight = Lx[i] * Ly[j] * Lz[k];
        Real g3d[6] = {0.0};
        for (int m = 0; m < 6; ++m) {
          g3d[m] = u0(interp_indcs[0], nvar+m,
                      interp_indcs[3] + NGHOST + k,
                      interp_indcs[2] + NGHOST + j,
                      interp_indcs[1] + NGHOST + i);
        }
        Real g3u[6] = {0.0};
        Primitive::InvertMatrix(g3u, g3d, Primitive::GetDeterminant(g3d));
        for (int m = 0; m < 6; ++m) { results[m] += weight * g3u[m]; }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void InverseMetricGradient
//! \brief d_i gamma^{jk} = -gamma^{ja} (d_i gamma_ab) gamma^{bk}, in the symmetric
//! 6-component storage {S11,S12,S13,S22,S23,S33} shared by geom_math and the ADM arrays.

KOKKOS_INLINE_FUNCTION
void InverseMetricGradient(Real dgu[6], const Real gu[6], const Real dgd[6]) {
  // symmetric-storage index of the (a,b) component, row-major: {xx,xy,xz,yy,yz,zz},
  // the same convention as the ADM arrays and geom_math's SpatialMetricIndex
  const int sidx[9] = {0, 1, 2,
                       1, 3, 4,
                       2, 4, 5};
  for (int j = 0; j < 3; ++j) {
    for (int k = j; k < 3; ++k) {
      Real sum = 0.0;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          sum += gu[sidx[3*j+a]] * dgd[sidx[3*a+b]] * gu[sidx[3*b+k]];
        }
      }
      dgu[sidx[3*j+k]] = -sum;
    }
  }
}

} // namespace particles
#endif // PARTICLES_LAGRANGE_INTERP_HPP_
