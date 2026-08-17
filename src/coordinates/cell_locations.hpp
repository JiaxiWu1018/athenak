#ifndef COORDINATES_CELL_LOCATIONS_HPP_
#define COORDINATES_CELL_LOCATIONS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cell_locations.hpp
//  \brief functions to compute locations on a uniform Cartesian grid
// They provide functionality of the Coordinates class in the C++ version of the code.
// Very similar to cc_pos.c function in C version of the code (Athena4.2)
// Not incorporated in Coordinates class so that they can be used anywhere (for example
// to compute locations of MeshBlocks in Mesh).

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn void LeftEdgeX()
// returns x-posn of left edge of i^th cell where index range [0,N] maps to [xmin,xmax]
// returns ghost cell posn if i outside range [0,N] (e.g. i=-1 is x-posn of first ghost
// cell). Averages linear interpolation from each side to symmetrize r.o. error

KOKKOS_INLINE_FUNCTION
static Real LeftEdgeX(int ith, int n, Real xmin, Real xmax) {
  Real x = (static_cast<Real>(ith)) / (static_cast<Real>(n));
  return (x*xmax - x*xmin) - (0.5*xmax - 0.5*xmin) + (0.5*xmin + 0.5*xmax);
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenterX()
// returns cell-center posn of i^th cell where index range [0,N] maps to [xmin,xmax]
// returns ghost cell posn if i outside range [0,N] (e.g. i=-1 is cc-posn of first ghost
// cell). Averages linear interpolation from each side to symmetrize r.o. error

KOKKOS_INLINE_FUNCTION
static Real CellCenterX(int ith, int n, Real xmin, Real xmax) {
  Real x = (static_cast<Real>(ith) + 0.5) / (static_cast<Real>(n));
  return (x*xmax - x*xmin) - (0.5*xmax - 0.5*xmin) + (0.5*xmin + 0.5*xmax);
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenterIndex()
// returns i-index of cell containing x position

// TODO(@user): set trap if out-of-range

KOKKOS_INLINE_FUNCTION
static int CellCenterIndex(Real x, int n, Real xmin, Real xmax) {
  return static_cast<int>(((x-xmin)/(xmax-xmin))*static_cast<Real>(n));
}

//----------------------------------------------------------------------------------------
//! \fn int LeftCenterIndex()
// returns the largest index i with CellCenterX(i) <= x, in [-1, n-1] for any
// x in [xmin, xmax). Unlike a pure arithmetic floor (which can disagree with the
// comparison predicates by half an ulp -- see the migration-offset discussion in
// bvals/prtcl_search.hpp), the float guess is corrected by direct comparison against
// CellCenterX, so callers that branch on the SAME comparisons (CIC deposition bands,
// weight clipping) are consistent with this index by construction.

KOKKOS_INLINE_FUNCTION
static int LeftCenterIndex(Real x, int n, Real xmin, Real xmax) {
  Real dx = (xmax - xmin)/static_cast<Real>(n);
  int idx = static_cast<int>(floor((x - xmin)/dx - 0.5));
  if (idx < -1) {idx = -1;}
  if (idx > n-1) {idx = n-1;}
  // the guess is off by at most one cell; one comparison step each way corrects it
  if (idx < n-1 && CellCenterX(idx+1, n, xmin, xmax) <= x) {
    ++idx;
  } else if (idx >= 0 && CellCenterX(idx, n, xmin, xmax) > x) {
    --idx;
  }
  return idx;
}

#endif // COORDINATES_CELL_LOCATIONS_HPP_
