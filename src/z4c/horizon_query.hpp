#ifndef Z4C_HORIZON_QUERY_HPP_
#define Z4C_HORIZON_QUERY_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file horizon_query.hpp
//! \brief device-callable point-vs-apparent-horizon test against a FastFlow surface.
//!
//! FastFlow represents surface n as a spectral graph over the unit sphere, measured from
//! FastFlow::center,
//!
//!   R(theta,phi) = sum_l a0[l] Y_l0 + sum_l sum_{m=1..l} sqrt(2) ( ac[lm] Re Y_lm
//!                                                               + as[lm] Im Y_lm ),
//!
//! which is the sum RadiiFromSphericalHarmonics evaluates at the Gauss-Legendre points,
//! with the sqrt(2) folded into the Yc/Ys stored by ComputeSphericalHarmonics and applied
//! explicitly here. Reusing the finder's own SphericalHarm() is deliberate: the surface
//! must not be re-derived from a second harmonic implementation.
//!
//! A consumer stages the surfaces into two flat views (host-filled, device-read):
//!
//!   par (nhorizon, NAH_PAR)  {cx, cy, cz, rmin, rmax, valid}
//!   coef(nhorizon, ncoef)    [ a0(0..lmax) | ac(0..lmpoints-1) | as(0..lmpoints-1) ]
//!                            with ncoef = (lmax+1) + 2*lmpoints, lmpoints = (lmax+1)^2
//!
//! and calls AHInside(). rmin/rmax are the angular extrema of that same surface, so
//! a point closer than rmin is inside and one farther than rmax is outside without
//! touching the harmonic sum, which costs O(lmax^2) SphericalHarm() calls.

#include "athena.hpp"
#include "utils/spherical_harm.hpp"

// column layout of the staged per-horizon parameter view
enum AHParIndex {IAHCX=0, IAHCY=1, IAHCZ=2, IAHRMIN=3, IAHRMAX=4, IAHVALID=5, NAH_PAR=6};

// "far outside" sentinel for the containment ratio: finite (not inf/NaN) so it is safe to
// reduce over and readable in the death-record CSV, and representable in single precision
#define AH_CRIT_FAR (static_cast<Real>(1.0e30))

//----------------------------------------------------------------------------------------
//! \fn Real AHSurfaceRadius
//! \brief evaluate R(theta,phi) of staged horizon h from its spectral coefficients

template <class CoefView>
KOKKOS_INLINE_FUNCTION
Real AHSurfaceRadius(const CoefView &coef, const int h, const int lmax,
                     const int lmpoints, const Real theta, const Real phi) {
  const Real sqrt2 = Kokkos::sqrt(2.0);
  const int lmax1 = lmax + 1;
  Real r = 0.0;
  for (int l = 0; l <= lmax; ++l) {
    Real ylmR, ylmI;
    SphericalHarm(&ylmR, &ylmI, l, 0, theta, phi);
    r += coef(h, l) * ylmR;
    for (int m = 1; m <= l; ++m) {
      SphericalHarm(&ylmR, &ylmI, l, m, theta, phi);
      const int l1 = lmindex(l, m, lmax);
      r += sqrt2 * (coef(h, lmax1 + l1) * ylmR + coef(h, lmax1 + lmpoints + l1) * ylmI);
    }
  }
  return r;
}

//----------------------------------------------------------------------------------------
//! \fn bool AHInside
//! \brief is (x1,x2,x3) inside horizon h? `crit` returns the containment ratio r/R in
//! [0,inf): < 1 inside, >= 1 outside. The surface is a graph over the sphere about
//! `center`, so r < R(theta,phi) is a complete interior test for it.

template <class ParView, class CoefView>
KOKKOS_INLINE_FUNCTION
bool AHInside(const ParView &par, const CoefView &coef, const int h, const int lmax,
              const int lmpoints, const Real x1, const Real x2, const Real x3,
              Real &crit) {
  crit = AH_CRIT_FAR;
  if (par(h, IAHVALID) <= 0.0) {return false;}

  const Real dx = x1 - par(h, IAHCX);
  const Real dy = x2 - par(h, IAHCY);
  const Real dz = x3 - par(h, IAHCZ);
  const Real r = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
  const Real rmin = par(h, IAHRMIN);
  const Real rmax = par(h, IAHRMAX);

  // Bracket first. rmin <= R(theta,phi) <= rmax, so both branches are exact, and both
  // report the ratio biased AWAY from the verdict they return: r/rmin is an upper bound
  // on the true r/R (still < 1 here) and r/rmax a lower bound (still >= 1 there).
  if (r < rmin) {crit = r/rmin; return true;}
  if (r >= rmax) {crit = (rmax > 0.0) ? r/rmax : AH_CRIT_FAR; return false;}

  // In the shell: evaluate the actual angular surface. r > rmin > 0 here, so theta/phi
  // are well defined (r == 0 can only reach the rmin branch above).
  const Real theta = Kokkos::acos(Kokkos::fmax(-1.0, Kokkos::fmin(1.0, dz/r)));
  const Real phi = Kokkos::atan2(dy, dx);
  const Real R = AHSurfaceRadius(coef, h, lmax, lmpoints, theta, phi);
  crit = (R > 0.0) ? r/R : AH_CRIT_FAR;
  return (r < R);
}

#endif // Z4C_HORIZON_QUERY_HPP_
