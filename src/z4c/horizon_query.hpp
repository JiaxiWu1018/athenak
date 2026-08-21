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
//! FastFlow represents surface n as a spectral graph over the unit sphere,
//!
//!   R(theta,phi) = sum_l a0[l] Y_l0 + sum_l sum_{m=1..l} sqrt(2) ( ac[lm] Re Y_lm
//!                                                               + as[lm] Im Y_lm ),
//!
//! measured from FastFlow::center. That is exactly the sum FastFlow itself evaluates at
//! the Gauss-Legendre collocation points (fastflow.cpp, RadiiFromSphericalHarmonics),
//! with the sqrt(2) folded into the stored Yc/Ys there (fastflow.cpp,
//! ComputeSphericalHarmonics) and applied explicitly here instead. Using the same
//! SphericalHarm() as the finder is deliberate: the surface must not be re-derived from a
//! second, possibly differently normalized, harmonic implementation.
//!
//! A consumer stages the surfaces into two flat views (host-filled, device-read):
//!
//!   par (nhorizon, NAH_PAR)  {cx, cy, cz, rmin, rmax, valid}
//!   coef(nhorizon, ncoef)    [ a0(0..lmax) | ac(0..lmpoints-1) | as(0..lmpoints-1) ]
//!                            with ncoef = (lmax+1) + 2*lmpoints, lmpoints = (lmax+1)^2
//!
//! and calls AHContainment(). rmin/rmax are the angular extrema of the SAME surface, so
//! a point closer than rmin is inside and one farther than rmax is outside without
//! touching the harmonic sum -- which matters because the sum costs O(lmax^2) calls to
//! SphericalHarm(), each of which runs its own Wigner-d loop with pow() and factorials.
//! In a collapse run the overwhelming majority of particles are far outside, so the
//! bracket removes essentially all of the cost.
//!
//! The surface is star-shaped about `center` by construction (it IS a graph over the
//! sphere), so "r < R(theta,phi)" is a complete interior test for it. Whether that
//! surface is a horizon at all is FastFlow's business, not this header's.

#include "athena.hpp"
#include "utils/spherical_harm.hpp"

// column layout of the staged per-horizon parameter view
enum AHParIndex {IAHCX=0, IAHCY=1, IAHCZ=2, IAHRMIN=3, IAHRMAX=4, IAHVALID=5, NAH_PAR=6};

// outcome of a point-vs-horizon test
enum AHContain {kAHOutside=0, kAHInside=1};

// "unboundedly far outside" sentinel for the containment ratio. A finite value (not
// inf/NaN) keeps the ratio safe to reduce over and readable in the death-record CSV;
// 1e30 is representable in single precision too.
#define AH_CRIT_FAR (static_cast<Real>(1.0e30))

//----------------------------------------------------------------------------------------
//! \fn Real AHSurfaceRadius
//! \brief evaluate R(theta,phi) of staged horizon h from its spectral coefficients.

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
//! \fn int AHContainment
//! \brief is (x1,x2,x3) inside horizon h, shrunk by the fractional margin `margin`?
//!
//! `margin` in [0,1) tests against (1-margin)*R instead of R, i.e. a point must be that
//! fraction further in than the surface before it counts as inside. margin = 0 is the
//! horizon itself. `use_surface = false` falls back to the largest sphere that fits
//! inside the surface (radius rmin), which is strictly more conservative at every angle.
//!
//! `crit` returns the containment ratio r/R_eff in [0, inf): < 1 inside, >= 1 outside.
//! It is written for every point, inside or not, so the caller can log how deep a
//! destroyed particle was and how close a surviving one came.

template <class ParView, class CoefView>
KOKKOS_INLINE_FUNCTION
int AHContainment(const ParView &par, const CoefView &coef, const int h,
                  const int lmax, const int lmpoints, const bool use_surface,
                  const Real margin, const Real x1, const Real x2, const Real x3,
                  Real &crit) {
  crit = AH_CRIT_FAR;
  if (par(h, IAHVALID) <= 0.0) {return kAHOutside;}

  const Real dx = x1 - par(h, IAHCX);
  const Real dy = x2 - par(h, IAHCY);
  const Real dz = x3 - par(h, IAHCZ);
  const Real r = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);

  const Real shrink = 1.0 - margin;
  const Real rmin_eff = shrink * par(h, IAHRMIN);
  if (!use_surface) {
    crit = (rmin_eff > 0.0) ? r/rmin_eff : AH_CRIT_FAR;
    return (r < rmin_eff) ? kAHInside : kAHOutside;
  }

  // Bracket first: rmin/rmax are the angular extrema of this same surface, so these two
  // tests are exact and skip the harmonic sum for all but a thin shell.
  const Real rmax_eff = shrink * par(h, IAHRMAX);
  // Both bracket branches report the containment ratio conservatively, i.e. biased
  // AWAY from the verdict they return: rmin_eff <= R_eff <= rmax_eff, so r/rmin_eff is
  // an upper bound on the true r/R_eff (still < 1 here) and r/rmax_eff a lower bound
  // (still >= 1 there). Neither can make a particle look more deeply excised than it is.
  if (r < rmin_eff) {crit = r/rmin_eff; return kAHInside;}
  if (r >= rmax_eff) {
    crit = (rmax_eff > 0.0) ? r/rmax_eff : AH_CRIT_FAR;
    return kAHOutside;
  }

  // In the shell: evaluate the actual angular surface. r > rmin_eff > 0 here, so
  // theta/phi are well defined (the r == 0 case can only reach the rmin branch above).
  const Real theta = Kokkos::acos(Kokkos::fmax(-1.0, Kokkos::fmin(1.0, dz/r)));
  const Real phi = Kokkos::atan2(dy, dx);
  const Real reff = shrink * AHSurfaceRadius(coef, h, lmax, lmpoints, theta, phi);
  crit = (reff > 0.0) ? r/reff : AH_CRIT_FAR;
  return (r < reff) ? kAHInside : kAHOutside;
}

#endif // Z4C_HORIZON_QUERY_HPP_
