#ifndef PGEN_PARTICLES_PLUMMER_PROFILE_HPP_
#define PGEN_PARTICLES_PLUMMER_PROFILE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file plummer_profile.hpp
//! \brief host-side 1D construction of the truncated relativistic Plummer Einstein
//! cluster used by nr_pic_plummer.cpp.
//!
//! The static-observer energy density eps = T_{mu nu} n^mu n^nu is prescribed to have
//! the Plummer form in AREAL radius r, hard-truncated at r = r_t with vacuum outside and
//! renormalised so that the ADM mass is exactly M (G = c = 1):
//!
//!   f_t   = r_t^3/(r_t^2+b^2)^{3/2},        M_P = M/f_t
//!   eps   = 3 M_P/(4 pi b^3) (1+r^2/b^2)^{-5/2}   (r <  r_t),   0   (r >  r_t)
//!   m(r)  = M_P r^3/(r^2+b^2)^{3/2}               (r <= r_t),   M   (r >= r_t)
//!   B     = (1-2m/r)^{-1/2},  v_c^2 = m/(r-2m),  W = sqrt((r-2m)/(r-3m))
//!   p_r   = 0,   p_t = eps v_c^2/2
//!   Phi'  = m/(r(r-2m)),   Phi = ln alpha
//!   j'    = (B-1)/r,       j   = ln(R/r),  R = isotropic radius
//!
//! matched to the exact Schwarzschild exterior at r_t:
//!   Phi(r_t) = 1/2 ln(1-2M/r_t),
//!   R_t      = (r_t - M + sqrt(r_t(r_t-2M)))/2,   j(r_t) = ln(R_t/r_t),
//! and integrated INWARD.  Both quadratures use composite Gauss-Legendre panels on an
//! exponentially stretched grid; the integrands are analytic on (0, r_t], so the table
//! reaches close to double-precision roundoff with no stepper error control.
//!
//! The rest-mass measure (needed by the particle sampler) is
//!   dM0/dr = 4 pi r^2 eps B/W,     F0(r) = M0(r)/M0(r_t),    mu = M0/N.
//!
//! Regular centre: m = O(r^3) so Phi', j' = O(r) and B-1 = O(r^2).  B-1 is evaluated
//! from a series in x = m/r below x = 1e-4 to avoid catastrophic cancellation.
//!
//! This header is deliberately free of Kokkos and AthenaK mesh headers so it can be
//! compiled and validated standalone against the independent Python implementation in
//! scripts/particles/plummer_id.py (see that file's --selftest).

#include <cmath>
#include <cstddef>
#include <vector>

namespace plummer {

#ifndef PLUMMER_REAL
using PReal = double;
#else
using PReal = PLUMMER_REAL;
#endif

//----------------------------------------------------------------------------------------
//! 16-point Gauss-Legendre nodes/weights on [-1,1] (symmetric halves listed once).
struct GL16 {
  static constexpr int N = 16;
  // abscissae (positive halves) and weights, standard values
  static const PReal *x() {
    static const PReal v[8] = {
      0.0950125098376374401853193, 0.2816035507792589132304605,
      0.4580167776572273863424194, 0.6178762444026437484466718,
      0.7554044083550030338951012, 0.8656312023878317438804679,
      0.9445750230732325760779884, 0.9894009349916499325961542};
    return v;
  }
  static const PReal *w() {
    static const PReal v[8] = {
      0.1894506104550684962853967, 0.1826034150449235888667637,
      0.1691565193950025381893121, 0.1495959888165767320815017,
      0.1246289712555338720524763, 0.0951585116824927848099251,
      0.0622535239386478928628438, 0.0271524594117540948517806};
    return v;
  }
};

//! \fn GaussLegendre16
//! \brief integral of f over [a,b] by one 16-point Gauss-Legendre panel.
template <class F>
inline PReal GaussLegendre16(const F &f, PReal a, PReal b) {
  const PReal half = 0.5*(b - a), mid = 0.5*(b + a);
  PReal s = 0.0;
  for (int q = 0; q < 8; ++q) {
    const PReal d = half*GL16::x()[q];
    s += GL16::w()[q]*(f(mid + d) + f(mid - d));
  }
  return half*s;
}

//----------------------------------------------------------------------------------------
//! \class PlummerProfile
//! \brief the truncated, ADM-normalised relativistic Plummer Einstein cluster.

class PlummerProfile {
 public:
  PReal M, b, rt;            // ADM mass, Plummer scale, areal cutoff
  PReal ft, MP;              // truncation factor and interior amplitude mass
  PReal Rt;                  // isotropic cutoff radius
  PReal M0;                  // total rest mass through r_t
  int npanel;                // number of quadrature panels
  PReal stretch;             // grid stretching exponent

  // node tables (npanel+1 entries); r_tab[0] = 0, r_tab[npanel] = rt
  std::vector<PReal> r_tab, Phi_tab, j_tab, M0_tab;

  PlummerProfile(PReal M_, PReal b_, PReal rt_, int npanel_ = 20000,
                 PReal stretch_ = 4.0)
      : M(M_), b(b_), rt(rt_), npanel(npanel_), stretch(stretch_) {
    ft = rt*rt*rt/std::pow(rt*rt + b*b, 1.5);
    MP = M/ft;
    Rt = 0.5*(rt - M + std::sqrt(rt*(rt - 2.0*M)));
    Build();
  }

  // ------------------------------------------------------------------ local profile
  //! energy density measured by the static observer (0 outside the cutoff)
  PReal eps(PReal r) const {
    if (r >= rt) { return 0.0; }
    const PReal y = r/b;
    return 3.0*MP/(4.0*M_PI*b*b*b)*std::pow(1.0 + y*y, -2.5);
  }
  //! gravitational mass inside areal radius r
  PReal m(PReal r) const {
    if (r >= rt) { return M; }
    return MP*r*r*r/std::pow(r*r + b*b, 1.5);
  }
  //! dm/dr = 4 pi r^2 eps
  PReal dmdr(PReal r) const {
    if (r >= rt) { return 0.0; }
    return 3.0*MP*b*b*r*r/std::pow(r*r + b*b, 2.5);
  }
  //! B - 1 = (1-2m/r)^{-1/2} - 1, stable as m/r -> 0
  PReal Bm1(PReal r) const {
    const PReal x = (r > 0.0) ? m(r)/r : 0.0;
    if (std::fabs(x) < 1.0e-4) {
      return x*(1.0 + 1.5*x*(1.0 + (5.0/3.0)*x*(1.0 + 1.75*x)));
    }
    return 1.0/std::sqrt(1.0 - 2.0*x) - 1.0;
  }
  PReal B(PReal r) const { return 1.0 + Bm1(r); }
  //! circular-orbit speed measured by the static observer, squared
  PReal vc2(PReal r) const {
    const PReal mm = m(r);
    return mm/(r - 2.0*mm);
  }
  //! Lorentz factor of the circular orbit relative to the static observer
  PReal W(PReal r) const {
    const PReal mm = m(r);
    return std::sqrt((r - 2.0*mm)/(r - 3.0*mm));
  }
  PReal pt(PReal r) const { return 0.5*eps(r)*vc2(r); }
  PReal dPhidr(PReal r) const {
    if (r <= 0.0) { return 0.0; }
    const PReal mm = m(r);
    return mm/(r*(r - 2.0*mm));
  }
  PReal djdr(PReal r) const { return (r > 0.0) ? Bm1(r)/r : 0.0; }
  PReal dM0dr(PReal r) const {
    return 4.0*M_PI*r*r*eps(r)*B(r)/W(r);
  }

  // ---------------------------------------------------------- exact node evaluation
  //! index of the smallest node >= r
  std::size_t NodeAbove(PReal r) const {
    if (r <= r_tab.front()) { return 0; }
    if (r >= r_tab.back()) { return r_tab.size() - 1; }
    // the grid is r_k = rt*(exp(s*k/npanel)-1)/(exp(s)-1); invert analytically
    const PReal s = stretch;
    const PReal frac = std::log1p(r/rt*std::expm1(s))/s;
    std::size_t k = static_cast<std::size_t>(std::ceil(frac*npanel));
    if (k >= r_tab.size()) { k = r_tab.size() - 1; }
    while (k > 0 && r_tab[k-1] >= r) { --k; }
    while (k + 1 < r_tab.size() && r_tab[k] < r) { ++k; }
    return k;
  }
  //! Phi(r), exact: tabulated node value minus the GL integral from r up to that node
  PReal Phi(PReal r) const {
    if (r >= rt) { return 0.5*std::log(1.0 - 2.0*M/r); }
    const std::size_t k = NodeAbove(r);
    return Phi_tab[k] - GaussLegendre16([&](PReal s){ return dPhidr(s); }, r, r_tab[k]);
  }
  //! j(r) = ln(R/r), exact
  PReal j(PReal r) const {
    if (r >= rt) {
      const PReal Rr = 0.5*(r - M + std::sqrt(r*(r - 2.0*M)));
      return std::log(Rr/r);
    }
    const std::size_t k = NodeAbove(r);
    return j_tab[k] - GaussLegendre16([&](PReal s){ return djdr(s); }, r, r_tab[k]);
  }
  //! F0(r) = M0(r)/M0, exact
  PReal F0(PReal r) const {
    if (r >= rt) { return 1.0; }
    const std::size_t k = NodeAbove(r);
    return (M0_tab[k] - GaussLegendre16([&](PReal s){ return dM0dr(s); }, r, r_tab[k]))/M0;
  }

  PReal alpha(PReal r) const { return std::exp(Phi(r)); }
  PReal Riso(PReal r) const { return (r >= rt)
      ? 0.5*(r - M + std::sqrt(r*(r - 2.0*M))) : r*std::exp(j(r)); }
  PReal psi(PReal r) const { return std::exp(-0.5*j(r)); }

  // ------------------------------------------------------------------- inversions
  //! r = F0^{-1}(q), bracketed on the table then Newton-polished to roundoff
  PReal InvertF0(PReal q) const {
    if (q <= 0.0) { return 0.0; }
    if (q >= 1.0) { return rt; }
    // bracket on the tabulated CDF
    std::size_t lo = 0, hi = r_tab.size() - 1;
    const PReal target = q*M0;
    while (hi - lo > 1) {
      const std::size_t mid = (lo + hi)/2;
      if (M0_tab[mid] <= target) { lo = mid; } else { hi = mid; }
    }
    const PReal dm = M0_tab[hi] - M0_tab[lo];
    PReal r = (dm > 0.0) ? r_tab[lo] + (target - M0_tab[lo])/dm*(r_tab[hi] - r_tab[lo])
                         : 0.5*(r_tab[lo] + r_tab[hi]);
    for (int it = 0; it < 6; ++it) {
      const PReal f = F0(r) - q;
      const PReal fp = dM0dr(r)/M0;
      if (!(fp > 0.0)) { break; }
      PReal rn = r - f/fp;
      if (!(rn > 0.0)) { rn = 0.5*r; }
      if (rn > rt) { rn = rt; }
      if (std::fabs(rn - r) <= 1.0e-15*std::fabs(rn)) { r = rn; break; }
      r = rn;
    }
    return r;
  }
  //! r = R^{-1}(R), the inverse of the monotone isotropic map
  PReal AreaFromIso(PReal R) const {
    if (R >= Rt) { const PReal q = 0.5*M/R; return R*(1.0 + q)*(1.0 + q); }
    if (R <= 0.0) { return 0.0; }
    // bracket: R(r) is monotone, R_tab[k] = r_tab[k]*exp(j_tab[k])
    std::size_t lo = 0, hi = r_tab.size() - 1;
    while (hi - lo > 1) {
      const std::size_t mid = (lo + hi)/2;
      if (r_tab[mid]*std::exp(j_tab[mid]) <= R) { lo = mid; } else { hi = mid; }
    }
    PReal a = r_tab[lo], c = r_tab[hi];
    // secant seed then safeguarded Newton: dR/dr = B exp(j)
    PReal r = 0.5*(a + c);
    for (int it = 0; it < 60; ++it) {
      const PReal jj = j(r);
      const PReal f = r*std::exp(jj) - R;
      if (f > 0.0) { c = r; } else { a = r; }
      const PReal fp = B(r)*std::exp(jj);
      PReal rn = (fp > 0.0) ? r - f/fp : 0.5*(a + c);
      if (!(rn > a) || !(rn < c)) { rn = 0.5*(a + c); }
      if (std::fabs(rn - r) <= 1.0e-15*std::fabs(rn)) { return rn; }
      r = rn;
    }
    return r;
  }

  // -------------------------------------------------------------------- key radii
  //! gravitational half-mass areal radius: m(r) = M/2 (analytic for the Plummer law)
  PReal HalfMassRadius() const {
    const PReal k = std::pow(0.5*M/MP, 2.0/3.0);
    return b*std::sqrt(k/(1.0 - k));
  }
  //! reference clock: circular coordinate period at the half-mass radius
  PReal HalfMassPeriod() const {
    const PReal rh = HalfMassRadius();
    return 2.0*M_PI*rh/(alpha(rh)*std::sqrt(vc2(rh)));
  }
  //! r^2 m' + r m - 6 m^2 : positive => individually radially stable circular orbit
  PReal RadialStability(PReal r) const {
    const PReal mm = m(r);
    return r*r*dmdr(r) + r*mm - 6.0*mm*mm;
  }

 private:
  void Build() {
    r_tab.assign(npanel + 1, 0.0);
    Phi_tab.assign(npanel + 1, 0.0);
    j_tab.assign(npanel + 1, 0.0);
    M0_tab.assign(npanel + 1, 0.0);
    const PReal denom = std::expm1(stretch);
    for (int k = 0; k <= npanel; ++k) {
      const PReal s = static_cast<PReal>(k)/static_cast<PReal>(npanel);
      r_tab[k] = rt*std::expm1(stretch*s)/denom;
    }
    r_tab[0] = 0.0;
    r_tab[npanel] = rt;

    // panel integrals
    std::vector<PReal> IP(npanel), IJ(npanel), IM(npanel);
    for (int k = 0; k < npanel; ++k) {
      const PReal a = r_tab[k], c = r_tab[k+1];
      IP[k] = GaussLegendre16([&](PReal s){ return dPhidr(s); }, a, c);
      IJ[k] = GaussLegendre16([&](PReal s){ return djdr(s); }, a, c);
      IM[k] = GaussLegendre16([&](PReal s){ return dM0dr(s); }, a, c);
    }
    // Phi and j integrate INWARD from the exact Schwarzschild match at r_t
    Phi_tab[npanel] = 0.5*std::log(1.0 - 2.0*M/rt);
    j_tab[npanel] = std::log(Rt/rt);
    for (int k = npanel - 1; k >= 0; --k) {
      Phi_tab[k] = Phi_tab[k+1] - IP[k];
      j_tab[k] = j_tab[k+1] - IJ[k];
    }
    // rest mass integrates OUTWARD from the regular centre
    M0_tab[0] = 0.0;
    for (int k = 0; k < npanel; ++k) { M0_tab[k+1] = M0_tab[k] + IM[k]; }
    M0 = M0_tab[npanel];
  }
};

}  // namespace plummer
#endif  // PGEN_PARTICLES_PLUMMER_PROFILE_HPP_
