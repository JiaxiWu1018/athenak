// ST equilibrium construction selectively reused from project/ST-cluster @ b51efb27.
// Original AthenaK BSD-3-Clause license applies. No historical sampler is imported.
#ifndef PGEN_EFFICIENCY_ST_PROFILE_HPP_
#define PGEN_EFFICIENCY_ST_PROFILE_HPP_
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "athena.hpp"
namespace efficiency_st {
struct ClusterState {
  Real y;
  Real v;
  Real v0;
  Real h;
};

struct ClusterRHS {
  Real dy;
  Real dv;
  Real dv0;
  Real dh;
};

struct ClusterProfile {
  std::vector<Real> x;
  std::vector<Real> y;
  std::vector<Real> v;
  std::vector<Real> v0;
  std::vector<Real> h;
  std::vector<Real> riso_over_m;
  std::vector<Real> conformal_a;
  std::vector<Real> lapse;
  std::vector<Real> cdf;
  Real xs;
  Real vs;
  Real v0s;
  Real hs;
  Real r_over_m;
  Real riso_over_m_surface;
  Real rest_mass_over_m;
};

ClusterState AddScaled(const ClusterState &s, const ClusterRHS &k, Real scale) {
  return {s.y + scale*k.dy, s.v + scale*k.dv, s.v0 + scale*k.dv0,
          s.h + scale*k.dh};
}

ClusterRHS EvaluateRHS(Real x, const ClusterState &s, Real yc, Real zeta) {
  Real y = std::min(std::max(s.y, yc), static_cast<Real>(1.0));
  Real inv_y_minus_one =
      std::max(static_cast<Real>(1.0)/y - static_cast<Real>(1.0),
               static_cast<Real>(0.0));
  Real inv_yc_minus_one = 1.0/yc - 1.0;
  Real ratio = inv_y_minus_one/inv_yc_minus_one;
  Real rho = std::pow(y/yc, -1.5)*std::sqrt(ratio);
  Real rho0 = std::sqrt(yc)*std::pow(y/yc, -1.0)*std::sqrt(ratio);
  Real pressure = std::pow(y/yc, -0.5)*std::pow(ratio, 1.5);
  Real compact = 1.0 - 2.0*zeta*s.v/x;
  Real radial_factor = 1.0/std::sqrt(compact);

  ClusterRHS rhs;
  rhs.dy = 2.0*zeta*y*(s.v + zeta*x*x*x*pressure)/(x*x*compact);
  rhs.dv = x*x*rho;
  rhs.dv0 = x*x*rho0*radial_factor;
  // h = ln(rbar/x) up to a constant. This removes the 1/x coordinate singularity
  // from Eq. (A21), leaving a regular h(0)=0 initial-value problem.
  rhs.dh = (radial_factor - 1.0)/x;
  return rhs;
}

ClusterState RK4Step(Real x, const ClusterState &s, Real dx, Real yc, Real zeta) {
  ClusterRHS k1 = EvaluateRHS(x, s, yc, zeta);
  ClusterState s2 = AddScaled(s, k1, 0.5*dx);
  ClusterRHS k2 = EvaluateRHS(x + 0.5*dx, s2, yc, zeta);
  ClusterState s3 = AddScaled(s, k2, 0.5*dx);
  ClusterRHS k3 = EvaluateRHS(x + 0.5*dx, s3, yc, zeta);
  ClusterState s4 = AddScaled(s, k3, dx);
  ClusterRHS k4 = EvaluateRHS(x + dx, s4, yc, zeta);

  return {
    s.y + dx*(k1.dy + 2.0*k2.dy + 2.0*k3.dy + k4.dy)/6.0,
    s.v + dx*(k1.dv + 2.0*k2.dv + 2.0*k3.dv + k4.dv)/6.0,
    s.v0 + dx*(k1.dv0 + 2.0*k2.dv0 + 2.0*k3.dv0 + k4.dv0)/6.0,
    s.h + dx*(k1.dh + 2.0*k2.dh + 2.0*k3.dh + k4.dh)/6.0
  };
}

ClusterProfile ConstructProfile(Real yc, Real dx) {
  constexpr int max_steps = 1000000;
  Real zeta = (1.0 - yc)/3.0;
  Real eps = std::min(static_cast<Real>(1.0e-6), static_cast<Real>(0.01)*dx);

  ClusterProfile profile;
  profile.x.reserve(50000);
  profile.y.reserve(50000);
  profile.v.reserve(50000);
  profile.v0.reserve(50000);
  profile.h.reserve(50000);

  profile.x.push_back(0.0);
  profile.y.push_back(yc);
  profile.v.push_back(0.0);
  profile.v0.push_back(0.0);
  profile.h.push_back(0.0);

  // Regular central series through the first nonzero radius.
  Real x = eps;
  ClusterState state {
    yc + zeta*yc*(1.0/3.0 + zeta)*x*x,
    x*x*x/3.0,
    x*x*x/3.0,
    zeta*x*x/6.0
  };
  profile.x.push_back(x);
  profile.y.push_back(state.y);
  profile.v.push_back(state.v);
  profile.v0.push_back(state.v0);
  profile.h.push_back(state.h);

  bool found_surface = false;
  for (int n = 0; n < max_steps; ++n) {
    Real x_old = x;
    ClusterState old = state;
    state = RK4Step(x_old, old, dx, yc, zeta);
    x = x_old + dx;
    if (!std::isfinite(state.y) || !std::isfinite(state.v) ||
        !std::isfinite(state.v0) || !std::isfinite(state.h)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Relativistic-cluster ODE integration became non-finite."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    if (state.y >= 1.0) {
      Real frac = (1.0 - old.y)/(state.y - old.y);
      x = x_old + frac*dx;
      state.y = 1.0;
      state.v = old.v + frac*(state.v - old.v);
      state.v0 = old.v0 + frac*(state.v0 - old.v0);
      state.h = old.h + frac*(state.h - old.h);
      found_surface = true;
    }

    profile.x.push_back(x);
    profile.y.push_back(state.y);
    profile.v.push_back(state.v);
    profile.v0.push_back(state.v0);
    profile.h.push_back(state.h);
    if (found_surface) { break; }
  }

  if (!found_surface) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Relativistic-cluster ODE did not reach y=1; decrease "
              << "cluster_profile_dx or check cluster_yc." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  profile.xs = profile.x.back();
  profile.vs = profile.v.back();
  profile.v0s = profile.v0.back();
  profile.hs = profile.h.back();
  profile.r_over_m = profile.xs/(profile.vs*zeta);
  if (profile.r_over_m <= 2.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Relativistic-cluster solution has R/M="
              << profile.r_over_m << " <= 2." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  profile.riso_over_m_surface =
      0.5*(profile.r_over_m - 1.0 +
           std::sqrt(profile.r_over_m*(profile.r_over_m - 2.0)));
  profile.rest_mass_over_m = profile.v0s/profile.vs;

  std::size_t nprof = profile.x.size();
  profile.riso_over_m.resize(nprof);
  profile.conformal_a.resize(nprof);
  profile.lapse.resize(nprof);
  profile.cdf.resize(nprof);
  Real alpha_surface = std::sqrt(1.0 - 2.0/profile.r_over_m);
  Real a_surface = profile.r_over_m/profile.riso_over_m_surface;
  for (std::size_t i = 0; i < nprof; ++i) {
    Real xfrac = (i == 0) ? 0.0 : profile.x[i]/profile.xs;
    profile.riso_over_m[i] =
        profile.riso_over_m_surface*xfrac*std::exp(profile.h[i] - profile.hs);
    profile.conformal_a[i] = a_surface*std::exp(profile.hs - profile.h[i]);
    profile.lapse[i] = alpha_surface*std::sqrt(profile.y[i]);
  }

  // Inverse-CDF representation of Eq. (A18). This samples the same distribution as
  // rejection sampling while avoiding a profile-dependent rejection bound.
  profile.cdf[0] = 0.0;
  Real integral = 0.0;
  for (std::size_t i = 1; i < nprof; ++i) {
    Real ym = profile.y[i-1];
    Real yp = profile.y[i];
    Real wm = profile.x[i-1]*profile.x[i-1]/ym*
              std::sqrt(std::max(static_cast<Real>(1.0)/ym - static_cast<Real>(1.0),
                                 static_cast<Real>(0.0)));
    Real wp = profile.x[i]*profile.x[i]/yp*
              std::sqrt(std::max(static_cast<Real>(1.0)/yp - static_cast<Real>(1.0),
                                 static_cast<Real>(0.0)));
    integral += 0.5*(wm + wp)*(profile.x[i] - profile.x[i-1]);
    profile.cdf[i] = integral;
  }
  if (!(integral > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Relativistic-cluster radial probability is zero."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (Real &value : profile.cdf) { value /= integral; }
  profile.cdf.back() = 1.0;

  return profile;
}

std::size_t CDFIndex(const std::vector<Real> &cdf, Real u) {
  auto it = std::lower_bound(cdf.begin() + 1, cdf.end(), u);
  return static_cast<std::size_t>(it - cdf.begin());
}

Real InterpolateCDF(const std::vector<Real> &cdf, const std::vector<Real> &values,
                    std::size_t hi, Real u) {
  std::size_t lo = hi - 1;
  Real denom = cdf[hi] - cdf[lo];
  Real frac = (denom > 0.0) ? (u - cdf[lo])/denom : 0.0;
  return values[lo] + frac*(values[hi] - values[lo]);
}


} // namespace efficiency_st
#endif
