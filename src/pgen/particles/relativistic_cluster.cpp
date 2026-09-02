//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file relativistic_cluster.cpp
//! \brief Relativistic monoenergetic collisionless cluster from Appendix A of
//! Bamber et al., "Evolution of a black hole cluster in full general relativity",
//! arXiv:2505.01495v2.
//!
//! This pgen constructs the smooth spherical equilibrium described by Eqs. (A11)-(A15),
//! maps it to isotropic coordinates using Eqs. (A21)-(A22), initializes the corresponding
//! time-symmetric ADM metric, and samples a finite-N equal-rest-mass particle realization
//! according to Eqs. (A18) and (A23). In this AthenaK tree the finite-N members are
//! self-gravitating particles, not Bowen-York punctures; a separate N-puncture constraint
//! solver would be required to reproduce the black-hole initial-data step of the paper.
//!
//! Required runtime configuration:
//!   - 3D mesh
//!   - <z4c> block
//!   - <particles> init=pgen, particle_type=dust, pusher=gr_boris, feedback=true
//!
//! <problem> parameters:
//!   cluster_n          number of equal-mass particles (default 25)
//!   cluster_yc         central redshift parameter y_c (default 0.819)
//!   cluster_mass       total gravitational mass M (default 1)
//!   cluster_seed       random seed (default 1)
//!   cluster_profile_dx dimensionless ODE step in x (default 1e-4)
//!   cluster_center_x1, cluster_center_x2, cluster_center_x3 (default 0)
//!
//! ST-migration perturbations. lambda_tan and eps_out leave the equilibrium metric, sampled
//! positions, per-particle rest mass and per-particle momentum magnitude |u_i| untouched.
//! The coherent radial mode instead adds a local Eulerian orthonormal radial 3-velocity and
//! therefore deliberately changes the particle energy. All three blocks are skipped at zero,
//! reproducing the previous pgen path bit for bit. See the sections below.
//!   cluster_lambda_tan tangential reorientation amplitude in [0,1] (default 0)
//!   cluster_eps_out    outward-bias amplitude eps_out in [0,1] (default 0)
//!   cluster_radial_mode_u coherent radial-mode surface 3-velocity U in (-1,1) (default 0)
//!   cluster_scpd_eps   sequence-tangent canonical phase-space displacement amplitude
//!                      (default 0; eps>0 moves toward the stable branch, eps<0 toward
//!                      prompt collapse, eps=1 puts every enclosed-mass radius on the
//!                      target model's)
//!   cluster_scpd_target_yc  target central parameter of the displacement map (default 0,
//!                      meaning the automatically located stable-branch model with the
//!                      same binding fraction, hence the same M and M0)
//!   cluster_solve_momentum_constraint initialize the spherical CTT K_ij correction
//!                                     sourced by the coherent mode (default false)
//!   cluster_solve_hamiltonian_constraint additionally solve the coupled incremental
//!                                      Hamiltonian correction (default false; requires
//!                                      cluster_solve_momentum_constraint=true)
//!   cluster_bias_mode  "stratified" (default) or "bernoulli"
//!   cluster_bias_seed  salt for the bernoulli-mode per-tag hash (default 0)
//!   cluster_t0_dump    if non-empty, rank 0 writes the full global t=0 particle sample
//!                      to this file in double precision (default "", i.e. off)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "particles/particles.hpp"
#include "pgen/pgen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

void STRefinementCondition(MeshBlockPack *pmbp);

namespace {

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
  Real cdf_normalization;
};

struct PrtclStage {
  std::vector<Real> x, y, z, ux, uy, uz, mass;
  std::vector<int> gid, tag;

  void Add(Real x_, Real y_, Real z_, Real ux_, Real uy_, Real uz_, Real mass_,
           int gid_, int tag_) {
    x.push_back(x_);
    y.push_back(y_);
    z.push_back(z_);
    ux.push_back(ux_);
    uy.push_back(uy_);
    uz.push_back(uz_);
    mass.push_back(mass_);
    gid.push_back(gid_);
    tag.push_back(tag_);
  }
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
  profile.cdf_normalization = integral;
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

Real InterpolateRadius(const std::vector<Real> &radius,
                       const std::vector<Real> &values, Real r) {
  if (r <= radius.front()) { return values.front(); }
  if (r >= radius.back()) { return values.back(); }
  auto it = std::lower_bound(radius.begin() + 1, radius.end(), r);
  std::size_t hi = static_cast<std::size_t>(it - radius.begin());
  std::size_t lo = hi - 1;
  Real denom = radius[hi] - radius[lo];
  Real frac = (denom > 0.0) ? (r - radius[lo])/denom : 0.0;
  return values[lo] + frac*(values[hi] - values[lo]);
}

//----------------------------------------------------------------------------------------
// Velocity-angle perturbations
// ----------------------------
// The equilibrium sample of Eq. (A23) draws the momentum DIRECTION isotropically and
// fixes its magnitude from the local y:  |p_hat|/m0 = sqrt(1/y - 1).  Writing the unit
// radial direction at the particle as n_hat and mu = (u_i n^i)/|u|, an isotropic sample
// has mu uniform on [-1,1], hence <mu> = 0 (no net radial current) and <mu^2> = 1/3.
//
// Tangential reorientation maps
//
//     mu -> (1 - lambda_tan) mu
//
// for every particle and rescales the existing tangential component to preserve |u_i|.
// It therefore preserves -u_t particle by particle, converts radial motion into
// tangential motion, raises |L|, and does not create a coherent radial flux. The
// lambda_tan = 0 block is skipped exactly.
//
// The independent outward radial-sign bias reflects the radial component for a controlled
// subset of the inward movers,
//
//     u_i  ->  u_i - 2 (u_j n^j) n_i    for the selected particles with u_j n^j < 0,
//
// which maps mu -> |mu| and leaves |u_i| (hence -u_t, hence the monoenergetic particle
// energy) unchanged particle by particle.  If a fraction eps_out of the inward movers is
// selected then, in the continuum limit,
//
//     P(sign = +1) = (1 + eps_out)/2,   <mu> = eps_out/2,   <mu^2> = 1/3 (unchanged),
//     <v_hat_r>(r) = (eps_out/2) sqrt(1 - y(r)).
//
// So every EVEN velocity moment - and therefore the energy density and the pressure the
// Hamiltonian constraint sees - is preserved, while the odd radial moment is dialled
// linearly in eps_out.  The momentum constraint is NOT preserved; that is the deliberate,
// measured cost of this experiment.
//
// Two selection rules are provided, both a deterministic function of the globally unique
// particle tag alone, so the realization is independent of the MPI decomposition, the
// rank count and the GPU execution order:
//
//   "stratified" (default) -- the inward movers are ordered by (isotropic radius, tag)
//       and the particle at rank j in that order is selected iff the base-2 radical
//       inverse (van der Corput) of j is < eps_out. Because the radical-inverse sequence
//       is low-discrepancy, the selected fraction is eps_out + O(log K / K) GLOBALLY and,
//       more usefully, in every contiguous window of radii: the induced radial current
//       profile carries essentially no extra shot noise. For eps_out = 1/2 the rule
//       reduces to "flip every other inward mover in radius order", for eps_out = 1/4
//       to "every fourth", and so on. The selected sets are nested in eps_out, so the
//       runs of an eps_out sweep differ only by the particles that the larger eps_out
//       adds.
//
//   "bernoulli" -- a tag is selected iff a SplitMix64 hash of the tag is < eps_out.
//       Independent per particle, giving the binomial shot noise the stratified rule
//       removes; retained as a cross-check that the stratification introduces no bias.
//       These sets are nested in eps_out as well.
//
// The independent coherent radial mode is an ENERGY-CHANGING perturbation inspired by
// neutron-star migration tests. In the local Eulerian orthonormal frame it applies
//
//     delta v^(hat r) = U (3 x - x^3)/2,
//     x = r_areal/R_areal = A(rbar) rbar/R_areal,
//
// to every particle. The tangential Eulerian 3-velocity is retained. With
// u_(hat i)=u_i/A=W v_(hat i), the updated stored covariant spatial 4-velocity is
//
//     W' = 1/sqrt(1-|v'|^2),       u_i' = A W' v'_(hat i).
//
// Thus the formula is never added to a coordinate component. For U>0 the profile is
// outward everywhere except at the regular center, reaches U at the surface, and has
// zero radial derivative there. The block is skipped exactly when U=0.
//----------------------------------------------------------------------------------------

//! \brief base-2 radical inverse (van der Corput) of j, in [0,1).
Real RadicalInverse2(std::uint64_t j) {
  j = (j << 32) | (j >> 32);
  j = ((j & 0x0000ffff0000ffffULL) << 16) | ((j & 0xffff0000ffff0000ULL) >> 16);
  j = ((j & 0x00ff00ff00ff00ffULL) << 8) | ((j & 0xff00ff00ff00ff00ULL) >> 8);
  j = ((j & 0x0f0f0f0f0f0f0f0fULL) << 4) | ((j & 0xf0f0f0f0f0f0f0f0ULL) >> 4);
  j = ((j & 0x3333333333333333ULL) << 2) | ((j & 0xccccccccccccccccULL) >> 2);
  j = ((j & 0x5555555555555555ULL) << 1) | ((j & 0xaaaaaaaaaaaaaaaaULL) >> 1);
  return static_cast<Real>(j >> 11)*(1.0/9007199254740992.0);
}

//! \brief SplitMix64 finalizer mapped to [0,1); a deterministic per-tag uniform draw.
Real SplitMixUniform(std::uint64_t key) {
  std::uint64_t z = key + 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30))*0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27))*0x94d049bb133111ebULL;
  z = z ^ (z >> 31);
  return static_cast<Real>(z >> 11)*(1.0/9007199254740992.0);
}

//! \struct DrawnParticle
//! \brief one realization of the equilibrium sample for a single tag.

struct DrawnParticle {
  Real px, py, pz;     // position, including the cluster-center offset
  Real ux, uy, uz;     // covariant spatial 4-velocity u_i per unit rest mass
  Real riso;           // isotropic radius measured from the cluster center
  Real ylocal;         // equilibrium y at the particle
  Real conf_a;         // conformal factor A at the particle (gamma_ij = A^2 delta_ij)
  Real umag;           // |u_i| = A |p_hat|/m0
  Real nx, ny, nz;     // unit radial direction at the particle
  Real cdf_u;          // the uniform deviate that drew the radius = enclosed mass fraction
  std::size_t cdf_index;  // its bracketing index in profile.cdf
};

//! \fn DrawParticle
//! \brief consume the five uniform deviates of one particle and build its sample.
//! The position and momentum expressions are written exactly as in the unperturbed pgen
//! so that eps_out = 0 reproduces the historical realization bit for bit.

DrawnParticle DrawParticle(std::mt19937_64 *generator,
                           std::uniform_real_distribution<Real> *uniform,
                           const ClusterProfile &profile, Real total_mass,
                           const Real center[3]) {
  DrawnParticle d;
  Real ur = (*uniform)(*generator);
  std::size_t hi = CDFIndex(profile.cdf, ur);
  Real riso = total_mass*InterpolateCDF(profile.cdf, profile.riso_over_m, hi, ur);
  Real y = InterpolateCDF(profile.cdf, profile.y, hi, ur);
  Real conformal_a = InterpolateCDF(profile.cdf, profile.conformal_a, hi, ur);

  Real cos_theta = 2.0*(*uniform)(*generator) - 1.0;
  Real sin_theta = std::sqrt(std::max(static_cast<Real>(1.0) - cos_theta*cos_theta,
                                      static_cast<Real>(0.0)));
  Real phi = 2.0*M_PI*(*uniform)(*generator);
  d.px = center[0] + riso*sin_theta*std::cos(phi);
  d.py = center[1] + riso*sin_theta*std::sin(phi);
  d.pz = center[2] + riso*cos_theta;

  Real cos_vtheta = 2.0*(*uniform)(*generator) - 1.0;
  Real sin_vtheta = std::sqrt(std::max(static_cast<Real>(1.0) -
                                       cos_vtheta*cos_vtheta,
                                       static_cast<Real>(0.0)));
  Real vphi = 2.0*M_PI*(*uniform)(*generator);
  Real phat_over_m =
      std::sqrt(std::max(static_cast<Real>(1.0)/y - static_cast<Real>(1.0),
                         static_cast<Real>(0.0)));
  // Appendix A gives p^i = p^(hat i)/A. AthenaK stores covariant u_i, so for
  // gamma_ij=A^2 delta_ij: u_i = gamma_ij p^j/m0 = A p^(hat i)/m0.
  Real umag = conformal_a*phat_over_m;
  d.ux = umag*sin_vtheta*std::cos(vphi);
  d.uy = umag*sin_vtheta*std::sin(vphi);
  d.uz = umag*cos_vtheta;

  d.riso = riso;
  d.ylocal = y;
  d.conf_a = conformal_a;
  d.umag = umag;
  d.cdf_u = ur;
  d.cdf_index = hi;
  d.nx = sin_theta*std::cos(phi);
  d.ny = sin_theta*std::sin(phi);
  d.nz = cos_theta;
  return d;
}

//! \brief signed radial component u_i n^i of the sampled momentum.
Real RadialMomentum(const DrawnParticle &d) {
  return d.ux*d.nx + d.uy*d.ny + d.uz*d.nz;
}

//! \brief reflect an inward radial momentum outward; a no-op if already outward.
//! |u_i| is preserved exactly in exact arithmetic (|n_hat| = 1) and to a few ulp in
//! floating point, so -u_t is unchanged.  Returns true if the reflection was applied.
bool ReflectRadialOutward(DrawnParticle *d) {
  Real udotn = RadialMomentum(*d);
  if (udotn >= 0.0) { return false; }
  d->ux -= 2.0*udotn*d->nx;
  d->uy -= 2.0*udotn*d->ny;
  d->uz -= 2.0*udotn*d->nz;
  return true;
}

//! \brief map mu -> (1-lambda_tan) mu at fixed |u_i|.
//! The existing tangential azimuth is retained. The measure-zero exactly radial case uses
//! a deterministic orthogonal direction, avoiding any new random draw or tag dependence.
void ReorientTangential(DrawnParticle *d, Real lambda_tan) {
  Real umag = std::sqrt(d->ux*d->ux + d->uy*d->uy + d->uz*d->uz);
  if (umag == 0.0) { return; }

  // Normalize the analytically unit radial vector once so the decomposition is an
  // orthogonal one even at floating-point precision.
  Real nmag = std::sqrt(d->nx*d->nx + d->ny*d->ny + d->nz*d->nz);
  Real nx = d->nx/nmag, ny = d->ny/nmag, nz = d->nz/nmag;
  Real ur = d->ux*nx + d->uy*ny + d->uz*nz;
  Real tx = d->ux - ur*nx;
  Real ty = d->uy - ur*ny;
  Real tz = d->uz - ur*nz;
  Real tmag = std::sqrt(std::max(tx*tx + ty*ty + tz*tz,
                                 static_cast<Real>(0.0)));

  Real ur_new = (1.0 - lambda_tan)*ur;
  Real tmag_new = std::sqrt(std::max(umag*umag - ur_new*ur_new,
                                     static_cast<Real>(0.0)));
  if (tmag > 0.0) {
    Real scale = tmag_new/tmag;
    tx *= scale;
    ty *= scale;
    tz *= scale;
  } else {
    // Pick the coordinate axis least aligned with n and project it into the tangent
    // plane. This branch is only a robustness guard; an isotropic random draw reaches it
    // with probability zero.
    if (std::abs(nx) <= std::abs(ny) && std::abs(nx) <= std::abs(nz)) {
      tx = 0.0;
      ty = -nz;
      tz = ny;
    } else if (std::abs(ny) <= std::abs(nz)) {
      tx = nz;
      ty = 0.0;
      tz = -nx;
    } else {
      tx = -ny;
      ty = nx;
      tz = 0.0;
    }
    Real inv_tmag = 1.0/std::sqrt(tx*tx + ty*ty + tz*tz);
    tx *= tmag_new*inv_tmag;
    ty *= tmag_new*inv_tmag;
    tz *= tmag_new*inv_tmag;
  }

  d->ux = ur_new*nx + tx;
  d->uy = ur_new*ny + ty;
  d->uz = ur_new*nz + tz;

  // Remove the last rounding error in the constructed magnitude. This scalar rescaling
  // leaves the intended direction unchanged and makes the energy-neutral invariant as
  // sharp as floating point permits.
  Real umag_after = std::sqrt(d->ux*d->ux + d->uy*d->uy + d->uz*d->uz);
  if (umag_after > 0.0) {
    Real scale = umag/umag_after;
    d->ux *= scale;
    d->uy *= scale;
    d->uz *= scale;
  }
}

//! \brief add the coherent Eulerian-orthonormal radial 3-velocity mode.
//!
//! Returns false if the requested velocity would be non-finite or non-timelike. Diagnostic
//! outputs describe the local frame immediately before and after this map.
bool AddCoherentRadialMode(DrawnParticle *d, Real radial_mode_u,
                           Real surface_areal_radius, Real mode_conformal_a,
                           Real *delta_vr,
                           Real *vr_before, Real *vr_after, Real *w_before,
                           Real *w_after, Real *v2_after,
                           Real *abs_delta_vtan, Real *normalization_residual) {
  Real nmag = std::sqrt(d->nx*d->nx + d->ny*d->ny + d->nz*d->nz);
  if (!(nmag > 0.0) || !(d->conf_a > 0.0) || !(surface_areal_radius > 0.0)) {
    return false;
  }
  Real nx = d->nx/nmag, ny = d->ny/nmag, nz = d->nz/nmag;

  // u_(hat i)=u_i/A is the Eulerian-frame spatial 4-momentum per unit rest mass.
  Real uhx = d->ux/d->conf_a;
  Real uhy = d->uy/d->conf_a;
  Real uhz = d->uz/d->conf_a;
  Real usq = uhx*uhx + uhy*uhy + uhz*uhz;
  Real w0 = std::sqrt(1.0 + usq);
  Real vx = uhx/w0, vy = uhy/w0, vz = uhz/w0;
  Real vr0 = vx*nx + vy*ny + vz*nz;
  Real tx = vx - vr0*nx;
  Real ty = vy - vr0*ny;
  Real tz = vz - vr0*nz;

  // Circumferential radius is r=A*rbar for gamma_ij=A^2 delta_ij. Clamp only protects
  // the polynomial from interpolation roundoff at the sampled surface.
  Real x = mode_conformal_a*d->riso/surface_areal_radius;
  x = std::min(std::max(x, static_cast<Real>(0.0)), static_cast<Real>(1.0));
  Real dv = 0.5*radial_mode_u*(3.0*x - x*x*x);
  Real vr1 = vr0 + dv;
  Real vx1 = tx + vr1*nx;
  Real vy1 = ty + vr1*ny;
  Real vz1 = tz + vr1*nz;
  Real v2 = vx1*vx1 + vy1*vy1 + vz1*vz1;
  if (!std::isfinite(v2) || !(v2 < 1.0)) { return false; }

  Real w1 = 1.0/std::sqrt(1.0 - v2);
  d->ux = d->conf_a*w1*vx1;
  d->uy = d->conf_a*w1*vy1;
  d->uz = d->conf_a*w1*vz1;
  d->umag = std::sqrt(d->ux*d->ux + d->uy*d->uy + d->uz*d->uz);

  // Reconstruct the Eulerian velocity from the stored u_i as an independent local check.
  Real uhx1 = d->ux/d->conf_a;
  Real uhy1 = d->uy/d->conf_a;
  Real uhz1 = d->uz/d->conf_a;
  Real w1_check = std::sqrt(1.0 + uhx1*uhx1 + uhy1*uhy1 + uhz1*uhz1);
  Real cvx = uhx1/w1_check, cvy = uhy1/w1_check, cvz = uhz1/w1_check;
  Real cvr = cvx*nx + cvy*ny + cvz*nz;
  Real ctx = cvx - cvr*nx;
  Real cty = cvy - cvr*ny;
  Real ctz = cvz - cvr*nz;

  *delta_vr = dv;
  *vr_before = vr0;
  *vr_after = cvr;
  *w_before = w0;
  *w_after = w1_check;
  *v2_after = v2;
  *abs_delta_vtan = std::sqrt((ctx-tx)*(ctx-tx) + (cty-ty)*(cty-ty) +
                              (ctz-tz)*(ctz-tz));
  *normalization_residual = std::abs(w1_check - w1)/w1;
  return std::isfinite(*abs_delta_vtan) && std::isfinite(*normalization_residual);
}

//----------------------------------------------------------------------------------------
// Sequence-tangent canonical phase-space displacement (SCPD)
// ---------------------------------------------------------
// Sessions 01, 02 and 04 perturbed the sampled MOMENTA of the monoenergetic ST cluster
// while leaving the sampled POSITIONS, and therefore the t=0 rest-mass profile, exactly at
// their unstable-equilibrium values.  For collisionless (Vlasov) matter that is the
// subdominant half of a radial mode: a spherically symmetric linear perturbation is
// generated by a phase-space (canonical) displacement, and near a turning point of the
// equilibrium sequence the marginal eigenmode is a PURE displacement with no velocity
// perturbation at all.  The velocity part carries the extra factor sigma/Omega_r, which is
// measured to be 0.35 for yc=0.605.
//
// SCPD applies the exact canonical lift of a radial point transformation of the isotropic
// radius,
//
//     rbar -> R(rbar),   u_rbar -> u_rbar/R'(rbar),   u_theta, u_phi unchanged,
//
// which in Cartesian components with n^i = x^i/rbar reads
//
//     x'^i    = (R/rbar) x^i,
//     u'_i    = (u_j n^j)/R'(rbar) n_i + (rbar/R(rbar)) (u_i - (u_j n^j) n_i).
//
// Properties, all verified in the t=0 gate:
//   * every particle's angular momentum L_i = eps_ijk x^j u_k is preserved EXACTLY (this
//     is forced for any l=0 perturbation: a spherically symmetric generator cannot depend
//     on the angle conjugate to L);
//   * the map is canonical in each (rbar,u_rbar), (theta,u_theta), (phi,u_phi) pair, hence
//     exactly Liouville/measure preserving, hence the enclosed rest-mass fraction of every
//     particle is preserved;
//   * it is ODD in u_rbar, so <u_hat_r> stays zero and S^i vanishes identically: unlike the
//     Session-04 mode it introduces NO momentum-constraint violation and keeps K_ij = 0;
//   * two particles at the same radius moving inward and outward receive OPPOSITE
//     delta u_r, in contrast to the Session-04 velocity kick which gives them the same one;
//   * the delta(E - alpha_s) energy shell is translated rather than broadened.
//
// The radial map is taken along the equilibrium sequence at fixed enclosed rest mass:
//
//     R_eps(rbar) = (1 - eps) rbar + eps R_*(rbar),
//     R_*(rbar)   = the radius at which the TARGET model contains the same sampler CDF
//                   fraction q as the START model contains inside rbar.
//
// The default target is the stable-branch model with the same binding fraction (hence the
// same M and M0) as the start; it may be overridden with cluster_scpd_target_yc.  eps -> 0
// is the sequence-tangent (marginal-mode) direction, eps = 1 puts every enclosed-mass
// radius exactly on the target model's, and eps < 0 drives the cluster toward prompt
// collapse.  Because the map moves the matter it changes the rest-mass profile and hence
// the Hamiltonian constraint; the metric is obtained from an incremental spherical solve
// which is an exact no-op at eps = 0.

//! \brief binding fraction B = (M0-M)/M0 of the ST model with central parameter yc.
Real ClusterBindingFraction(Real yc, Real dx) {
  ClusterProfile p = ConstructProfile(yc, dx);
  return 1.0 - p.vs/p.v0s;
}

//! \brief central parameter at the maximum of the binding fraction (golden section).
Real ClusterTurningPointYc(Real dx) {
  const Real inv_phi = 0.6180339887498949;
  Real a = 0.40, b = 0.95;
  Real c = b - inv_phi*(b - a), d = a + inv_phi*(b - a);
  Real fc = -ClusterBindingFraction(c, dx), fd = -ClusterBindingFraction(d, dx);
  for (int it = 0; it < 100 && (b - a) > 1.0e-12; ++it) {
    if (fc < fd) {
      b = d; d = c; fd = fc; c = b - inv_phi*(b - a); fc = -ClusterBindingFraction(c, dx);
    } else {
      a = c; c = d; fc = fd; d = a + inv_phi*(b - a); fd = -ClusterBindingFraction(d, dx);
    }
  }
  return 0.5*(a + b);
}

//! \brief the STABLE-branch model with the same binding fraction as yc (bisection).
Real ClusterEqualBindingPartnerYc(Real yc, Real dx) {
  Real target = ClusterBindingFraction(yc, dx);
  Real tp = ClusterTurningPointYc(dx);
  Real lo = tp + 1.0e-9, hi = 0.999;
  Real flo = ClusterBindingFraction(lo, dx) - target;
  Real fhi = ClusterBindingFraction(hi, dx) - target;
  if (flo*fhi > 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "No stable-branch model with the binding fraction of yc="
              << yc << "; supply cluster_scpd_target_yc explicitly." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (int it = 0; it < 200 && (hi - lo) > 1.0e-13; ++it) {
    Real mid = 0.5*(lo + hi);
    Real fm = ClusterBindingFraction(mid, dx) - target;
    if (flo*fm <= 0.0) { hi = mid; fhi = fm; } else { lo = mid; flo = fm; }
  }
  return 0.5*(lo + hi);
}

//! \struct DisplacementMap
//! \brief the enclosed-rest-mass radial map R_*(rbar) and its derivative, tabulated on the
//! START profile's own nodes and indexed by the sampler CDF value q of that node.
struct DisplacementMap {
  std::vector<Real> r_map;    // R_*/M at each start node
  std::vector<Real> dr_map;   // dR_*/drbar at each start node
  Real target_yc = 0.0;
  Real target_binding = 0.0;
  Real start_binding = 0.0;
  Real rest_mass_scale = 1.0;
  Real target_riso_over_m_surface = 0.0;
  Real min_dr = 0.0, max_dr = 0.0;
};

//! \brief build R_*(rbar) by matching the sampler CDF fraction between the two models.
//! Interpolation is linear in q^(1/3) because both radii vary linearly in q^(1/3) at the
//! regular centre, where a linear-in-q interpolation would lose all accuracy.
DisplacementMap ConstructDisplacementMap(const ClusterProfile &start, Real target_yc,
                                         Real dx) {
  DisplacementMap map;
  ClusterProfile tgt = ConstructProfile(target_yc, dx);
  map.target_yc = target_yc;
  map.start_binding = 1.0 - start.vs/start.v0s;
  map.target_binding = 1.0 - tgt.vs/tgt.v0s;
  // Fix the REST mass, not the gravitational mass: radii of the target model are quoted in
  // units of its own M, so rescale by (M0/M)_start/(M0/M)_target.  For the default
  // equal-binding partner this factor is 1 to roundoff.
  map.rest_mass_scale = start.rest_mass_over_m/tgt.rest_mass_over_m;
  map.target_riso_over_m_surface = tgt.riso_over_m_surface*map.rest_mass_scale;

  std::size_t nt = tgt.x.size();
  std::vector<Real> s_t(nt), r_t(nt);
  for (std::size_t j = 0; j < nt; ++j) {
    s_t[j] = std::cbrt(tgt.cdf[j]);
    r_t[j] = tgt.riso_over_m[j]*map.rest_mass_scale;
  }
  std::size_t ns = start.x.size();
  map.r_map.assign(ns, 0.0);
  map.dr_map.assign(ns, 1.0);
  std::size_t j = 1;
  for (std::size_t i = 0; i < ns; ++i) {
    Real s = std::cbrt(start.cdf[i]);
    while (j + 1 < nt && s_t[j] < s) { ++j; }
    Real denom = s_t[j] - s_t[j-1];
    Real frac = (denom > 0.0) ? (s - s_t[j-1])/denom : 0.0;
    frac = std::min(std::max(frac, static_cast<Real>(0.0)), static_cast<Real>(1.0));
    map.r_map[i] = r_t[j-1] + frac*(r_t[j] - r_t[j-1]);
  }
  map.r_map[0] = 0.0;
  map.r_map[ns-1] = map.target_riso_over_m_surface;
  for (std::size_t i = 1; i + 1 < ns; ++i) {
    Real dr = start.riso_over_m[i+1] - start.riso_over_m[i-1];
    map.dr_map[i] = (dr > 0.0) ? (map.r_map[i+1] - map.r_map[i-1])/dr : 1.0;
  }
  if (ns > 2) {
    map.dr_map[0] = map.dr_map[1];
    map.dr_map[ns-1] = map.dr_map[ns-2];
  }
  map.min_dr = map.dr_map[0];
  map.max_dr = map.dr_map[0];
  for (std::size_t i = 0; i < ns; ++i) {
    map.min_dr = std::min(map.min_dr, map.dr_map[i]);
    map.max_dr = std::max(map.max_dr, map.dr_map[i]);
  }
  return map;
}

//! \brief <W> and <1/W> over an isotropic mu in [-1,1] for
//! W = sqrt(1 + a^2 mu^2 + b^2 (1-mu^2)); both closed form.
void IsotropicLorentzMoments(Real a2, Real b2, Real *mean_w, Real *mean_inv_w) {
  Real c = 1.0 + b2;
  Real d = a2 - b2;
  if (std::abs(d) < 1.0e-13*std::max(c, static_cast<Real>(1.0))) {
    *mean_w = std::sqrt(c);
    *mean_inv_w = 1.0/std::sqrt(c);
    return;
  }
  Real sd = std::sqrt(std::abs(d));
  Real z = (d > 0.0) ? std::asinh(std::sqrt(d/c)) : std::asin(std::sqrt(-d/c));
  *mean_w = 0.5*std::sqrt(std::max(c + d, static_cast<Real>(0.0))) + 0.5*c*z/sd;
  *mean_inv_w = z/sd;
}

//! \struct DisplacementMetric
//! \brief incremental spherical Hamiltonian and maximal-slicing solves for the displaced
//! matter.  Both are written as corrections to the exact eps=0 solution using the SAME
//! sampler continuum in both terms, so eps=0 is an exact no-op and the validated ST
//! initialization is reproduced bit for bit.
//!
//!   Laplacian(psi)    = -2 pi D_eps <W>_eps / psi                        (K_ij = 0)
//!   Laplacian(alpha psi) = 2 pi (alpha psi) D_eps <3W - 2/W>_eps / psi^2 (maximal slicing)
//!
//! with D_eps(R) = D_0(rbar) rbar^2 / (R^2 R') the mapped coordinate rest-mass density.
struct DisplacementMetric {
  std::vector<Real> radius, conformal_a, lapse;
  Real delta_adm_mass = 0.0;
  Real max_rel_delta_psi = 0.0;
  Real max_rel_delta_alpha = 0.0;
  Real central_alpha = 0.0;
  Real central_conformal_a = 0.0;
  Real surface_radius = 0.0;
  Real psi_residual = 0.0, lapse_residual = 0.0;
  int psi_iterations = 0, lapse_iterations = 0;
};

DisplacementMetric ConstructDisplacementMetric(const ClusterProfile &profile,
                                               const DisplacementMap &map,
                                               Real total_mass, Real eps) {
  constexpr int n_outer = 2048;
  constexpr int max_iterations = 20000;
  constexpr Real relaxation = 0.5;
  constexpr Real tolerance = 5.0e-14;
  DisplacementMetric out;
  std::size_t nm = profile.x.size();
  std::size_t n = nm + static_cast<std::size_t>(n_outer);
  Real rest_mass = total_mass*profile.rest_mass_over_m;
  Real zeta = (1.0 - profile.y.front())/3.0;

  std::vector<Real> r0(n), psi0(n), alpha0(n), d0(n), mw0(n), mwi0(n);
  std::vector<Real> rr(n), dmap(n), dnew(n), umag(n);
  for (std::size_t i = 0; i < nm; ++i) {
    Real r = total_mass*profile.riso_over_m[i];
    r0[i] = r;
    psi0[i] = std::sqrt(profile.conformal_a[i]);
    alpha0[i] = profile.lapse[i];
    mw0[i] = 1.0/std::sqrt(profile.y[i]);
    mwi0[i] = std::sqrt(profile.y[i]);
    umag[i] = profile.conformal_a[i]*
              std::sqrt(std::max(1.0/profile.y[i] - 1.0, static_cast<Real>(0.0)));
    if (i > 0) {
      Real x = profile.x[i];
      Real compact = 1.0 - 2.0*zeta*profile.v[i]/x;
      Real driso_dx = r/(x*std::sqrt(compact));
      Real inv_y_minus_one = std::max(1.0/profile.y[i] - 1.0, static_cast<Real>(0.0));
      Real dprob_dx = x*x/profile.y[i]*std::sqrt(inv_y_minus_one)/profile.cdf_normalization;
      Real denom = 4.0*M_PI*r*r*driso_dx;
      d0[i] = (denom > 0.0) ? rest_mass*dprob_dx/denom : 0.0;
    }
    Real rmap = total_mass*map.r_map[i];
    rr[i] = (1.0 - eps)*r + eps*rmap;
    dmap[i] = (1.0 - eps) + eps*map.dr_map[i];
  }
  d0[0] = d0.size() > 1 ? d0[1] : 0.0;
  Real start_surface = r0[nm-1];
  Real new_surface = rr[nm-1];
  out.surface_radius = new_surface;
  if (!(new_surface > 0.0) || map.min_dr <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "cluster_scpd_eps produced a non-monotonic radial map."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Real outer_limit = std::max(1.0e6*total_mass, 10.0*std::max(new_surface, start_surface));
  for (int jj = 0; jj < n_outer; ++jj) {
    std::size_t i = nm + static_cast<std::size_t>(jj);
    Real f = static_cast<Real>(jj)/static_cast<Real>(n_outer - 1);
    Real rs = new_surface*(1.0 + 1.0e-9);
    Real r = std::exp(std::log(rs) + f*(std::log(outer_limit) - std::log(rs)));
    rr[i] = r;
    dmap[i] = 1.0;
    umag[i] = 0.0;
    mw0[i] = 1.0; mwi0[i] = 1.0; d0[i] = 0.0;
    Real rs0 = start_surface*(1.0 + 1.0e-9);
    Real r_ref = std::exp(std::log(rs0) + f*(std::log(outer_limit) - std::log(rs0)));
    r0[i] = r_ref;
    psi0[i] = 1.0 + 0.5*total_mass/r_ref;
    alpha0[i] = (1.0 - 0.5*total_mass/r_ref)/(1.0 + 0.5*total_mass/r_ref);
  }
  for (std::size_t i = 0; i < n; ++i) {
    dnew[i] = (i < nm && rr[i] > 0.0)
        ? d0[i]*r0[i]*r0[i]/(rr[i]*rr[i]*dmap[i]) : 0.0;
  }
  if (nm > 1) { dnew[0] = dnew[1]; }

  out.radius.assign(rr.begin(), rr.end());
  out.conformal_a.assign(n, 1.0);
  out.lapse.assign(n, 1.0);

  // Reference fields evaluated at the NEW field points.
  std::vector<Real> psi_ref(n), d0_ref(n), mw0_ref(n), mwi0_ref(n), ap_ref(n);
  std::vector<Real> ap0(n);
  for (std::size_t i = 0; i < n; ++i) { ap0[i] = alpha0[i]*psi0[i]; }
  for (std::size_t i = 0; i < n; ++i) {
    Real r = rr[i];
    psi_ref[i] = InterpolateRadius(r0, psi0, r);
    ap_ref[i] = InterpolateRadius(r0, ap0, r);
    d0_ref[i] = (r <= start_surface) ? InterpolateRadius(r0, d0, r) : 0.0;
    mw0_ref[i] = (r <= start_surface) ? InterpolateRadius(r0, mw0, r) : 1.0;
    mwi0_ref[i] = (r <= start_surface) ? InterpolateRadius(r0, mwi0, r) : 1.0;
  }

  // eps == 0 is the exact mathematical zero of both increments.
  if (eps == 0.0) {
    for (std::size_t i = 0; i < n; ++i) {
      out.conformal_a[i] = psi_ref[i]*psi_ref[i];
      out.lapse[i] = ap_ref[i]/psi_ref[i];
    }
    out.central_alpha = out.lapse[0];
    out.central_conformal_a = out.conformal_a[0];
    out.psi_iterations = 1;
    out.lapse_iterations = 1;
    return out;
  }

  std::vector<Real> psi(psi_ref), mw(n, 1.0), mwi(n, 1.0);
  std::vector<Real> src(n, 0.0), inner(n, 0.0), outer(n, 0.0), trial(n, 1.0);
  auto green = [&](const std::vector<Real> &q, std::vector<Real> *u) {
    inner[0] = 0.0;
    for (std::size_t i = 1; i < n; ++i) {
      Real a = rr[i-1], b = rr[i];
      inner[i] = inner[i-1] + 0.5*(a*a*q[i-1] + b*b*q[i])*(b - a);
    }
    outer[n-1] = 0.0;
    for (std::size_t i = n-1; i > 0; --i) {
      Real a = rr[i-1], b = rr[i];
      outer[i-1] = outer[i] + 0.5*(a*q[i-1] + b*q[i])*(b - a);
    }
    (*u)[0] = outer[0];
    for (std::size_t i = 1; i < n; ++i) { (*u)[i] = outer[i] + inner[i]/rr[i]; }
  };
  auto moments = [&](void) {
    for (std::size_t i = 0; i < nm; ++i) {
      Real a_eps = psi[i]*psi[i];
      Real sr = umag[i]/(dmap[i]*a_eps);
      Real st = (rr[i] > 0.0) ? umag[i]*r0[i]/(rr[i]*a_eps) : umag[i]/(dmap[i]*a_eps);
      IsotropicLorentzMoments(sr*sr, st*st, &mw[i], &mwi[i]);
    }
    for (std::size_t i = nm; i < n; ++i) { mw[i] = 1.0; mwi[i] = 1.0; }
  };

  bool converged = false;
  std::vector<Real> corr(n, 0.0);
  for (int it = 0; it < max_iterations; ++it) {
    moments();
    for (std::size_t i = 0; i < n; ++i) {
      src[i] = 2.0*M_PI*(dnew[i]*mw[i]/psi[i] - d0_ref[i]*mw0_ref[i]/psi_ref[i]);
    }
    green(src, &corr);
    Real update = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      trial[i] = psi_ref[i] + corr[i];
      update = std::max(update, std::abs(trial[i] - psi[i])/psi_ref[i]);
    }
    for (std::size_t i = 0; i < n; ++i) {
      psi[i] = (1.0 - relaxation)*psi[i] + relaxation*trial[i];
    }
    out.psi_iterations = it + 1;
    out.psi_residual = update;
    if (update < tolerance) { converged = true; break; }
  }
  if (!converged) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "SCPD Hamiltonian solve did not converge; residual="
              << out.psi_residual << "." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  moments();

  std::vector<Real> ap(ap_ref);
  converged = false;
  for (int it = 0; it < max_iterations; ++it) {
    for (std::size_t i = 0; i < n; ++i) {
      Real q_new = -2.0*M_PI*ap[i]*dnew[i]*(3.0*mw[i] - 2.0*mwi[i])/(psi[i]*psi[i]);
      Real q_ref = -2.0*M_PI*ap_ref[i]*d0_ref[i]*(3.0*mw0_ref[i] - 2.0*mwi0_ref[i])/
                   (psi_ref[i]*psi_ref[i]);
      src[i] = q_new - q_ref;
    }
    green(src, &corr);
    Real update = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      trial[i] = ap_ref[i] + corr[i];
      update = std::max(update, std::abs(trial[i] - ap[i]));
    }
    for (std::size_t i = 0; i < n; ++i) {
      ap[i] = (1.0 - relaxation)*ap[i] + relaxation*trial[i];
    }
    out.lapse_iterations = it + 1;
    out.lapse_residual = update;
    if (update < tolerance) { converged = true; break; }
  }
  if (!converged) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "SCPD maximal-slicing lapse solve did not converge; residual="
              << out.lapse_residual << "." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  for (std::size_t i = 0; i < n; ++i) {
    out.conformal_a[i] = psi[i]*psi[i];
    out.lapse[i] = ap[i]/psi[i];
    out.max_rel_delta_psi = std::max(out.max_rel_delta_psi,
                                     std::abs(psi[i]/psi_ref[i] - 1.0));
    out.max_rel_delta_alpha = std::max(out.max_rel_delta_alpha,
        std::abs(out.lapse[i]*psi_ref[i]/ap_ref[i] - 1.0));
  }
  out.central_alpha = out.lapse[0];
  out.central_conformal_a = out.conformal_a[0];
  out.delta_adm_mass = 2.0*(psi[n-1] - 1.0)*rr[n-1] - total_mass;
  return out;
}

//! \brief apply the canonical phase-space displacement to one sampled particle.
//! Returns false only on an invalid (non-monotonic or non-timelike) map.
bool ApplyDisplacement(DrawnParticle *d, Real eps, const DisplacementMap &map,
                       const ClusterProfile &profile, const DisplacementMetric &metric,
                       Real total_mass, const Real center[3],
                       Real *r_before, Real *r_after, Real *dr, Real *rel_dl,
                       Real *energy, Real *v2_after) {
  Real nmag = std::sqrt(d->nx*d->nx + d->ny*d->ny + d->nz*d->nz);
  if (!(nmag > 0.0)) { return false; }
  Real nx = d->nx/nmag, ny = d->ny/nmag, nz = d->nz/nmag;
  Real rstar = total_mass*InterpolateCDF(profile.cdf, map.r_map, d->cdf_index, d->cdf_u);
  Real drstar = InterpolateCDF(profile.cdf, map.dr_map, d->cdf_index, d->cdf_u);
  Real rnew = (1.0 - eps)*d->riso + eps*rstar;
  Real drnew = (1.0 - eps) + eps*drstar;
  if (!(rnew > 0.0) || !(drnew > 0.0)) { return false; }

  Real ur = d->ux*nx + d->uy*ny + d->uz*nz;
  Real tx = d->ux - ur*nx, ty = d->uy - ur*ny, tz = d->uz - ur*nz;
  Real tangential_scale = (d->riso > 0.0) ? d->riso/rnew : 1.0/drnew;
  Real ur_new = ur/drnew;
  Real l_before = d->riso*std::sqrt(std::max(tx*tx + ty*ty + tz*tz,
                                             static_cast<Real>(0.0)));
  d->ux = ur_new*nx + tangential_scale*tx;
  d->uy = ur_new*ny + tangential_scale*ty;
  d->uz = ur_new*nz + tangential_scale*tz;
  Real ur_check = d->ux*nx + d->uy*ny + d->uz*nz;
  Real cx = d->ux - ur_check*nx, cy = d->uy - ur_check*ny, cz = d->uz - ur_check*nz;
  Real l_after = rnew*std::sqrt(std::max(cx*cx + cy*cy + cz*cz, static_cast<Real>(0.0)));

  *r_before = d->riso;
  *r_after = rnew;
  *dr = drnew;
  *rel_dl = (l_before > 0.0) ? std::abs(l_after/l_before - 1.0) : 0.0;

  d->px = center[0] + rnew*nx;
  d->py = center[1] + rnew*ny;
  d->pz = center[2] + rnew*nz;
  d->riso = rnew;
  d->conf_a = InterpolateRadius(metric.radius, metric.conformal_a, rnew);
  if (!(d->conf_a > 0.0)) { return false; }
  d->umag = std::sqrt(d->ux*d->ux + d->uy*d->uy + d->uz*d->uz);
  Real uhat2 = d->umag*d->umag/(d->conf_a*d->conf_a);
  Real w = std::sqrt(1.0 + uhat2);
  Real alpha = InterpolateRadius(metric.radius, metric.lapse, rnew);
  *energy = alpha*w;
  *v2_after = uhat2/(1.0 + uhat2);
  return std::isfinite(*energy) && std::isfinite(d->umag) && (*v2_after < 1.0);
}

//! \struct CTTMomentumProfile
//! \brief spherical conformal-transverse-traceless momentum-constraint correction.
//!
//! With gamma_ij=psi^4 delta_ij, K=0 and
//!
//!   (L W)^ij = 2 a(r) (n^i n^j - delta^ij/3),  a=W'-W/r,
//!
//! the radial momentum constraint is
//!
//!   W'' + 2 W'/r - 2 W/r^2 = F(r),  F=6 pi psi^6 S_r,
//!
//! because the particle deposition stores the covariant Eulerian momentum density S_r.
//! The regular/asymptotically-decaying Green solution gives the especially simple
//!
//!   a(r) = r^-3 integral_0^r s^3 F(s) ds.
//!
//! The source below is the exact spherical continuum average of the sampled equilibrium
//! distribution after AddCoherentRadialMode. It removes the coherent current while leaving
//! the unavoidable finite-N nonspherical shot-noise part to the evolution constraints.

struct CTTMomentumProfile {
  std::vector<Real> a;
  Real moment = 0.0;       // integral_0^R r^3 F dr; a=moment/r^3 outside matter
  Real max_abs_source = 0.0;
  Real max_abs_a = 0.0;
  Real max_abs_mean_uhat_r = 0.0;
};

//! \brief angular average <u_(hat r)> after adding q=delta v^(hat r).
//!
//! The base velocity has magnitude sqrt(1-y) and isotropic mu. A symmetric 64-point
//! midpoint quadrature makes the q=0 result cancel pairwise exactly and is amply converged
//! for the small Session-04 amplitudes.
Real MeanPerturbedRadialUhat(Real y, Real q) {
  constexpr int nmu_half = 32;
  Real v0 = std::sqrt(std::max(static_cast<Real>(1.0) - y,
                               static_cast<Real>(0.0)));
  Real sum = 0.0;
  for (int k = 0; k < nmu_half; ++k) {
    Real mu = (static_cast<Real>(k) + 0.5)/static_cast<Real>(nmu_half);
    Real den_plus = y - q*q - 2.0*v0*q*mu;
    Real den_minus = y - q*q + 2.0*v0*q*mu;
    if (!(den_plus > 0.0) || !(den_minus > 0.0)) {
      return std::numeric_limits<Real>::quiet_NaN();
    }
    sum += (v0*mu + q)/std::sqrt(den_plus);
    sum += (-v0*mu + q)/std::sqrt(den_minus);
  }
  return sum/static_cast<Real>(2*nmu_half);
}

CTTMomentumProfile ConstructCTTMomentumProfile(const ClusterProfile &profile,
                                                Real total_mass,
                                                Real radial_mode_u) {
  CTTMomentumProfile correction;
  std::size_t nprof = profile.x.size();
  correction.a.assign(nprof, 0.0);
  std::vector<Real> source(nprof, 0.0);
  std::vector<Real> integrand(nprof, 0.0);
  std::vector<Real> cumulative(nprof, 0.0);
  Real zeta = (1.0 - profile.y.front())/3.0;
  Real rest_mass = total_mass*profile.rest_mass_over_m;

  for (std::size_t i = 1; i < nprof; ++i) {
    Real x = profile.x[i];
    Real riso = total_mass*profile.riso_over_m[i];
    Real compact = 1.0 - 2.0*zeta*profile.v[i]/x;
    Real radial_factor = 1.0/std::sqrt(compact);
    // From rbar proportional to x exp(h), d rbar/dx = rbar*radial_factor/x.
    Real driso_dx = riso*radial_factor/x;
    Real inv_y_minus_one = std::max(static_cast<Real>(1.0)/profile.y[i] - 1.0,
                                    static_cast<Real>(0.0));
    Real cdf_weight = x*x/profile.y[i]*std::sqrt(inv_y_minus_one);
    Real dprob_dx = cdf_weight/profile.cdf_normalization;
    Real x_areal = x/profile.xs;
    Real q = 0.5*radial_mode_u*(3.0*x_areal - x_areal*x_areal*x_areal);
    Real mean_uhat_r = MeanPerturbedRadialUhat(profile.y[i], q);
    if (!std::isfinite(mean_uhat_r)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Non-timelike continuum velocity in CTT source at x="
                << x << ", U=" << radial_mode_u << "." << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // rho_0 = M0 (dP/dx)/(4 pi A^3 rbar^2 drbar/dx), S_r=A rho_0<u_hat_r>.
    // Hence F=6 pi psi^6 S_r=(3/2) M0 A (dP/dx)<u_hat_r>/(rbar^2 drbar/dx).
    Real denom = riso*riso*driso_dx;
    source[i] = (denom > 0.0)
        ? 1.5*rest_mass*profile.conformal_a[i]*dprob_dx*mean_uhat_r/denom
        : 0.0;
    integrand[i] = riso*riso*riso*source[i];
    correction.max_abs_source = std::max(correction.max_abs_source,
                                         std::abs(source[i]));
    correction.max_abs_mean_uhat_r = std::max(correction.max_abs_mean_uhat_r,
                                              std::abs(mean_uhat_r));
  }

  for (std::size_t i = 1; i < nprof; ++i) {
    Real r0 = total_mass*profile.riso_over_m[i-1];
    Real r1 = total_mass*profile.riso_over_m[i];
    cumulative[i] = cumulative[i-1] +
                    0.5*(integrand[i-1] + integrand[i])*(r1 - r0);
    correction.a[i] = cumulative[i]/(r1*r1*r1);
    correction.max_abs_a = std::max(correction.max_abs_a,
                                    std::abs(correction.a[i]));
  }
  correction.moment = cumulative.back();
  return correction;
}

//! \brief angular average <W> after adding q=delta v^(hat r).
Real MeanPerturbedLorentz(Real y, Real q) {
  constexpr int nmu_half = 32;
  Real v0 = std::sqrt(std::max(static_cast<Real>(1.0) - y,
                               static_cast<Real>(0.0)));
  Real sum = 0.0;
  for (int k = 0; k < nmu_half; ++k) {
    Real mu = (static_cast<Real>(k) + 0.5)/static_cast<Real>(nmu_half);
    Real den_plus = y - q*q - 2.0*v0*q*mu;
    Real den_minus = y - q*q + 2.0*v0*q*mu;
    if (!(den_plus > 0.0) || !(den_minus > 0.0)) {
      return std::numeric_limits<Real>::quiet_NaN();
    }
    sum += 1.0/std::sqrt(den_plus) + 1.0/std::sqrt(den_minus);
  }
  return sum/static_cast<Real>(2*nmu_half);
}

//! \struct IncrementalConstraintProfile
//! \brief coupled spherical CTT correction relative to the exact ST equilibrium.
//!
//! The historical pgen samples the printed Eq. (A18) radial measure, whereas the smooth
//! equilibrium ODE uses the proper-volume rest-mass measure. An absolute particle-source
//! Hamiltonian solve would therefore move even U=0 away from the already validated ST
//! equilibrium. Instead write psi=psi_0+delta psi and solve only for the change made by
//! the coherent mode, using the same A18 coordinate density in both terms:
//!
//!   Laplacian(delta psi) = -delta Q,
//!   delta Q = 2 pi D [<W>_U/psi - <W>_0/psi_0] + a^2/(3 psi^7),
//!   D = M0 (dP/dr)/(4 pi r^2),
//!   F = 6 pi D psi^2 <u_hat_r>,  a=r^-3 integral_0^r s^3 F ds.
//!
//! Thus U=0 is an exact no-op by construction. The logarithmic exterior grid resolves
//! the a=C/r^3 Hamiltonian tail and enforces delta psi -> 0 at infinity.
struct IncrementalConstraintProfile {
  std::vector<Real> radius;
  std::vector<Real> conformal_a;
  std::vector<Real> a;
  Real moment = 0.0;
  Real delta_adm_mass = 0.0;
  Real max_abs_source = 0.0;
  Real max_abs_a = 0.0;
  Real max_abs_mean_uhat_r = 0.0;
  Real max_abs_delta_psi = 0.0;
  Real max_rel_delta_psi = 0.0;
  Real max_abs_delta_q = 0.0;
  Real final_relative_update = 0.0;
  int iterations = 0;
};

IncrementalConstraintProfile ConstructIncrementalConstraintProfile(
    const ClusterProfile &profile, Real total_mass, Real radial_mode_u) {
  constexpr int n_outer = 2048;
  constexpr int max_iterations = 10000;
  constexpr Real relaxation = 0.35;
  constexpr Real tolerance = 2.0e-13;
  IncrementalConstraintProfile correction;
  std::size_t n_matter = profile.x.size();
  std::size_t n_total = n_matter + n_outer;
  correction.radius.resize(n_total);
  correction.conformal_a.resize(n_total);
  correction.a.assign(n_total, 0.0);

  std::vector<Real> psi0(n_total, 1.0), psi(n_total, 1.0);
  std::vector<Real> dcoord(n_total, 0.0), mean_w(n_total, 1.0);
  std::vector<Real> mean_w0(n_total, 1.0), mean_uhat_r(n_total, 0.0);
  std::vector<Real> source(n_total, 0.0), moment(n_total, 0.0);
  std::vector<Real> delta_q(n_total, 0.0), inner(n_total, 0.0);
  std::vector<Real> outer(n_total, 0.0), trial(n_total, 1.0);
  Real rest_mass = total_mass*profile.rest_mass_over_m;
  Real zeta = (1.0 - profile.y.front())/3.0;

  for (std::size_t i = 0; i < n_matter; ++i) {
    correction.radius[i] = total_mass*profile.riso_over_m[i];
    psi0[i] = std::sqrt(profile.conformal_a[i]);
    psi[i] = psi0[i];
    mean_w0[i] = 1.0/std::sqrt(profile.y[i]);
    Real xfrac = (i == 0) ? 0.0 : profile.x[i]/profile.xs;
    Real q = 0.5*radial_mode_u*(3.0*xfrac - xfrac*xfrac*xfrac);
    mean_w[i] = MeanPerturbedLorentz(profile.y[i], q);
    mean_uhat_r[i] = MeanPerturbedRadialUhat(profile.y[i], q);
    if (!std::isfinite(mean_w[i]) || !std::isfinite(mean_uhat_r[i])) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Non-timelike continuum velocity in coupled CTT source."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    correction.max_abs_mean_uhat_r = std::max(
        correction.max_abs_mean_uhat_r, std::abs(mean_uhat_r[i]));
    if (i > 0) {
      Real x = profile.x[i];
      Real r = correction.radius[i];
      Real compact = 1.0 - 2.0*zeta*profile.v[i]/x;
      Real radial_factor = 1.0/std::sqrt(compact);
      Real driso_dx = r*radial_factor/x;
      Real inv_y_minus_one = std::max(1.0/profile.y[i] - 1.0,
                                      static_cast<Real>(0.0));
      Real cdf_weight = x*x/profile.y[i]*std::sqrt(inv_y_minus_one);
      Real dprob_dx = cdf_weight/profile.cdf_normalization;
      Real denom = 4.0*M_PI*r*r*driso_dx;
      dcoord[i] = (denom > 0.0) ? rest_mass*dprob_dx/denom : 0.0;
    }
  }

  Real surface = correction.radius[n_matter - 1];
  Real outer_limit = std::max(1.0e6*total_mass, 10.0*surface);
  Real log_start = std::log(surface*(1.0 + 1.0e-8));
  Real log_end = std::log(outer_limit);
  for (int j = 0; j < n_outer; ++j) {
    std::size_t i = n_matter + static_cast<std::size_t>(j);
    Real f = static_cast<Real>(j)/static_cast<Real>(n_outer - 1);
    Real r = std::exp(log_start + f*(log_end - log_start));
    correction.radius[i] = r;
    psi0[i] = 1.0 + 0.5*total_mass/r;
    psi[i] = psi0[i];
  }

  // This branch is both a regression guarantee and the mathematical zero of the
  // incremental equations: no arithmetic is allowed to perturb the U=0 metric.
  if (radial_mode_u == 0.0) {
    for (std::size_t i = 0; i < n_total; ++i) {
      correction.conformal_a[i] = psi0[i]*psi0[i];
    }
    correction.iterations = 1;
    return correction;
  }

  bool converged = false;
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    for (std::size_t i = 0; i < n_total; ++i) {
      source[i] = 6.0*M_PI*dcoord[i]*psi[i]*psi[i]*mean_uhat_r[i];
    }
    moment[0] = 0.0;
    for (std::size_t i = 1; i < n_total; ++i) {
      Real r0 = correction.radius[i-1], r1 = correction.radius[i];
      Real f0 = r0*r0*r0*source[i-1];
      Real f1 = r1*r1*r1*source[i];
      moment[i] = moment[i-1] + 0.5*(f0 + f1)*(r1 - r0);
      correction.a[i] = moment[i]/(r1*r1*r1);
    }
    correction.a[0] = 0.0;
    for (std::size_t i = 0; i < n_total; ++i) {
      Real matter_change = 2.0*M_PI*dcoord[i]*
          (mean_w[i]/psi[i] - mean_w0[i]/psi0[i]);
      Real extrinsic = correction.a[i]*correction.a[i]/
                       (3.0*std::pow(psi[i], 7));
      delta_q[i] = matter_change + extrinsic;
    }
    inner[0] = 0.0;
    for (std::size_t i = 1; i < n_total; ++i) {
      Real r0 = correction.radius[i-1], r1 = correction.radius[i];
      Real f0 = r0*r0*delta_q[i-1];
      Real f1 = r1*r1*delta_q[i];
      inner[i] = inner[i-1] + 0.5*(f0 + f1)*(r1 - r0);
    }
    outer[n_total-1] = 0.0;
    for (std::size_t i = n_total-1; i > 0; --i) {
      Real r0 = correction.radius[i-1], r1 = correction.radius[i];
      Real f0 = r0*delta_q[i-1];
      Real f1 = r1*delta_q[i];
      outer[i-1] = outer[i] + 0.5*(f0 + f1)*(r1 - r0);
    }
    Real radius_last = correction.radius.back();
    Real ctt_moment = moment.back();
    Real tail = ctt_moment*ctt_moment/
                (12.0*std::pow(radius_last, 4));
    Real max_relative_update = 0.0;
    for (std::size_t i = 0; i < n_total; ++i) {
      Real green = outer[i] + tail;
      if (i > 0) { green += inner[i]/correction.radius[i]; }
      trial[i] = psi0[i] + green;
      max_relative_update = std::max(
          max_relative_update, std::abs(trial[i] - psi[i])/psi0[i]);
    }
    for (std::size_t i = 0; i < n_total; ++i) {
      psi[i] = (1.0 - relaxation)*psi[i] + relaxation*trial[i];
    }
    correction.iterations = iteration + 1;
    correction.final_relative_update = max_relative_update;
    if (max_relative_update < tolerance) {
      converged = true;
      break;
    }
  }
  if (!converged) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Incremental coupled CTT solve did not converge after "
              << max_iterations << " iterations; last relative update="
              << correction.final_relative_update << "." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Re-evaluate all sources at the converged iterate for diagnostics and ADM mass.
  for (std::size_t i = 0; i < n_total; ++i) {
    source[i] = 6.0*M_PI*dcoord[i]*psi[i]*psi[i]*mean_uhat_r[i];
    correction.max_abs_source = std::max(correction.max_abs_source,
                                         std::abs(source[i]));
  }
  moment[0] = 0.0;
  for (std::size_t i = 1; i < n_total; ++i) {
    Real r0 = correction.radius[i-1], r1 = correction.radius[i];
    Real f0 = r0*r0*r0*source[i-1];
    Real f1 = r1*r1*r1*source[i];
    moment[i] = moment[i-1] + 0.5*(f0 + f1)*(r1 - r0);
    correction.a[i] = moment[i]/(r1*r1*r1);
    correction.max_abs_a = std::max(correction.max_abs_a,
                                    std::abs(correction.a[i]));
  }
  correction.a[0] = 0.0;
  correction.moment = moment.back();
  inner[0] = 0.0;
  for (std::size_t i = 0; i < n_total; ++i) {
    Real matter_change = 2.0*M_PI*dcoord[i]*
        (mean_w[i]/psi[i] - mean_w0[i]/psi0[i]);
    Real extrinsic = correction.a[i]*correction.a[i]/
                     (3.0*std::pow(psi[i], 7));
    delta_q[i] = matter_change + extrinsic;
    correction.max_abs_delta_q = std::max(correction.max_abs_delta_q,
                                          std::abs(delta_q[i]));
    correction.max_abs_delta_psi = std::max(correction.max_abs_delta_psi,
                                            std::abs(psi[i] - psi0[i]));
    correction.max_rel_delta_psi = std::max(correction.max_rel_delta_psi,
                                            std::abs(psi[i]/psi0[i] - 1.0));
    correction.conformal_a[i] = psi[i]*psi[i];
    if (i > 0) {
      Real r0 = correction.radius[i-1], r1 = correction.radius[i];
      Real f0 = r0*r0*delta_q[i-1];
      Real f1 = r1*r1*delta_q[i];
      inner[i] = inner[i-1] + 0.5*(f0 + f1)*(r1 - r0);
    }
  }
  Real radius_last = correction.radius.back();
  Real mass_tail = 2.0*correction.moment*correction.moment/
                   (9.0*std::pow(radius_last, 3));
  correction.delta_adm_mass = 2.0*inner.back() + mass_tail;
  return correction;
}

//! \struct BiasStats
//! \brief t=0 bookkeeping for the outward bias, accumulated over ALL tags (every rank
//! reproduces the same global draw, so rank 0 can report them without any reduction).

struct BiasStats {
  std::int64_t n_inward_base = 0;    // particles with u_i n^i < 0 in the base sample
  std::int64_t n_reoriented = 0;     // particles receiving the tangential map
  std::int64_t n_reflected = 0;      // reflections actually applied
  std::int64_t n_selected = 0;       // tags selected by the bias rule
  Real sum_mu_base = 0.0;            // sum of mu = (u_i n^i)/|u| before the bias
  Real sum_mu = 0.0;                 // ... and after
  Real sum_mu2_base = 0.0;           // sum of mu^2 before the perturbations
  Real sum_mu2 = 0.0;                // ... and after
  Real sum_abs_l_base = 0.0;         // sum of |(x-center) cross u| before
  Real sum_abs_l = 0.0;              // ... and after
  Real sum_vr_base = 0.0;            // sum of the local radial 3-velocity before
  Real sum_vr = 0.0;                 // ... and after
  Real sum_pr = 0.0;                 // sum of m0 (u_i n^i)/A  (radial momentum)
  Real p_tot[3] = {0.0, 0.0, 0.0};   // sum of m0 u_i/A        (linear momentum)
  Real j_tot[3] = {0.0, 0.0, 0.0};   // sum of m0 (x cross u)_i (angular momentum)
  Real max_rel_dumag = 0.0;          // max |(|u'| - |u|)|/|u| over all perturbations
  Real sum_mode_dv = 0.0;            // coherent-mode delta v^(hat r)
  Real sum_mode_vr_before = 0.0;     // Eulerian radial 3-velocity immediately pre-mode
  Real sum_mode_vr_after = 0.0;      // ... post-mode
  Real sum_mode_w_before = 0.0;      // Eulerian Lorentz factor immediately pre-mode
  Real sum_mode_w_after = 0.0;       // ... post-mode
  Real min_mode_dv = 1.0;            // extrema are meaningful only when U != 0
  Real max_mode_dv = -1.0;
  Real max_mode_v2 = 0.0;
  Real max_abs_delta_vtan = 0.0;
  Real max_mode_normalization_residual = 0.0;
  // Sequence-tangent canonical phase-space displacement bookkeeping.
  Real sum_scpd_r_before = 0.0;      // sum of the unperturbed isotropic radius
  Real sum_scpd_r_after = 0.0;       // ... and of the displaced one
  Real min_scpd_ratio = 1.0e300;     // extrema of R/rbar over the sample
  Real max_scpd_ratio = -1.0e300;
  Real min_scpd_dr = 1.0e300;        // extrema of R'(rbar)
  Real max_scpd_dr = -1.0e300;
  Real max_scpd_rel_dl = 0.0;        // max |L_after/L_before - 1| (must be roundoff)
  Real sum_scpd_energy = 0.0;        // sum of E = -u_t = alpha W after the map
  Real sum_scpd_energy2 = 0.0;
  Real min_scpd_energy = 1.0e300;
  Real max_scpd_energy = -1.0e300;
  Real max_scpd_v2 = 0.0;            // max local 3-velocity squared after the map
  Real sum_scpd_vr2 = 0.0;           // sum of v_hat_r^2 after the map
  Real sum_scpd_vt2 = 0.0;           // sum of v_hat_t^2 after the map
};

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  // The hook is inert on uniform/static meshes. Adaptive Session-02 decks use the
  // standard Z4c chi/radius criteria, following the proven OS particle-AMR pattern.
  user_ref_func = STRefinementCondition;
  auto &indcs = pmy_mesh_->mb_indcs;

  if (pmbp->pz4c == nullptr || pmbp->padm == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "relativistic_cluster requires a <z4c> block." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->ppart == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "relativistic_cluster requires a <particles> block."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!pmy_mesh_->three_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "relativistic_cluster is 3D-only." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->ppart->pusher != ParticlesPusher::gr_boris || !pmbp->ppart->feedback) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "relativistic_cluster requires <particles> pusher=gr_boris "
              << "and feedback=true." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto SeedSnapshots = [&]() {
    Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->adm_last, pmbp->padm->u_adm);
    Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->z4c_last, pmbp->pz4c->u0);
  };
  if (restart) {
    SeedSnapshots();
    return;
  }

  std::string init = pin->GetOrAddString("particles", "init", "ppc");
  if (init.compare("pgen") != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "relativistic_cluster requires <particles> init=pgen."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int npart_total = pin->GetOrAddInteger("problem", "cluster_n", 25);
  Real yc = pin->GetOrAddReal("problem", "cluster_yc", 0.819);
  Real total_mass = pin->GetOrAddReal("problem", "cluster_mass", 1.0);
  int seed = pin->GetOrAddInteger("problem", "cluster_seed", 1);
  Real profile_dx = pin->GetOrAddReal("problem", "cluster_profile_dx", 1.0e-4);
  Real center[3] = {
    pin->GetOrAddReal("problem", "cluster_center_x1", 0.0),
    pin->GetOrAddReal("problem", "cluster_center_x2", 0.0),
    pin->GetOrAddReal("problem", "cluster_center_x3", 0.0)
  };
  Real lambda_tan = pin->GetOrAddReal("problem", "cluster_lambda_tan", 0.0);
  Real eps_out = pin->GetOrAddReal("problem", "cluster_eps_out", 0.0);
  Real radial_mode_u = pin->GetOrAddReal("problem", "cluster_radial_mode_u", 0.0);
  Real scpd_eps = pin->GetOrAddReal("problem", "cluster_scpd_eps", 0.0);
  Real scpd_target_yc = pin->GetOrAddReal("problem", "cluster_scpd_target_yc", 0.0);
  // Diagnostic switch only: with the solves disabled the displaced matter is placed on the
  // UNPERTURBED equilibrium metric, which is how large the constraint violation would be
  // without the incremental solves.  Never use false for a physics run.
  bool scpd_solve_metric = pin->GetOrAddBoolean("problem",
                                                "cluster_scpd_solve_metric", true);
  bool solve_momentum_constraint = pin->GetOrAddBoolean(
      "problem", "cluster_solve_momentum_constraint", false);
  bool solve_hamiltonian_constraint = pin->GetOrAddBoolean(
      "problem", "cluster_solve_hamiltonian_constraint", false);
  std::string bias_mode = pin->GetOrAddString("problem", "cluster_bias_mode",
                                              "stratified");
  int bias_seed = pin->GetOrAddInteger("problem", "cluster_bias_seed", 0);
  std::string t0_dump = pin->GetOrAddString("problem", "cluster_t0_dump", "");
  if (npart_total <= 0 || yc <= 0.0 || yc >= 1.0 || total_mass <= 0.0 ||
      profile_dx <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Require cluster_n>0, 0<cluster_yc<1, cluster_mass>0, "
              << "and cluster_profile_dx>0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (lambda_tan < 0.0 || lambda_tan > 1.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Require 0 <= cluster_lambda_tan <= 1 (got "
              << lambda_tan << ")." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (eps_out < 0.0 || eps_out > 1.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Require 0 <= cluster_eps_out <= 1 (got " << eps_out
              << ")." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!std::isfinite(radial_mode_u) || std::abs(radial_mode_u) >= 1.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Require finite -1 < cluster_radial_mode_u < 1 (got "
              << radial_mode_u << ")." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (bias_mode.compare("stratified") != 0 && bias_mode.compare("bernoulli") != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "cluster_bias_mode must be 'stratified' or 'bernoulli' "
              << "(got '" << bias_mode << "')." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (solve_hamiltonian_constraint && !solve_momentum_constraint) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "cluster_solve_hamiltonian_constraint requires "
              << "cluster_solve_momentum_constraint=true." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!std::isfinite(scpd_eps)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "cluster_scpd_eps must be finite (got " << scpd_eps
              << ")." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (scpd_eps != 0.0 && (lambda_tan != 0.0 || eps_out != 0.0 ||
                          radial_mode_u != 0.0 || solve_momentum_constraint ||
                          solve_hamiltonian_constraint)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "cluster_scpd_eps is exclusive of the Session-01/02/04 "
              << "perturbations and of their constraint solvers; the displacement carries "
              << "its own Hamiltonian and maximal-slicing solves and leaves K_ij=0."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (scpd_target_yc != 0.0 && (scpd_target_yc <= 0.0 || scpd_target_yc >= 1.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Require 0 < cluster_scpd_target_yc < 1 (got "
              << scpd_target_yc << ")." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (solve_momentum_constraint && (lambda_tan != 0.0 || eps_out != 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "cluster_solve_momentum_constraint currently supports "
              << "the coherent radial mode alone; require lambda_tan=eps_out=0."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  ClusterProfile profile = ConstructProfile(yc, profile_dx);
  DisplacementMap displacement_map;
  DisplacementMetric displacement_metric;
  bool apply_displacement = (scpd_eps != 0.0);
  if (apply_displacement) {
    Real target_yc = (scpd_target_yc != 0.0)
        ? scpd_target_yc : ClusterEqualBindingPartnerYc(yc, profile_dx);
    displacement_map = ConstructDisplacementMap(profile, target_yc, profile_dx);
    displacement_metric = ConstructDisplacementMetric(profile, displacement_map,
                                                     total_mass,
                                                     scpd_solve_metric ? scpd_eps : 0.0);
  }
  CTTMomentumProfile momentum_correction;
  IncrementalConstraintProfile constraint_correction;
  if (solve_hamiltonian_constraint) {
    constraint_correction = ConstructIncrementalConstraintProfile(
        profile, total_mass, radial_mode_u);
  } else if (solve_momentum_constraint) {
    momentum_correction = ConstructCTTMomentumProfile(profile, total_mass, radial_mode_u);
  }
  // Preserve the historical U=0 initialization bit for bit even when both opt-in solver
  // flags are enabled as a regression test. The incremental solution is exactly zero.
  bool apply_hamiltonian_correction = solve_hamiltonian_constraint &&
                                      radial_mode_u != 0.0;

  // Copy the isotropic radial metric profile to the device.
  int nprof = static_cast<int>(profile.x.size());
  DvceArray1D<Real> radius_d("cluster_radius", nprof);
  DvceArray1D<Real> conf_d("cluster_conformal_a", nprof);
  DvceArray1D<Real> lapse_d("cluster_lapse", nprof);
  DvceArray1D<Real> ctt_a_d("cluster_ctt_a", nprof);
  int nham = solve_hamiltonian_constraint
      ? static_cast<int>(constraint_correction.radius.size()) : 1;
  DvceArray1D<Real> ham_radius_d("cluster_ham_radius", nham);
  DvceArray1D<Real> ham_conf_d("cluster_ham_conformal_a", nham);
  int nscpd = apply_displacement
      ? static_cast<int>(displacement_metric.radius.size()) : 1;
  DvceArray1D<Real> scpd_radius_d("cluster_scpd_radius", nscpd);
  DvceArray1D<Real> scpd_conf_d("cluster_scpd_conformal_a", nscpd);
  DvceArray1D<Real> scpd_lapse_d("cluster_scpd_lapse", nscpd);
  auto scpd_radius_h = Kokkos::create_mirror_view(scpd_radius_d);
  auto scpd_conf_h = Kokkos::create_mirror_view(scpd_conf_d);
  auto scpd_lapse_h = Kokkos::create_mirror_view(scpd_lapse_d);
  auto radius_h = Kokkos::create_mirror_view(radius_d);
  auto conf_h = Kokkos::create_mirror_view(conf_d);
  auto lapse_h = Kokkos::create_mirror_view(lapse_d);
  auto ctt_a_h = Kokkos::create_mirror_view(ctt_a_d);
  auto ham_radius_h = Kokkos::create_mirror_view(ham_radius_d);
  auto ham_conf_h = Kokkos::create_mirror_view(ham_conf_d);
  for (int i = 0; i < nprof; ++i) {
    radius_h(i) = total_mass*profile.riso_over_m[i];
    conf_h(i) = profile.conformal_a[i];
    lapse_h(i) = profile.lapse[i];
    ctt_a_h(i) = solve_hamiltonian_constraint ? constraint_correction.a[i] :
                   (solve_momentum_constraint ? momentum_correction.a[i] : 0.0);
  }
  Kokkos::deep_copy(radius_d, radius_h);
  Kokkos::deep_copy(conf_d, conf_h);
  Kokkos::deep_copy(lapse_d, lapse_h);
  if (solve_hamiltonian_constraint) {
    for (int i = 0; i < nham; ++i) {
      ham_radius_h(i) = constraint_correction.radius[i];
      ham_conf_h(i) = constraint_correction.conformal_a[i];
    }
  } else {
    ham_radius_h(0) = 0.0;
    ham_conf_h(0) = 1.0;
  }
  if (apply_displacement) {
    for (int i = 0; i < nscpd; ++i) {
      scpd_radius_h(i) = displacement_metric.radius[i];
      scpd_conf_h(i) = displacement_metric.conformal_a[i];
      scpd_lapse_h(i) = displacement_metric.lapse[i];
    }
  } else {
    scpd_radius_h(0) = 0.0;
    scpd_conf_h(0) = 1.0;
    scpd_lapse_h(0) = 1.0;
  }
  Kokkos::deep_copy(ctt_a_d, ctt_a_h);
  Kokkos::deep_copy(ham_radius_d, ham_radius_h);
  Kokkos::deep_copy(ham_conf_d, ham_conf_h);
  Kokkos::deep_copy(scpd_radius_d, scpd_radius_h);
  Kokkos::deep_copy(scpd_conf_d, scpd_conf_h);
  Kokkos::deep_copy(scpd_lapse_d, scpd_lapse_h);

  // Smooth equilibrium metric in isotropic Cartesian coordinates. The optional
  // Hamiltonian solve replaces A=psi^2 by its incremental coupled solution while the
  // equilibrium lapse is retained as a regular gauge choice.
  auto &size = pmbp->pmb->mb_size;
  auto &adm = pmbp->padm->adm;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int isg = is - indcs.ng, ieg = indcs.ie + indcs.ng;
  int jsg = js - indcs.ng, jeg = indcs.je + indcs.ng;
  int ksg = ks - indcs.ng, keg = indcs.ke + indcs.ng;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb = pmbp->nmb_thispack;
  Real surface_radius = total_mass*profile.riso_over_m_surface;
  Real mass = total_mass;
  Real corrected_mass = total_mass + (apply_hamiltonian_correction
      ? constraint_correction.delta_adm_mass : 0.0);
  Real scpd_mass = total_mass + (apply_displacement
      ? displacement_metric.delta_adm_mass : 0.0);
  bool use_displacement_metric = apply_displacement;
  if (apply_displacement && !scpd_solve_metric) { scpd_mass = total_mass; }
  Real ctt_moment = solve_hamiltonian_constraint ? constraint_correction.moment :
                    (solve_momentum_constraint ? momentum_correction.moment : 0.0);
  Real cx = center[0], cy = center[1], cz = center[2];
  par_for("pgen relativistic cluster metric", DevExeSpace(), 0, nmb-1,
          ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real x1 = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max) - cx;
    Real x2 = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max) - cy;
    Real x3 = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max) - cz;
    Real riso = std::sqrt(x1*x1 + x2*x2 + x3*x3);
    Real conformal_a;
    Real alpha;
    Real ctt_a = 0.0;
    if (riso < surface_radius) {
      int lo = 0;
      int hi = nprof - 1;
      while (lo + 1 < hi) {
        int mid = (lo + hi)/2;
        if (radius_d(mid) <= riso) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      Real denom = radius_d(hi) - radius_d(lo);
      Real frac = (denom > 0.0) ? (riso - radius_d(lo))/denom : 0.0;
      conformal_a = conf_d(lo) + frac*(conf_d(hi) - conf_d(lo));
      alpha = lapse_d(lo) + frac*(lapse_d(hi) - lapse_d(lo));
      if (solve_momentum_constraint) {
        ctt_a = ctt_a_d(lo) + frac*(ctt_a_d(hi) - ctt_a_d(lo));
      }
    } else {
      Real q = 0.5*mass/riso;
      conformal_a = (1.0 + q)*(1.0 + q);
      alpha = (1.0 - q)/(1.0 + q);
      if (solve_momentum_constraint) {
        ctt_a = ctt_moment/(riso*riso*riso);
      }
    }
    if (apply_hamiltonian_correction) {
      if (riso <= ham_radius_d(nham-1)) {
        int hlo = 0;
        int hhi = nham - 1;
        while (hlo + 1 < hhi) {
          int mid = (hlo + hhi)/2;
          if (ham_radius_d(mid) <= riso) {
            hlo = mid;
          } else {
            hhi = mid;
          }
        }
        Real denom = ham_radius_d(hhi) - ham_radius_d(hlo);
        Real frac = (denom > 0.0) ? (riso - ham_radius_d(hlo))/denom : 0.0;
        conformal_a = ham_conf_d(hlo) + frac*(ham_conf_d(hhi) - ham_conf_d(hlo));
      } else {
        // This branch lies beyond the 10^6 M solver grid in normal use; the omitted
        // CTT r^-6 source is then negligible compared with the ADM 1/r term.
        Real q = 0.5*corrected_mass/riso;
        conformal_a = (1.0 + q)*(1.0 + q);
      }
    }
    if (use_displacement_metric) {
      if (riso <= scpd_radius_d(nscpd-1)) {
        int slo = 0;
        int shi = nscpd - 1;
        while (slo + 1 < shi) {
          int mid = (slo + shi)/2;
          if (scpd_radius_d(mid) <= riso) { slo = mid; } else { shi = mid; }
        }
        Real denom = scpd_radius_d(shi) - scpd_radius_d(slo);
        Real frac = (denom > 0.0) ? (riso - scpd_radius_d(slo))/denom : 0.0;
        conformal_a = scpd_conf_d(slo) + frac*(scpd_conf_d(shi) - scpd_conf_d(slo));
        alpha = scpd_lapse_d(slo) + frac*(scpd_lapse_d(shi) - scpd_lapse_d(slo));
      } else {
        Real q = 0.5*scpd_mass/riso;
        conformal_a = (1.0 + q)*(1.0 + q);
        alpha = (1.0 - q)/(1.0 + q);
      }
    }
    Real gamma_diag = conformal_a*conformal_a;
    Real nx = (riso > 0.0) ? x1/riso : 0.0;
    Real ny = (riso > 0.0) ? x2/riso : 0.0;
    Real nz = (riso > 0.0) ? x3/riso : 0.0;
    Real nhat[3] = {nx, ny, nz};
    adm.psi4(m,k,j,i) = gamma_diag;
    adm.alpha(m,k,j,i) = alpha;
    for (int a = 0; a < 3; ++a) {
      adm.beta_u(m,a,k,j,i) = 0.0;
      for (int b = a; b < 3; ++b) {
        adm.g_dd(m,a,b,k,j,i) = (a == b) ? gamma_diag : 0.0;
        if (solve_momentum_constraint) {
          Real tf = nhat[a]*nhat[b] - ((a == b) ? 1.0/3.0 : 0.0);
          // K_ij=psi^-2 (L W)_ij=A^-1 2 a(n_i n_j-delta_ij/3).
          adm.vK_dd(m,a,b,k,j,i) = 2.0*ctt_a*tf/conformal_a;
        } else {
          adm.vK_dd(m,a,b,k,j,i) = 0.0;
        }
      }
    }
  });
  Kokkos::fence();

  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin); break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin); break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin); break;
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "relativistic_cluster supports nghost=2,3,4."
                << std::endl;
      std::exit(EXIT_FAILURE);
  }
  pmbp->pz4c->Z4cToADM(pmbp);

  // Every rank reproduces the same seeded global draw and retains only particles owned
  // by its local MeshBlocks. Tags therefore remain rank-count and decomposition
  // invariant.
  particles::Particles *ppart = pmbp->ppart;
  Real particle_mass = total_mass*profile.rest_mass_over_m/
                       static_cast<Real>(npart_total);

  // Pass 1 (skipped entirely for eps_out = 0): decide, for every globally unique tag,
  // whether its radial momentum is to be reflected outward.  The decision is a pure
  // function of the tag, so it is identical on every rank and for every decomposition.
  std::vector<unsigned char> select;
  if (eps_out > 0.0) {
    select.assign(npart_total, 0);
    if (bias_mode.compare("bernoulli") == 0) {
      std::uint64_t salt = 0x9e3779b97f4a7c15ULL*static_cast<std::uint64_t>(bias_seed);
      for (int tag = 0; tag < npart_total; ++tag) {
        if (SplitMixUniform(static_cast<std::uint64_t>(tag) + salt) < eps_out) {
          select[tag] = 1;
        }
      }
    } else {
      // Replay the same seeded global draw, collect the inward movers, order them by
      // (isotropic radius, tag) and select a low-discrepancy eps_out-fraction of that
      // ordering.  The ordering is a global, decomposition-independent total order and
      // the selected sets are nested in eps_out.
      std::mt19937_64 scan_generator(static_cast<std::uint64_t>(seed));
      std::uniform_real_distribution<Real> scan_uniform(0.0, 1.0);
      std::vector<std::pair<Real, int>> inward;
      inward.reserve(static_cast<std::size_t>(npart_total)/2 + 16);
      for (int tag = 0; tag < npart_total; ++tag) {
        DrawnParticle d = DrawParticle(&scan_generator, &scan_uniform, profile,
                                       total_mass, center);
        if (RadialMomentum(d) < 0.0) { inward.emplace_back(d.riso, tag); }
      }
      std::sort(inward.begin(), inward.end());
      for (std::size_t j = 0; j < inward.size(); ++j) {
        if (RadicalInverse2(static_cast<std::uint64_t>(j)) < eps_out) {
          select[inward[j].second] = 1;
        }
      }
    }
  }

  // Pass 2: the actual realization.
  std::mt19937_64 generator(static_cast<std::uint64_t>(seed));
  std::uniform_real_distribution<Real> uniform(0.0, 1.0);
  PrtclStage stage;
  BiasStats bias;
  std::vector<double> dump;
  bool want_dump = (!t0_dump.empty() && global_variable::my_rank == 0);
  if (want_dump) { dump.reserve(static_cast<std::size_t>(npart_total)*8); }
  for (int tag = 0; tag < npart_total; ++tag) {
    DrawnParticle d = DrawParticle(&generator, &uniform, profile, total_mass, center);

    // t=0 bookkeeping of the UNPERTURBED sample
    Real udotn_base = RadialMomentum(d);
    Real mu_base = (d.umag > 0.0) ? udotn_base/d.umag : 0.0;
    Real speed = std::sqrt(std::max(static_cast<Real>(1.0) - d.ylocal,
                                    static_cast<Real>(0.0)));
    if (udotn_base < 0.0) { ++bias.n_inward_base; }
    bias.sum_mu_base += mu_base;
    bias.sum_mu2_base += mu_base*mu_base;
    bias.sum_vr_base += mu_base*speed;
    Real rx = d.px - center[0], ry = d.py - center[1], rz = d.pz - center[2];
    Real lx_base = ry*d.uz - rz*d.uy;
    Real ly_base = rz*d.ux - rx*d.uz;
    Real lz_base = rx*d.uy - ry*d.ux;
    bias.sum_abs_l_base += std::sqrt(lx_base*lx_base + ly_base*ly_base +
                                    lz_base*lz_base);

    if (apply_displacement) {
      Real r_before, r_after, dr_map_value, rel_dl, energy, v2_after;
      bool ok = ApplyDisplacement(&d, scpd_eps, displacement_map, profile,
                                  displacement_metric, total_mass, center,
                                  &r_before, &r_after, &dr_map_value, &rel_dl,
                                  &energy, &v2_after);
      if (!ok) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "cluster_scpd_eps=" << scpd_eps
                  << " produced an invalid displaced state for tag=" << tag << "."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // The Cartesian offsets used by the base bookkeeping refer to the DISPLACED
      // position from here on.
      rx = d.px - center[0]; ry = d.py - center[1]; rz = d.pz - center[2];
      bias.sum_scpd_r_before += r_before;
      bias.sum_scpd_r_after += r_after;
      Real ratio = (r_before > 0.0) ? r_after/r_before : dr_map_value;
      bias.min_scpd_ratio = std::min(bias.min_scpd_ratio, ratio);
      bias.max_scpd_ratio = std::max(bias.max_scpd_ratio, ratio);
      bias.min_scpd_dr = std::min(bias.min_scpd_dr, dr_map_value);
      bias.max_scpd_dr = std::max(bias.max_scpd_dr, dr_map_value);
      bias.max_scpd_rel_dl = std::max(bias.max_scpd_rel_dl, rel_dl);
      bias.sum_scpd_energy += energy;
      bias.sum_scpd_energy2 += energy*energy;
      bias.min_scpd_energy = std::min(bias.min_scpd_energy, energy);
      bias.max_scpd_energy = std::max(bias.max_scpd_energy, energy);
      bias.max_scpd_v2 = std::max(bias.max_scpd_v2, v2_after);
      Real udotn_scpd = RadialMomentum(d);
      Real w_scpd = std::sqrt(1.0 + d.umag*d.umag/(d.conf_a*d.conf_a));
      Real vr = udotn_scpd/(d.conf_a*w_scpd);
      Real vt2 = std::max(d.umag*d.umag - udotn_scpd*udotn_scpd,
                          static_cast<Real>(0.0))/(d.conf_a*d.conf_a*w_scpd*w_scpd);
      bias.sum_scpd_vr2 += vr*vr;
      bias.sum_scpd_vt2 += vt2;
    }

    // When the incremental Hamiltonian solve changes A=psi^2, rescale u_i by
    // A_new/A_0 so the original local physical velocity is preserved before the mode.
    // The mode profile itself remains tied to the equilibrium areal radius as prescribed.
    Real mode_conformal_a = d.conf_a;
    if (apply_hamiltonian_correction) {
      Real corrected_conformal_a = InterpolateRadius(
          constraint_correction.radius, constraint_correction.conformal_a, d.riso);
      if (!(corrected_conformal_a > 0.0)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Invalid corrected conformal factor for tag="
                  << tag << "." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      Real metric_scale = corrected_conformal_a/d.conf_a;
      d.ux *= metric_scale;
      d.uy *= metric_scale;
      d.uz *= metric_scale;
      d.umag *= metric_scale;
      d.conf_a = corrected_conformal_a;
    }

    Real umag_before = 0.0;
    if (lambda_tan > 0.0 || (!select.empty() && select[tag]) ||
        radial_mode_u != 0.0) {
      umag_before = std::sqrt(d.ux*d.ux + d.uy*d.uy + d.uz*d.uz);
    }
    if (lambda_tan > 0.0) {
      ReorientTangential(&d, lambda_tan);
      ++bias.n_reoriented;
    }
    bool reflected = false;
    if (!select.empty() && select[tag]) {
      ++bias.n_selected;
      reflected = ReflectRadialOutward(&d);
      if (reflected) {
        ++bias.n_reflected;
      }
    }
    if (radial_mode_u != 0.0) {
      Real dv, vr_before, vr_after, w_before, w_after, v2_after;
      Real abs_delta_vtan, normalization_residual;
      bool mode_ok = AddCoherentRadialMode(
          &d, radial_mode_u, total_mass*profile.r_over_m, mode_conformal_a,
          &dv, &vr_before,
          &vr_after, &w_before, &w_after, &v2_after, &abs_delta_vtan,
          &normalization_residual);
      if (!mode_ok) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "cluster_radial_mode_u=" << radial_mode_u
                  << " produced a non-timelike or invalid velocity for tag=" << tag
                  << "." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      bias.sum_mode_dv += dv;
      bias.sum_mode_vr_before += vr_before;
      bias.sum_mode_vr_after += vr_after;
      bias.sum_mode_w_before += w_before;
      bias.sum_mode_w_after += w_after;
      bias.min_mode_dv = std::min(bias.min_mode_dv, dv);
      bias.max_mode_dv = std::max(bias.max_mode_dv, dv);
      bias.max_mode_v2 = std::max(bias.max_mode_v2, v2_after);
      bias.max_abs_delta_vtan = std::max(bias.max_abs_delta_vtan, abs_delta_vtan);
      bias.max_mode_normalization_residual = std::max(
          bias.max_mode_normalization_residual, normalization_residual);
    }
    if (umag_before > 0.0) {
      Real umag_after = std::sqrt(d.ux*d.ux + d.uy*d.uy + d.uz*d.uz);
      bias.max_rel_dumag = std::max(bias.max_rel_dumag,
                                    std::abs(umag_after - umag_before)/umag_before);
    }

    // t=0 bookkeeping of the PERTURBED sample.  The orthonormal-frame momentum per unit
    // rest mass is p^(hat i)/m0 = u_i/A, so the physical sums below carry the 1/A factor.
    Real udotn = RadialMomentum(d);
    Real mu = (d.umag > 0.0) ? udotn/d.umag : 0.0;
    bias.sum_mu += mu;
    bias.sum_mu2 += mu*mu;
    Real speed_after = speed;
    if (radial_mode_u != 0.0 || apply_displacement) {
      Real phat2 = d.umag*d.umag/(d.conf_a*d.conf_a);
      speed_after = std::sqrt(phat2/(1.0 + phat2));
    }
    bias.sum_vr += mu*speed_after;
    Real w = particle_mass/d.conf_a;
    bias.sum_pr += w*udotn;
    bias.p_tot[0] += w*d.ux;
    bias.p_tot[1] += w*d.uy;
    bias.p_tot[2] += w*d.uz;
    Real lx = ry*d.uz - rz*d.uy;
    Real ly = rz*d.ux - rx*d.uz;
    Real lz = rx*d.uy - ry*d.ux;
    bias.sum_abs_l += std::sqrt(lx*lx + ly*ly + lz*lz);
    bias.j_tot[0] += particle_mass*lx;
    bias.j_tot[1] += particle_mass*ly;
    bias.j_tot[2] += particle_mass*lz;

    if (want_dump) {
      dump.push_back(static_cast<double>(tag));
      dump.push_back(reflected ? 1.0 : 0.0);
      dump.push_back(d.px);
      dump.push_back(d.py);
      dump.push_back(d.pz);
      dump.push_back(d.ux);
      dump.push_back(d.uy);
      dump.push_back(d.uz);
    }

    int m = ppart->FindContainingMeshBlock(d.px, d.py, d.pz);
    if (m >= 0) {
      stage.Add(d.px, d.py, d.pz, d.ux, d.uy, d.uz, particle_mass,
                pmbp->gids + m, tag);
    }
  }

  int npart = static_cast<int>(stage.x.size());
  Kokkos::realloc(ppart->prtcl_rdata, ppart->nrdata, npart);
  Kokkos::realloc(ppart->prtcl_idata, ppart->nidata, npart);
  auto hr = Kokkos::create_mirror_view(ppart->prtcl_rdata);
  auto hi = Kokkos::create_mirror_view(ppart->prtcl_idata);
  for (int p = 0; p < npart; ++p) {
    hi(PGID,p) = stage.gid[p];
    hi(PTAG,p) = stage.tag[p];
    hr(IPM,p) = stage.mass[p];
    hr(IPEN,p) = 0.0;
    hr(IPX,p) = stage.x[p];
    hr(IPVX,p) = stage.ux[p];
    hr(IPY,p) = stage.y[p];
    hr(IPVY,p) = stage.uy[p];
    hr(IPZ,p) = stage.z[p];
    hr(IPVZ,p) = stage.uz[p];
  }
  Kokkos::deep_copy(ppart->prtcl_rdata, hr);
  Kokkos::deep_copy(ppart->prtcl_idata, hi);
  ppart->nprtcl_thispack = npart;

  pmy_mesh_->nprtcl_thisrank = npart;
  pmy_mesh_->nprtcl_eachrank[global_variable::my_rank] = npart;
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&npart, 1, MPI_INT, pmy_mesh_->nprtcl_eachrank, 1, MPI_INT,
                MPI_COMM_WORLD);
#endif
  pmy_mesh_->nprtcl_total = 0;
  for (int rank = 0; rank < global_variable::nranks; ++rank) {
    pmy_mesh_->nprtcl_total += pmy_mesh_->nprtcl_eachrank[rank];
  }
  if (pmy_mesh_->nprtcl_total != npart_total) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Placed " << pmy_mesh_->nprtcl_total << " of "
              << npart_total << " cluster particles; enlarge or recenter the mesh."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  SeedSnapshots();

  if (global_variable::my_rank == 0) {
    std::cout << "Relativistic cluster initialized: N=" << npart_total
              << ", yc=" << yc << ", R/M=" << profile.r_over_m
              << ", Riso/M=" << profile.riso_over_m_surface
              << ", M0/M=" << profile.rest_mass_over_m
              << ", m0=" << particle_mass << ", seed=" << seed << std::endl;
    std::cout << "Cluster CTT momentum solve: enabled="
              << (solve_momentum_constraint ? 1 : 0);
    if (solve_hamiltonian_constraint) {
      std::cout << ", quadrature_nmu=64, moment=" << constraint_correction.moment
                << ", max|F|=" << constraint_correction.max_abs_source
                << ", max|a|=" << constraint_correction.max_abs_a
                << ", max|<u_hat_r>|="
                << constraint_correction.max_abs_mean_uhat_r;
    } else if (solve_momentum_constraint) {
      std::cout << ", quadrature_nmu=64, moment=" << momentum_correction.moment
                << ", max|F|=" << momentum_correction.max_abs_source
                << ", max|a|=" << momentum_correction.max_abs_a
                << ", max|<u_hat_r>|="
                << momentum_correction.max_abs_mean_uhat_r;
    }
    std::cout << std::endl;
    std::cout << "Cluster incremental Hamiltonian solve: enabled="
              << (solve_hamiltonian_constraint ? 1 : 0);
    if (solve_hamiltonian_constraint) {
      std::cout << ", iterations=" << constraint_correction.iterations
                << ", final_relative_update="
                << constraint_correction.final_relative_update
                << ", delta_ADM_mass=" << constraint_correction.delta_adm_mass
                << ", max|delta_psi|=" << constraint_correction.max_abs_delta_psi
                << ", max|delta_psi/psi0|="
                << constraint_correction.max_rel_delta_psi
                << ", max|delta_Q|=" << constraint_correction.max_abs_delta_q;
    }
    std::cout << std::endl;
    std::cout << "Cluster SCPD displacement: enabled=" << (apply_displacement ? 1 : 0)
              << ", eps=" << scpd_eps
              << ", solve_metric=" << (scpd_solve_metric ? 1 : 0);
    if (apply_displacement) {
      Real inv_n_scpd = 1.0/static_cast<Real>(npart_total);
      Real mean_e = bias.sum_scpd_energy*inv_n_scpd;
      Real var_e = std::max(bias.sum_scpd_energy2*inv_n_scpd - mean_e*mean_e,
                            static_cast<Real>(0.0));
      Real mean_vr2 = bias.sum_scpd_vr2*inv_n_scpd;
      Real mean_vt2 = bias.sum_scpd_vt2*inv_n_scpd;
      std::cout << ", target_yc=" << displacement_map.target_yc
                << ", B_start=" << displacement_map.start_binding
                << ", B_target=" << displacement_map.target_binding
                << ", M0_scale=" << displacement_map.rest_mass_scale
                << ", map_dR_range=[" << displacement_map.min_dr << ","
                << displacement_map.max_dr << "]"
                << ", sample_dR_range=[" << bias.min_scpd_dr << ","
                << bias.max_scpd_dr << "]"
                << ", sample_R/rbar_range=[" << bias.min_scpd_ratio << ","
                << bias.max_scpd_ratio << "]"
                << ", <rbar> " << bias.sum_scpd_r_before*inv_n_scpd << " -> "
                << bias.sum_scpd_r_after*inv_n_scpd
                << ", max|dL/L|=" << bias.max_scpd_rel_dl
                << ", <E>=" << mean_e << ", sigma_E=" << std::sqrt(var_e)
                << ", E_range=[" << bias.min_scpd_energy << ","
                << bias.max_scpd_energy << "]"
                << ", max_v2=" << bias.max_scpd_v2
                << ", beta_v=" << (mean_vr2 > 0.0 ? 1.0 - mean_vt2/(2.0*mean_vr2) : 0.0)
                << ", psi_its=" << displacement_metric.psi_iterations
                << ", psi_resid=" << displacement_metric.psi_residual
                << ", lapse_its=" << displacement_metric.lapse_iterations
                << ", lapse_resid=" << displacement_metric.lapse_residual
                << ", delta_ADM_mass=" << displacement_metric.delta_adm_mass
                << ", max|delta_psi/psi0|=" << displacement_metric.max_rel_delta_psi
                << ", max|delta_alpha/alpha0|="
                << displacement_metric.max_rel_delta_alpha
                << ", alpha_c=" << displacement_metric.central_alpha
                << ", A_c=" << displacement_metric.central_conformal_a
                << ", surface_rbar=" << displacement_metric.surface_radius;
    }
    std::cout << std::endl;

    // t=0 bias bookkeeping.  These sums run over ALL tags on every rank, so they are
    // global by construction and need no MPI reduction.
    Real inv_n = 1.0/static_cast<Real>(npart_total);
    Real f_out_base = 1.0 - static_cast<Real>(bias.n_inward_base)*inv_n;
    Real f_out = f_out_base + static_cast<Real>(bias.n_reflected)*inv_n;
    Real frac_reflected = (bias.n_inward_base > 0)
        ? static_cast<Real>(bias.n_reflected)/static_cast<Real>(bias.n_inward_base)
        : 0.0;
    Real mean_mu2_base = bias.sum_mu2_base*inv_n;
    Real mean_mu2 = bias.sum_mu2*inv_n;
    Real beta_base = (mean_mu2_base > 0.0)
        ? 1.0 - (1.0 - mean_mu2_base)/(2.0*mean_mu2_base) : 0.0;
    Real beta = (mean_mu2 > 0.0)
        ? 1.0 - (1.0 - mean_mu2)/(2.0*mean_mu2) : 0.0;
    if (radial_mode_u == 0.0) {
      std::cout << "Cluster tangential bias: lambda_tan=" << lambda_tan
                << ", n_reoriented=" << bias.n_reoriented
                << ", <|L|> " << bias.sum_abs_l_base*inv_n << " -> "
                << bias.sum_abs_l*inv_n
                << ", ratio=" << bias.sum_abs_l/bias.sum_abs_l_base << std::endl;
      std::cout << "Cluster tangential bias: <mu^2> " << mean_mu2_base << " -> "
                << mean_mu2 << ", beta " << beta_base << " -> " << beta << std::endl;
    } else {
      std::cout << "Cluster perturbation summary: lambda_tan=" << lambda_tan
                << ", radial_mode_u=" << radial_mode_u
                << ", <|L|> " << bias.sum_abs_l_base*inv_n << " -> "
                << bias.sum_abs_l*inv_n
                << ", ratio=" << bias.sum_abs_l/bias.sum_abs_l_base << std::endl;
      std::cout << "Cluster perturbation summary: <mu^2> " << mean_mu2_base << " -> "
                << mean_mu2 << ", directional beta " << beta_base << " -> " << beta
                << std::endl;
    }
    std::cout << "Cluster outward bias: eps_out=" << eps_out
              << ", mode=" << bias_mode
              << ", n_inward_base=" << bias.n_inward_base
              << ", n_selected=" << bias.n_selected
              << ", n_reflected=" << bias.n_reflected
              << ", reflected/inward=" << frac_reflected
              << " (target " << eps_out << ")" << std::endl;
    if (radial_mode_u == 0.0) {
      std::cout << "Cluster outward bias: f_outward " << f_out_base << " -> " << f_out
                << " (target " << 0.5*(1.0 + eps_out) << ")"
                << ", <mu> " << bias.sum_mu_base*inv_n << " -> " << bias.sum_mu*inv_n
                << " (target " << 0.5*eps_out << ")"
                << ", <v_r> " << bias.sum_vr_base*inv_n << " -> " << bias.sum_vr*inv_n
                << std::endl;
    } else {
      std::cout << "Cluster perturbation summary: f_outward " << f_out_base << " -> "
                << f_out << ", <mu> " << bias.sum_mu_base*inv_n << " -> "
                << bias.sum_mu*inv_n << ", <v_r> " << bias.sum_vr_base*inv_n << " -> "
                << bias.sum_vr*inv_n << std::endl;
    }
    std::cout << ((radial_mode_u == 0.0) ? "Cluster outward bias: P_r="
                                         : "Cluster perturbation summary: P_r=")
              << bias.sum_pr
              << ", P=(" << bias.p_tot[0] << "," << bias.p_tot[1] << ","
              << bias.p_tot[2] << ")"
              << ", J=(" << bias.j_tot[0] << "," << bias.j_tot[1] << ","
              << bias.j_tot[2] << ")"
              << ", max|d|u||/|u|=" << bias.max_rel_dumag << std::endl;
    if (radial_mode_u != 0.0) {
      std::cout << "Cluster coherent radial mode: U=" << radial_mode_u
                << ", radius=areal, <delta v_r>=" << bias.sum_mode_dv*inv_n
                << ", range(delta v_r)=[" << bias.min_mode_dv << ","
                << bias.max_mode_dv << "]"
                << ", <v_r> " << bias.sum_mode_vr_before*inv_n << " -> "
                << bias.sum_mode_vr_after*inv_n
                << ", <W> " << bias.sum_mode_w_before*inv_n << " -> "
                << bias.sum_mode_w_after*inv_n << std::endl;
      std::cout << "Cluster coherent radial mode: max(v^2)=" << bias.max_mode_v2
                << ", max|delta v_tan|=" << bias.max_abs_delta_vtan
                << ", max normalization residual="
                << bias.max_mode_normalization_residual << std::endl;
    }
  }

  if (want_dump) {
    // Self-describing dump of the complete GLOBAL t=0 sample in double precision: one
    // ASCII header line, then npart_total records of 8 little-endian doubles
    // (tag, reflected, x, y, z, u_x, u_y, u_z).  Written by rank 0 only; every rank
    // holds the identical global draw, so this is the full realization, not a subset.
    std::ofstream fh(t0_dump.c_str(), std::ios::out | std::ios::binary);
    if (!fh) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Could not open cluster_t0_dump file '" << t0_dump
                << "'." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    fh.precision(17);
    fh << "STMIGP01 npart=" << npart_total << " nfield=8"
       << " fields=tag,reflected,x,y,z,ux,uy,uz"
       << " yc=" << yc << " lambda_tan=" << lambda_tan
       << " eps_out=" << eps_out << " bias_mode=" << bias_mode
       << " radial_mode_u=" << radial_mode_u << " radial_mode_radius=areal"
       << " solve_momentum_constraint=" << (solve_momentum_constraint ? 1 : 0)
       << " solve_hamiltonian_constraint=" << (solve_hamiltonian_constraint ? 1 : 0)
       << " delta_adm_mass=" << (solve_hamiltonian_constraint
                                      ? constraint_correction.delta_adm_mass : 0.0)
       << " bias_seed=" << bias_seed << " seed=" << seed
       << " mass=" << total_mass << " m0=" << particle_mass
       << " R_over_M=" << profile.r_over_m
       << " Riso_over_M=" << profile.riso_over_m_surface
       << " M0_over_M=" << profile.rest_mass_over_m
       << " center=" << center[0] << "," << center[1] << "," << center[2]
       << " profile_dx=" << profile_dx << "\n";
    fh.write(reinterpret_cast<const char*>(dump.data()),
             static_cast<std::streamsize>(dump.size()*sizeof(double)));
    fh.close();
    std::cout << "Cluster outward bias: wrote t=0 sample to '" << t0_dump << "' ("
              << npart_total << " records)" << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \brief apply the standard Z4c AMR criterion selected by <z4c_amr>.
void STRefinementCondition(MeshBlockPack *pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}
