//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gi_cluster.cpp
//! \brief Constraint-satisfying Einstein-Vlasov initial data for local gravitational
//! collapse and hierarchical mergers: a static spherical envelope plus cold, localized
//! Gaussian overdensities ("clumps").
//!
//! Reference: "Constraint-Satisfying Einstein-Vlasov Initial Data for Local Collapse and
//! Hierarchical Black-Hole Mergers" (2026-08-11). Equation numbers below refer to it.
//!
//! Construction summary:
//!  - Sec. II: isotropic equilibrium f0(E) = A (Ecut - E)_+ with E = e^mu sqrt(1+v^2),
//!    where v^i is the orthonormal specific momentum and W = sqrt(1+|v|^2). The TOV-like
//!    system Eq. (10) with closed-form moments Eq. (12) is integrated in the scaled
//!    variables y = ln(Ecut) - mu, B = A*Ecut. An exact scaling freedom
//!    (r -> lam r, m -> lam m, B -> B/lam^2 at fixed y) reduces the two-parameter
//!    shooting to a 1D solve of the compactness 2 M0/R0 in y_c alone with B = 1.
//!  - Sec. II B: transform to isotropic radius R via dR/R = e^lambda dr/r, integrated as
//!    the regular h(r) = int (e^lambda - 1)/r dr with the boundary value Eq. (17);
//!    psi0 = sqrt(r/R), alpha0 = e^mu, q0 = psi0^5 rho0.
//!  - Sec. III: conformal sources q = q0 + sum_a Ma Ga, rho = psi^-5 q, j^i = 0, with the
//!    closed-form solution Eq. (26): psi = psi0(R) + sum_a (Ma/2 r_a) erf(r_a/sqrt2 s_a),
//!    time symmetric (K_ij = 0), conformally flat. Momentum constraint is identical zero;
//!    the Hamiltonian constraint is the linear Poisson equation Eq. (25) solved exactly.
//!  - Sec. IV: PIC realization. Envelope f_env = (psi0/psi)^5 f0(R,v) sampled with rest
//!    weight mu_env = N_env^-1 int psi psi0^5 n0 d3X (n0 = int f0 d3v); positions from
//!    the coordinate rest-mass measure psi psi0^5 n0 (rejection on psi/psi0), momenta
//!    from f0(R,.)/n0(R). Clump a: positions from P_a = psi Ma Ga/I_a (Eq. 36), momenta
//!    from the Maxwellian h_a (Eq. 33), rest weight mu_a = I_a/(N_a <W>_a) (Eq. 37).
//!    Thermal momenta are inserted in antithetic rest-frame pairs.  With zero bulk boost
//!    the deposited current cancels exactly pair by pair.  An optional clump y-boost applies
//!    the same proper Lorentz boost to both members and rescales the rest weight by the
//!    changed mean Lorentz factor, preserving the target energy density while intentionally
//!    introducing a local momentum current.  Because K_ij remains zero, such boosted data
//!    are explicitly constraint-imperfect and require a separate momentum-constraint gate.
//!  - Sec. V: gauge init alpha = alpha0 (psi0/psi)^2, beta^i = 0 (Eq. 39).
//!
//! AthenaK conventions (audited against this tree):
//!  - particle velocity slots store the covariant u_i; for gamma_ij = psi^4 delta_ij the
//!    conversion from the paper's orthonormal v^i is u_i = psi^2 v^i.
//!  - the Tmunu deposit is E += m W, S_i += m u_i, S_ij += (m/W) u_i u_j per particle,
//!    CIC-weighted and divided by sqrt(det gamma) dV, so the continuum limit of the
//!    deposited E is exactly the paper's rho.
//!
//! Required runtime configuration:
//!   - 3D mesh covering the whole envelope (isotropic surface radius printed in banner)
//!   - <z4c> block
//!   - <particles> init=pgen, particle_type=dust, pusher=gr_boris, feedback=true
//!
//! Mesh refinement. Static refinement (<mesh_refinement> refinement = static plus
//! <refined_region*> blocks) needs nothing from the pgen: the ADM fill loops over
//! nmb_thispack and takes every coordinate from that block's own mb_size, all the
//! normalization integrals are 1D radial quadratures, and particle placement goes
//! through the level-agnostic FindContainingMeshBlock. For ADAPTIVE refinement the
//! Mesh needs a criterion, and none of the built-in <amr_criterion> variables exist in
//! a Z4c + particle run (they are all hydro/MHD/radiation fields). RefinementCondition
//! below therefore forwards to the shared Z4c refinement module, exactly as the vacuum
//! Z4c pgens do, so that <z4c_amr> method = tracker | chi | dchi | loehner and the
//! radius_<n>_rad / radius_<n>_reflevel shells drive the hierarchy. Enable it with
//!     <mesh_refinement>  refinement = adaptive
//!     <amr_criterion1>   method = user
//!
//! <problem> parameters:
//!   gi_m0               envelope gravitational mass M0 (default 0.61)
//!   gi_r0               envelope areal surface radius R0 (default 30.0)
//!   gi_n_env            envelope particle count, must be even (default 100000)
//!   gi_n_clump          total clump particles, split among clumps prop. to Ma
//!                       in antithetic pairs (default 40000)
//!   gi_seed             random seed (default 1)
//!   gi_profile_dr       ODE step in scaled (B=1) units (default 1e-3)
//!   gi_nclumps          number of clumps (default 4)
//!   gi_clumpN_mass, gi_clumpN_x1/x2/x3, gi_clumpN_sigma, gi_clumpN_s  (N = 1..nclumps;
//!                       defaults for N<=4 from Table I of the paper)
//!   gi_clumpN_bulk_vy orthonormal bulk velocity beta_y; |beta_y|<1 (default 0)
//!   gi_clump_mass_scale multiplier applied to every Ma; 0 disables all clumps and
//!                       recovers the exact static background (default 1.0)
//!   gi_dump_profile     if non-empty, rank 0 writes the radial profile table to this
//!                       file (default empty)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
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

namespace {

//----------------------------------------------------------------------------------------
// Closed-form moments of f0 = A (Ecut - E)_+, Eq. (12), in units of 4 pi B (B = A Ecut),
// with v_y = sqrt(e^{2y}-1) and u_y = asinh(v_y). N0Bar is the rest-mass density
// n0 = int f0 d3v (no W factor), derived the same way and verified against quadrature.

Real Rho0Bar(Real y) {
  if (y <= 0.0) return 0.0;
  Real vy = std::sqrt(std::expm1(2.0*y));
  Real uy = std::asinh(vy);
  Real ey = std::exp(y);
  return (vy*ey*(2.0*vy*vy + 1.0) - uy)/8.0
         - std::exp(-y)*(vy*vy*vy/3.0 + vy*vy*vy*vy*vy/5.0);
}

Real P0Bar(Real y) {
  if (y <= 0.0) return 0.0;
  Real vy = std::sqrt(std::expm1(2.0*y));
  Real uy = std::asinh(vy);
  Real ey = std::exp(y);
  return ((vy*ey*(2.0*vy*vy - 3.0) + 3.0*uy)/8.0
          - std::exp(-y)*vy*vy*vy*vy*vy/5.0)/3.0;
}

Real N0Bar(Real y) {
  if (y <= 0.0) return 0.0;
  Real vy = std::sqrt(std::expm1(2.0*y));
  Real uy = std::asinh(vy);
  return vy*vy*vy/3.0 - vy*std::exp(2.0*y)/4.0 + vy/8.0 + uy*std::exp(-y)/8.0;
}

//----------------------------------------------------------------------------------------
// Scaled (B=1) integration of Eq. (10) plus the regular isotropic-transform variable
// h(r) = int_0^r (e^lambda - 1)/r' dr', stopped at the support surface y = 0.

struct ScaledSolution {
  std::vector<Real> r, m, y, h;
  Real rs, ms, hs;  // surface values
};

struct GIRhs {
  Real dm, dy, dh;
};

GIRhs EvalRHS(Real r, Real m, Real y) {
  Real rho = 4.0*M_PI*Rho0Bar(y);
  Real p = 4.0*M_PI*P0Bar(y);
  GIRhs out;
  out.dm = 4.0*M_PI*r*r*rho;
  out.dy = -(m + 4.0*M_PI*r*r*r*p)/(r*(r - 2.0*m));
  Real compact = 1.0 - 2.0*m/r;
  out.dh = (1.0/std::sqrt(std::max(compact, static_cast<Real>(1.0e-30))) - 1.0)/r;
  return out;
}

ScaledSolution IntegrateScaled(Real yc, Real dr, bool store) {
  constexpr int max_steps = 10000000;
  Real rho_c = 4.0*M_PI*Rho0Bar(yc);
  Real p_c = 4.0*M_PI*P0Bar(yc);

  ScaledSolution sol;
  // Regular center series through the first grid point r = dr:
  //   m ~ (4 pi/3) rho_c r^3, y ~ yc - 2 pi (rho_c/3 + p_c) r^2, h ~ (2 pi/3) rho_c r^2.
  Real r = dr;
  Real m = (4.0*M_PI/3.0)*rho_c*r*r*r;
  Real y = yc - 2.0*M_PI*(rho_c/3.0 + p_c)*r*r;
  Real h = (2.0*M_PI/3.0)*rho_c*r*r;
  if (store) {
    sol.r = {0.0, r};
    sol.m = {0.0, m};
    sol.y = {yc, y};
    sol.h = {0.0, h};
  }

  bool found = false;
  for (int n = 0; n < max_steps; ++n) {
    GIRhs k1 = EvalRHS(r, m, y);
    GIRhs k2 = EvalRHS(r + 0.5*dr, m + 0.5*dr*k1.dm, y + 0.5*dr*k1.dy);
    GIRhs k3 = EvalRHS(r + 0.5*dr, m + 0.5*dr*k2.dm, y + 0.5*dr*k2.dy);
    GIRhs k4 = EvalRHS(r + dr, m + dr*k3.dm, y + dr*k3.dy);
    Real mn = m + dr*(k1.dm + 2.0*k2.dm + 2.0*k3.dm + k4.dm)/6.0;
    Real yn = y + dr*(k1.dy + 2.0*k2.dy + 2.0*k3.dy + k4.dy)/6.0;
    Real hn = h + dr*(k1.dh + 2.0*k2.dh + 2.0*k3.dh + k4.dh)/6.0;
    Real rn = r + dr;
    if (!std::isfinite(mn) || !std::isfinite(yn) || !std::isfinite(hn)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "gi_cluster equilibrium ODE became non-finite."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (yn <= 0.0) {
      Real frac = y/(y - yn);
      sol.rs = r + frac*dr;
      sol.ms = m + frac*(mn - m);
      sol.hs = h + frac*(hn - h);
      if (store) {
        sol.r.push_back(sol.rs);
        sol.m.push_back(sol.ms);
        sol.y.push_back(0.0);
        sol.h.push_back(sol.hs);
      }
      found = true;
      break;
    }
    r = rn; m = mn; y = yn; h = hn;
    if (store) {
      sol.r.push_back(r);
      sol.m.push_back(m);
      sol.y.push_back(y);
      sol.h.push_back(h);
    }
  }
  if (!found) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "gi_cluster equilibrium ODE found no surface (yc="
              << yc << "); check gi_profile_dr." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return sol;
}

Real Compactness(Real yc, Real dr) {
  ScaledSolution s = IntegrateScaled(yc, dr, false);
  return 2.0*s.ms/s.rs;
}

//----------------------------------------------------------------------------------------
// Physical background profile in isotropic coordinates.

struct GIProfile {
  std::vector<Real> riso;    // isotropic radius R
  std::vector<Real> rareal;  // areal radius r(R)
  std::vector<Real> yv;      // y(R)
  std::vector<Real> psi0;    // background conformal factor psi0(R)
  std::vector<Real> alpha0;  // background lapse alpha0(R) = e^mu
  std::vector<Real> rho0;    // energy density (Eq. 12a)
  std::vector<Real> n0;      // rest-mass density
  std::vector<Real> q0;      // conformal source psi0^5 rho0 (Eq. 18)
  std::vector<Real> cdf;     // envelope radial rest-mass CDF, weight psi0^6 n0 R^2
  Real yc, ecut, aval, bval;
  Real m0, r0, riso_s;
  Real rest_mass;            // int psi0^6 n0 d3X (unperturbed envelope rest mass)
  Real q0_integral;          // int 4 pi R^2 q0 dR (should equal M0 by Eq. 20 + Gauss)
};

GIProfile BuildProfile(Real m0, Real r0, Real dr) {
  Real target_c = 2.0*m0/r0;
  // 1D bisection in yc on the scaled compactness. The bracket [0.005, 0.5] covers all
  // sensible targets for this family (compactness ~ 0.74 yc at small yc).
  Real lo = 0.005, hi = 0.5;
  Real flo = Compactness(lo, dr) - target_c;
  const Real fhi = Compactness(hi, dr) - target_c;
  if (flo*fhi > 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "gi_cluster shooting bracket does not straddle 2M0/R0="
              << target_c << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (int it = 0; it < 100; ++it) {
    Real mid = 0.5*(lo + hi);
    Real fmid = Compactness(mid, dr) - target_c;
    if (flo*fmid < 0.0) {
      hi = mid;
    } else {
      lo = mid; flo = fmid;
    }
    if (hi - lo < 1.0e-15) break;
  }
  Real yc = 0.5*(lo + hi);
  ScaledSolution sol = IntegrateScaled(yc, dr, true);

  // Rescale: r -> lam r, m -> lam m, B = 1/lam^2 (exact symmetry of Eqs. 10+12).
  Real lam = r0/sol.rs;
  Real ecut = std::sqrt(1.0 - target_c);            // Eq. (13)
  Real bval = 1.0/(lam*lam);
  Real riso_s = 0.5*(r0 - m0 + std::sqrt(r0*(r0 - 2.0*m0)));  // Eq. (17)
  // R(r) = C r e^{h(r)} with C fixed by the boundary value.
  Real cfac = riso_s/(r0*std::exp(sol.hs));

  GIProfile p;
  p.yc = yc; p.ecut = ecut; p.bval = bval; p.aval = bval/ecut;
  p.m0 = m0; p.r0 = r0; p.riso_s = riso_s;
  std::size_t n = sol.r.size();
  p.riso.resize(n); p.rareal.resize(n); p.yv.resize(n); p.psi0.resize(n);
  p.alpha0.resize(n); p.rho0.resize(n); p.n0.resize(n); p.q0.resize(n); p.cdf.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    Real r = lam*sol.r[i];
    Real y = sol.y[i];
    p.rareal[i] = r;
    p.yv[i] = y;
    p.riso[i] = cfac*r*std::exp(sol.h[i]);
    p.psi0[i] = (i == 0) ? 1.0/std::sqrt(cfac) : std::sqrt(r/p.riso[i]);
    p.alpha0[i] = ecut*std::exp(-y);
    p.rho0[i] = 4.0*M_PI*bval*Rho0Bar(y);
    p.n0[i] = 4.0*M_PI*bval*N0Bar(y);
    p.q0[i] = std::pow(p.psi0[i], 5)*p.rho0[i];
  }
  // Force the tabulated surface to the exact boundary value so exterior matching is
  // continuous to roundoff.
  p.riso.back() = riso_s;
  p.psi0.back() = 1.0 + 0.5*m0/riso_s;

  // Envelope radial rest-mass CDF (coordinate measure): w = psi0^6 n0 R^2, and the two
  // scalar integrals used for weights and sanity checks.
  p.cdf[0] = 0.0;
  Real acc = 0.0, accq = 0.0;
  for (std::size_t i = 1; i < n; ++i) {
    Real wm = std::pow(p.psi0[i-1], 6)*p.n0[i-1]*p.riso[i-1]*p.riso[i-1];
    Real wp = std::pow(p.psi0[i], 6)*p.n0[i]*p.riso[i]*p.riso[i];
    Real dx = p.riso[i] - p.riso[i-1];
    acc += 0.5*(wm + wp)*dx;
    accq += 0.5*(p.q0[i-1]*p.riso[i-1]*p.riso[i-1] + p.q0[i]*p.riso[i]*p.riso[i])*dx;
    p.cdf[i] = acc;
  }
  p.rest_mass = 4.0*M_PI*acc;
  p.q0_integral = 4.0*M_PI*accq;
  if (!(acc > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "gi_cluster envelope rest-mass integral is zero."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (Real &v : p.cdf) { v /= acc; }
  p.cdf.back() = 1.0;
  return p;
}

//----------------------------------------------------------------------------------------
// Host-side table interpolation (monotone abscissa) and background field evaluation with
// the exact Schwarzschild exterior.

Real InterpTable(const std::vector<Real> &x, const std::vector<Real> &f, Real xq) {
  if (xq <= x.front()) return f.front();
  if (xq >= x.back()) return f.back();
  auto it = std::upper_bound(x.begin(), x.end(), xq);
  std::size_t hi = static_cast<std::size_t>(it - x.begin());
  std::size_t lo = hi - 1;
  Real denom = x[hi] - x[lo];
  Real frac = (denom > 0.0) ? (xq - x[lo])/denom : 0.0;
  return f[lo] + frac*(f[hi] - f[lo]);
}

Real Psi0OfR(const GIProfile &p, Real riso) {
  if (riso >= p.riso_s) return 1.0 + 0.5*p.m0/riso;
  return InterpTable(p.riso, p.psi0, riso);
}

Real Alpha0OfR(const GIProfile &p, Real riso) {
  if (riso >= p.riso_s) {
    Real q = 0.5*p.m0/riso;
    return (1.0 - q)/(1.0 + q);
  }
  return InterpTable(p.riso, p.alpha0, riso);
}

//----------------------------------------------------------------------------------------
// Clump potential term of Eq. (26): (M/2r) erf(r/(sqrt2 sigma)), with the regular center
// limit Eq. (27) handled through the Taylor series of erf(z)/z. Callable on host and
// device (Kokkos::erf).

KOKKOS_INLINE_FUNCTION
Real ClumpPsiTerm(Real mass, Real sigma, Real r) {
  const Real sqrt2 = 1.4142135623730951;
  const Real two_over_sqrtpi = 1.1283791670955126;
  Real z = r/(sqrt2*sigma);
  if (z < 1.0e-4) {
    // erf(z)/z = (2/sqrt(pi)) (1 - z^2/3 + z^4/10 - ...)
    return mass/(2.0*sqrt2*sigma)*two_over_sqrtpi*(1.0 - z*z/3.0 + z*z*z*z/10.0);
  }
  return 0.5*mass*Kokkos::erf(z)/r;
}

struct Clump {
  Real mass, x1, x2, x3, sigma, svel, bulk_vy;
  int npart;
  Real ia, wavg, mu;
};

// Full conformal factor Eq. (26) on the host.
Real PsiFull(const GIProfile &p, const std::vector<Clump> &clumps,
             Real x, Real y, Real z) {
  Real riso = std::sqrt(x*x + y*y + z*z);
  Real psi = Psi0OfR(p, riso);
  for (const Clump &c : clumps) {
    Real dx = x - c.x1, dy = y - c.x2, dz = z - c.x3;
    psi += ClumpPsiTerm(c.mass, c.sigma, std::sqrt(dx*dx + dy*dy + dz*dz));
  }
  return psi;
}

//----------------------------------------------------------------------------------------
// Quadratures for the weight normalizations (all deterministic Simpson rules).

// <W>_a = int sqrt(1+v^2) h_a d3v for the Maxwellian h_a of width s (Eq. 33).
Real MeanLorentzMaxwell(Real s) {
  const int nq = 4096;
  Real vmax = 12.0*s;
  Real dv = vmax/nq;
  Real sum = 0.0;
  for (int i = 0; i <= nq; ++i) {
    Real v = i*dv;
    Real w = (i == 0 || i == nq) ? 1.0 : ((i % 2 == 1) ? 4.0 : 2.0);
    Real pd = v*v*std::exp(-0.5*v*v/(s*s));
    sum += w*pd*std::sqrt(1.0 + v*v);
  }
  sum *= dv/3.0;
  Real norm = std::sqrt(M_PI/2.0)*s*s*s;  // int v^2 exp(-v^2/2s^2) dv over [0,inf)
  return sum/norm;
}

// Expectation of func(r) where r = |X - X0|, X ~ isotropic Gaussian(width sigma) centered
// a distance d from X0. Radial density: noncentral-chi form for d>0, Maxwellian for d=0.
template <typename F>
Real GaussianRadialExpect(Real d, Real sigma, F func) {
  const int nq = 8192;
  Real rlo = std::max(static_cast<Real>(0.0), static_cast<Real>(d - 12.0*sigma));
  Real rhi = d + 12.0*sigma;
  Real drq = (rhi - rlo)/nq;
  Real sum = 0.0, norm = 0.0;
  for (int i = 0; i <= nq; ++i) {
    Real r = rlo + i*drq;
    Real w = (i == 0 || i == nq) ? 1.0 : ((i % 2 == 1) ? 4.0 : 2.0);
    Real pd;
    if (d > 1.0e-12) {
      Real em = std::exp(-0.5*(r - d)*(r - d)/(sigma*sigma));
      Real ep = std::exp(-0.5*(r + d)*(r + d)/(sigma*sigma));
      pd = r*(em - ep);  // times const/(d sigma sqrt(2 pi)) — cancels in ratio
    } else {
      pd = r*r*std::exp(-0.5*r*r/(sigma*sigma));
    }
    sum += w*pd*func(r);
    norm += w*pd;
  }
  return sum/norm;
}

// J_a = int [erf(r_a/sqrt2 sigma_a)/(2 r_a)] psi0^5(R) n0(R) d3X for unit clump mass:
// the clump-induced correction to the envelope rest-mass integral N_env. Evaluated as a
// 2D (R, mu) quadrature over the tabulated profile.
Real EnvelopePerturbIntegral(const GIProfile &p, Real d, Real sigma) {
  const int nmu = 128;
  Real dmu = 2.0/nmu;
  Real total = 0.0;
  std::size_t n = p.riso.size();
  std::vector<Real> gint(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    if (p.n0[i] <= 0.0) continue;
    Real riso = p.riso[i];
    Real angsum = 0.0;
    for (int j = 0; j <= nmu; ++j) {
      Real mu = -1.0 + j*dmu;
      Real w = (j == 0 || j == nmu) ? 1.0 : ((j % 2 == 1) ? 4.0 : 2.0);
      Real ra = std::sqrt(std::max(static_cast<Real>(riso*riso + d*d - 2.0*riso*d*mu),
                                   static_cast<Real>(0.0)));
      angsum += w*ClumpPsiTerm(1.0, sigma, ra);
    }
    angsum *= dmu/3.0;
    gint[i] = std::pow(p.psi0[i], 5)*p.n0[i]*riso*riso*angsum;
  }
  for (std::size_t i = 1; i < n; ++i) {
    total += 0.5*(gint[i-1] + gint[i])*(p.riso[i] - p.riso[i-1]);
  }
  return 2.0*M_PI*total;
}

//----------------------------------------------------------------------------------------
// Particle staging (host), CDF helpers — same pattern as relativistic_cluster.cpp.

struct PrtclStage {
  std::vector<Real> x, y, z, ux, uy, uz, mass;
  std::vector<int> gid, tag, component;

  void Add(Real x_, Real y_, Real z_, Real ux_, Real uy_, Real uz_, Real mass_,
           int gid_, int tag_, int component_) {
    x.push_back(x_);
    y.push_back(y_);
    z.push_back(z_);
    ux.push_back(ux_);
    uy.push_back(uy_);
    uz.push_back(uz_);
    mass.push_back(mass_);
    gid.push_back(gid_);
    tag.push_back(tag_);
    component.push_back(component_);
  }
};

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

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn RefinementCondition
//! \brief adaptive-refinement criterion: delegate to the shared Z4c module.

void RefinementCondition(MeshBlockPack *pmbp);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  // Enrolled unconditionally and before the early restart return: Mesh only calls it when
  // an <amr_criterion> asks for method = user, and a restarted adaptive run needs it too.
  user_ref_func = RefinementCondition;

  if (pmbp->pz4c == nullptr || pmbp->padm == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "gi_cluster requires a <z4c> block." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->ppart == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "gi_cluster requires a <particles> block." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!pmy_mesh_->three_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "gi_cluster is 3D-only." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->ppart->pusher != ParticlesPusher::gr_boris || !pmbp->ppart->feedback) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "gi_cluster requires <particles> pusher=gr_boris "
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
              << std::endl << "gi_cluster requires <particles> init=pgen." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  //--------------------------------------------------------------------------------------
  // Runtime parameters

  Real m0 = pin->GetOrAddReal("problem", "gi_m0", 0.61);
  Real r0 = pin->GetOrAddReal("problem", "gi_r0", 30.0);
  int n_env = pin->GetOrAddInteger("problem", "gi_n_env", 100000);
  int n_clump_total = pin->GetOrAddInteger("problem", "gi_n_clump", 40000);
  int seed = pin->GetOrAddInteger("problem", "gi_seed", 1);
  Real profile_dr = pin->GetOrAddReal("problem", "gi_profile_dr", 1.0e-3);
  int nclumps = pin->GetOrAddInteger("problem", "gi_nclumps", 4);
  Real mass_scale = pin->GetOrAddReal("problem", "gi_clump_mass_scale", 1.0);
  std::string dump_profile = pin->GetOrAddString("problem", "gi_dump_profile", "");

  if (m0 <= 0.0 || r0 <= 2.0*m0 || n_env <= 0 || (n_env % 2) != 0 ||
      n_clump_total < 0 || profile_dr <= 0.0 || nclumps < 0 || mass_scale < 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "gi_cluster requires gi_m0>0, gi_r0>2*gi_m0, even "
              << "gi_n_env>0, gi_n_clump>=0, gi_profile_dr>0, gi_nclumps>=0, "
              << "gi_clump_mass_scale>=0." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Clump table (Table I of the paper as defaults for the first four).
  const Real def_mass[4] = {0.12, 0.12, 0.09, 0.06};
  const Real def_x1[4] = {-3.0, 3.0, 0.0, 0.0};
  const Real def_x2[4] = {0.0, 0.0, 8.0, -12.0};
  const Real def_sigma[4] = {0.70, 0.70, 0.85, 0.90};
  const Real def_svel = 0.02;
  std::vector<Clump> clumps(nclumps);
  for (int a = 0; a < nclumps; ++a) {
    std::string tagc = "gi_clump" + std::to_string(a+1);
    Real dm = (a < 4) ? def_mass[a] : 0.0;
    Real dx1 = (a < 4) ? def_x1[a] : 0.0;
    Real dx2 = (a < 4) ? def_x2[a] : 0.0;
    Real dsg = (a < 4) ? def_sigma[a] : 1.0;
    clumps[a].mass = mass_scale*pin->GetOrAddReal("problem", tagc + "_mass", dm);
    clumps[a].x1 = pin->GetOrAddReal("problem", tagc + "_x1", dx1);
    clumps[a].x2 = pin->GetOrAddReal("problem", tagc + "_x2", dx2);
    clumps[a].x3 = pin->GetOrAddReal("problem", tagc + "_x3", 0.0);
    clumps[a].sigma = pin->GetOrAddReal("problem", tagc + "_sigma", dsg);
    clumps[a].svel = pin->GetOrAddReal("problem", tagc + "_s", def_svel);
    clumps[a].bulk_vy = pin->GetOrAddReal("problem", tagc + "_bulk_vy", 0.0);
    if (clumps[a].mass < 0.0 || clumps[a].sigma <= 0.0 || clumps[a].svel <= 0.0
        || std::abs(clumps[a].bulk_vy) >= 1.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "gi_cluster clump " << a+1
                << " requires mass>=0, sigma>0, s>0, |bulk_vy|<1." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  // Drop zero-mass clumps (mass_scale=0 recovers the exact static background).
  std::vector<Clump> active;
  for (const Clump &c : clumps) {
    if (c.mass > 0.0) active.push_back(c);
  }

  //--------------------------------------------------------------------------------------
  // Background equilibrium (Sec. II) and normalization integrals (Sec. IV)

  GIProfile prof = BuildProfile(m0, r0, profile_dr);

  // Loud internal consistency checks of the construction.
  Real q0_err = std::abs(prof.q0_integral - m0)/m0;
  if (q0_err > 1.0e-4) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "gi_cluster: int 4 pi R^2 q0 dR = " << prof.q0_integral
              << " differs from M0 = " << m0 << " by " << q0_err
              << " (rel); decrease gi_profile_dr." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real sum_ma = 0.0, dip1 = 0.0, dip2 = 0.0, dip3 = 0.0;
  for (const Clump &c : active) {
    sum_ma += c.mass;
    dip1 += c.mass*c.x1;
    dip2 += c.mass*c.x2;
    dip3 += c.mass*c.x3;
  }

  // Clump normalizations: <W>_a (Eq. 33), I_a (Eq. 36) via 1D radial-distance
  // expectations of psi0 and of every clump term (including self), mu_a (Eq. 37).
  // Envelope: N_env_tot = int psi psi0^5 n0 d3X = rest_mass + sum_a Ma J_a.
  Real nenv_integral = prof.rest_mass;
  for (const Clump &c : active) {
    Real d0 = std::sqrt(c.x1*c.x1 + c.x2*c.x2 + c.x3*c.x3);
    nenv_integral += c.mass*EnvelopePerturbIntegral(prof, d0, c.sigma);
  }
  for (Clump &c : active) {
    Real d0 = std::sqrt(c.x1*c.x1 + c.x2*c.x2 + c.x3*c.x3);
    Real mean_psi = GaussianRadialExpect(d0, c.sigma,
        [&](Real riso) { return Psi0OfR(prof, riso); });
    for (const Clump &b : active) {
      Real dab = std::sqrt((c.x1 - b.x1)*(c.x1 - b.x1) + (c.x2 - b.x2)*(c.x2 - b.x2) +
                           (c.x3 - b.x3)*(c.x3 - b.x3));
      mean_psi += GaussianRadialExpect(dab, c.sigma,
          [&](Real rb) { return ClumpPsiTerm(b.mass, b.sigma, rb); });
    }
    c.ia = c.mass*mean_psi;     // I_a = Ma <psi>_{G_a}
    Real gamma_bulk = 1.0/std::sqrt(1.0 - c.bulk_vy*c.bulk_vy);
    // For an isotropic rest distribution, <W_boost> = gamma_bulk <W_rest> because
    // <v_y>_rest=0.  This keeps the deposited clump energy normalization I_a unchanged.
    c.wavg = gamma_bulk*MeanLorentzMaxwell(c.svel);
  }

  // Particle counts: clump particles proportional to Ma, forced even (pairs).
  int n_clump_sum = 0;
  for (Clump &c : active) {
    Real frac = (sum_ma > 0.0) ? c.mass/sum_ma : 0.0;
    int na = 2*static_cast<int>(std::floor(0.5*frac*n_clump_total + 0.5));
    c.npart = na;
    n_clump_sum += na;
  }
  for (Clump &c : active) {
    if (c.npart <= 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "gi_cluster: clump particle split gave npart="
                << c.npart << "; increase gi_n_clump." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    c.mu = c.ia/(static_cast<Real>(c.npart)*c.wavg);  // Eq. (37)
  }
  Real mu_env = nenv_integral/static_cast<Real>(n_env);
  int npart_total = n_env + n_clump_sum;

  // Rejection bounds: psi - psi0 <= sum_a Ma/(sigma_a sqrt(2 pi)) everywhere; psi0 is
  // radially decreasing, so on the envelope support psi/psi0 <= k_env and globally
  // psi <= psi_bound. Violations (impossible unless the construction is wrong) are fatal.
  Real dpsi_max = 0.0;
  for (const Clump &c : active) {
    dpsi_max += c.mass/(c.sigma*std::sqrt(2.0*M_PI));
  }
  Real psi0_surf = 1.0 + 0.5*m0/prof.riso_s;
  Real k_env = 1.0 + dpsi_max/psi0_surf;
  Real psi_bound = prof.psi0[0] + dpsi_max;

  //--------------------------------------------------------------------------------------
  // Metric and gauge on the grid (Eqs. 21, 26, 38, 39), including ghost zones.

  int nprof = static_cast<int>(prof.riso.size());
  DvceArray1D<Real> radius_d("gi_radius", nprof);
  DvceArray1D<Real> psi0_d("gi_psi0", nprof);
  DvceArray1D<Real> alpha0_d("gi_alpha0", nprof);
  {
    auto radius_h = Kokkos::create_mirror_view(radius_d);
    auto psi0_h = Kokkos::create_mirror_view(psi0_d);
    auto alpha0_h = Kokkos::create_mirror_view(alpha0_d);
    for (int i = 0; i < nprof; ++i) {
      radius_h(i) = prof.riso[i];
      psi0_h(i) = prof.psi0[i];
      alpha0_h(i) = prof.alpha0[i];
    }
    Kokkos::deep_copy(radius_d, radius_h);
    Kokkos::deep_copy(psi0_d, psi0_h);
    Kokkos::deep_copy(alpha0_d, alpha0_h);
  }
  int ncl = static_cast<int>(active.size());
  DvceArray2D<Real> clump_d("gi_clumps", 5, std::max(ncl, 1));
  {
    auto clump_h = Kokkos::create_mirror_view(clump_d);
    for (int a = 0; a < ncl; ++a) {
      clump_h(0,a) = active[a].mass;
      clump_h(1,a) = active[a].x1;
      clump_h(2,a) = active[a].x2;
      clump_h(3,a) = active[a].x3;
      clump_h(4,a) = active[a].sigma;
    }
    Kokkos::deep_copy(clump_d, clump_h);
  }

  auto &size = pmbp->pmb->mb_size;
  auto &adm = pmbp->padm->adm;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int isg = is - indcs.ng, ieg = indcs.ie + indcs.ng;
  int jsg = js - indcs.ng, jeg = indcs.je + indcs.ng;
  int ksg = ks - indcs.ng, keg = indcs.ke + indcs.ng;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb = pmbp->nmb_thispack;
  Real surface_radius = prof.riso_s;
  Real mass_bg = m0;
  par_for("pgen gi_cluster metric", DevExeSpace(), 0, nmb-1,
          ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real x1 = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2 = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3 = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real riso = std::sqrt(x1*x1 + x2*x2 + x3*x3);
    Real psi0v, alpha0v;
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
      psi0v = psi0_d(lo) + frac*(psi0_d(hi) - psi0_d(lo));
      alpha0v = alpha0_d(lo) + frac*(alpha0_d(hi) - alpha0_d(lo));
    } else {
      Real q = 0.5*mass_bg/riso;
      psi0v = 1.0 + q;
      alpha0v = (1.0 - q)/(1.0 + q);
    }
    // psi = psi0 + clump potentials (Eq. 26); alpha = alpha0 (psi0/psi)^2 (Eq. 39).
    Real psi = psi0v;
    for (int a = 0; a < ncl; ++a) {
      Real dx = x1 - clump_d(1,a);
      Real dy = x2 - clump_d(2,a);
      Real dz = x3 - clump_d(3,a);
      Real ra = std::sqrt(dx*dx + dy*dy + dz*dz);
      psi += ClumpPsiTerm(clump_d(0,a), clump_d(4,a), ra);
    }
    Real alpha = alpha0v*(psi0v/psi)*(psi0v/psi);
    Real psi4 = psi*psi*psi*psi;
    adm.psi4(m,k,j,i) = psi4;
    adm.alpha(m,k,j,i) = alpha;
    for (int a = 0; a < 3; ++a) {
      adm.beta_u(m,a,k,j,i) = 0.0;
      for (int b = a; b < 3; ++b) {
        adm.g_dd(m,a,b,k,j,i) = (a == b) ? psi4 : 0.0;
        adm.vK_dd(m,a,b,k,j,i) = 0.0;
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
                << std::endl << "gi_cluster supports nghost=2,3,4." << std::endl;
      std::exit(EXIT_FAILURE);
  }
  pmbp->pz4c->Z4cToADM(pmbp);

  //--------------------------------------------------------------------------------------
  // Particle realization (Sec. IV). Every rank reproduces the same seeded global draw
  // and keeps only particles owned by its local MeshBlocks, so tags are decomposition
  // invariant (relativistic_cluster pattern). Gaussians use explicit Box-Muller so the
  // draw sequence is implementation-independent.

  particles::Particles *ppart = pmbp->ppart;
  std::mt19937_64 generator(static_cast<std::uint64_t>(seed));
  std::uniform_real_distribution<Real> uniform(0.0, 1.0);
  auto BoxMuller = [&](Real *n1, Real *n2) {
    Real u1 = std::max(uniform(generator), static_cast<Real>(1.0e-30));
    Real u2 = uniform(generator);
    Real fac = std::sqrt(-2.0*std::log(u1));
    *n1 = fac*std::cos(2.0*M_PI*u2);
    *n2 = fac*std::sin(2.0*M_PI*u2);
  };
  PrtclStage stage;
  int tag = 0;
  std::int64_t env_pos_trials = 0, env_mom_trials = 0;

  // ---- Envelope: positions from psi psi0^5 n0 (CDF proposal + psi/psi0 rejection),
  //      momenta |v| from f0(R,.) proportional to (1 - e^{-y} W) v^2 on [0, vmax],
  //      antithetic pairs (v, -v) at the same position.
  int placed = 0;
  while (placed < n_env) {
    ++env_pos_trials;
    Real ur = uniform(generator);
    std::size_t hicdf = CDFIndex(prof.cdf, ur);
    Real riso = InterpolateCDF(prof.cdf, prof.riso, hicdf, ur);
    Real yhere = InterpolateCDF(prof.cdf, prof.yv, hicdf, ur);
    Real psi0v = InterpolateCDF(prof.cdf, prof.psi0, hicdf, ur);

    Real cos_t = 2.0*uniform(generator) - 1.0;
    Real sin_t = std::sqrt(std::max(1.0 - cos_t*cos_t, 0.0));
    Real phi = 2.0*M_PI*uniform(generator);
    Real px = riso*sin_t*std::cos(phi);
    Real py = riso*sin_t*std::sin(phi);
    Real pz = riso*cos_t;

    Real psi = PsiFull(prof, active, px, py, pz);
    Real ratio = psi/psi0v;
    const Real bound_tol = 100.0*std::numeric_limits<Real>::epsilon();
    if (ratio > k_env*(1.0 + bound_tol)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "gi_cluster envelope rejection bound violated: "
                << "psi/psi0 = " << ratio << " > " << k_env << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (uniform(generator)*k_env > ratio) continue;

    // Momentum magnitude by rejection against v^2 on [0, vmax(y)] (Eqs. 7, 9, 30):
    // accept with probability (1 - e^{-y} W)/(1 - e^{-y}).
    Real vmax = std::sqrt(std::max(static_cast<Real>(std::expm1(2.0*yhere)),
                                   static_cast<Real>(0.0)));
    Real vmag = 0.0;
    if (vmax > 1.0e-12) {
      Real den = -std::expm1(-yhere);
      bool ok = false;
      for (int trial = 0; trial < 100000; ++trial) {
        ++env_mom_trials;
        Real v = vmax*std::cbrt(uniform(generator));
        Real w = std::sqrt(1.0 + v*v);
        Real num = -std::expm1(-yhere) - std::exp(-yhere)*v*v/(1.0 + w);
        if (uniform(generator)*den <= num) {
          vmag = v;
          ok = true;
          break;
        }
      }
      if (!ok) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "gi_cluster envelope momentum rejection stalled."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    Real cos_vt = 2.0*uniform(generator) - 1.0;
    Real sin_vt = std::sqrt(std::max(1.0 - cos_vt*cos_vt, 0.0));
    Real vphi = 2.0*M_PI*uniform(generator);
    // Stored covariant momentum: u_i = psi^2 v^ihat for gamma_ij = psi^4 delta_ij.
    Real umag = psi*psi*vmag;
    Real ux = umag*sin_vt*std::cos(vphi);
    Real uy = umag*sin_vt*std::sin(vphi);
    Real uz = umag*cos_vt;

    int mloc = ppart->FindContainingMeshBlock(px, py, pz);
    if (mloc >= 0) {
      stage.Add(px, py, pz, ux, uy, uz, mu_env, pmbp->gids + mloc, tag, 0);
      stage.Add(px, py, pz, -ux, -uy, -uz, mu_env, pmbp->gids + mloc, tag + 1, 0);
    }
    tag += 2;
    placed += 2;
  }

  // ---- Clumps: positions X ~ Ga rejected against psi/psi_bound (target P_a = psi Ga up
  //      to normalization, Eq. 36); momenta from the Maxwellian h_a; antithetic pairs.
  std::int64_t clump_pos_trials = 0;
  for (std::size_t a = 0; a < active.size(); ++a) {
    const Clump &c = active[a];
    int placed_c = 0;
    while (placed_c < c.npart) {
      ++clump_pos_trials;
      Real g1, g2, g3, g4;
      BoxMuller(&g1, &g2);
      BoxMuller(&g3, &g4);
      Real px = c.x1 + c.sigma*g1;
      Real py = c.x2 + c.sigma*g2;
      Real pz = c.x3 + c.sigma*g3;
      Real psi = PsiFull(prof, active, px, py, pz);
      if (psi > psi_bound*(1.0 + 100.0*std::numeric_limits<Real>::epsilon())) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "gi_cluster clump rejection bound violated: psi = "
                  << psi << " > " << psi_bound << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (uniform(generator)*psi_bound > psi) continue;

      Real v1, v2, v3, v4;
      BoxMuller(&v1, &v2);
      BoxMuller(&v3, &v4);
      // Form an antithetic thermal pair in the clump rest frame, then apply the same
      // orthonormal y-directed Lorentz boost to both four-velocities:
      //   W' = gamma (W + beta v_y),  v'_y = gamma (v_y + beta W).
      // Only the spatial four-velocity is stored; its covariant coordinate components are
      // u_i = psi^2 v'_ihat for gamma_ij = psi^4 delta_ij.
      Real vx = c.svel*v1;
      Real vy = c.svel*v2;
      Real vz = c.svel*v3;
      Real w_rest = std::sqrt(1.0 + vx*vx + vy*vy + vz*vz);
      Real gamma_bulk = 1.0/std::sqrt(1.0 - c.bulk_vy*c.bulk_vy);
      Real vyp = gamma_bulk*(vy + c.bulk_vy*w_rest);
      Real vym = gamma_bulk*(-vy + c.bulk_vy*w_rest);
      Real psi2 = psi*psi;

      int mloc = ppart->FindContainingMeshBlock(px, py, pz);
      if (mloc >= 0) {
        stage.Add(px, py, pz, psi2*vx, psi2*vyp, psi2*vz, c.mu,
                  pmbp->gids + mloc, tag, static_cast<int>(a) + 1);
        stage.Add(px, py, pz, -psi2*vx, psi2*vym, -psi2*vz, c.mu,
                  pmbp->gids + mloc, tag + 1, static_cast<int>(a) + 1);
      }
      tag += 2;
      placed_c += 2;
    }
  }

  //--------------------------------------------------------------------------------------
  // Fill the particle arrays and the global census (relativistic_cluster pattern).

  int npart = static_cast<int>(stage.x.size());

  // Exact finite-N matter ledger on the initialized slice.  The global staged particle
  // set is a disjoint partition across ranks, so an MPI sum counts every particle once.
  // P_i and L_z use the stored covariant coordinate momentum, matching the matter source
  // in the ADM momentum/angular-momentum volume integrals.  P_hat is included to expose
  // the intended orthonormal boost directly.
  constexpr int nledger = 8;
  int ncomponents = static_cast<int>(active.size()) + 1;  // envelope + clumps
  std::vector<Real> ledger(ncomponents*nledger, 0.0);
  std::vector<std::int64_t> ledger_count(ncomponents, 0);
  for (int p = 0; p < npart; ++p) {
    int comp = stage.component[p];
    Real psi = PsiFull(prof, active, stage.x[p], stage.y[p], stage.z[p]);
    Real inv_psi2 = 1.0/(psi*psi);
    Real vx = stage.ux[p]*inv_psi2;
    Real vy = stage.uy[p]*inv_psi2;
    Real vz = stage.uz[p]*inv_psi2;
    Real w = std::sqrt(1.0 + vx*vx + vy*vy + vz*vz);
    Real mass = stage.mass[p];
    Real *sum = &ledger[comp*nledger];
    sum[0] += mass*w;
    sum[1] += mass*stage.ux[p];
    sum[2] += mass*stage.uy[p];
    sum[3] += mass*stage.uz[p];
    sum[4] += mass*vx;
    sum[5] += mass*vy;
    sum[6] += mass*vz;
    sum[7] += mass*(stage.x[p]*stage.uy[p] - stage.y[p]*stage.ux[p]);
    ++ledger_count[comp];
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, ledger.data(), static_cast<int>(ledger.size()),
                MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, ledger_count.data(), static_cast<int>(ledger_count.size()),
                MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif

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
              << npart_total << " gi_cluster particles; the mesh must cover the whole "
              << "envelope (isotropic surface radius " << prof.riso_s << ") and all "
              << "clump Gaussians." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  SeedSnapshots();

  //--------------------------------------------------------------------------------------
  // Banner and normalization record (rank 0).

  if (global_variable::my_rank == 0) {
    std::cout << "GI cluster initialized (Einstein-Vlasov envelope + clumps):"
              << std::endl;
    std::printf("  background: M0=%.10g R0=%.10g Riso_s=%.10g yc=%.10g\n",
                m0, r0, prof.riso_s, prof.yc);
    std::printf("  f0 consts : Ecut=%.10g A=%.10g B=%.10g\n",
                prof.ecut, prof.aval, prof.bval);
    std::printf("  checks    : int 4piR^2 q0 dR=%.10g (M0 target, rel err %.2e); "
                "psi0_c=%.8g alpha0_c=%.8g\n",
                prof.q0_integral, q0_err, prof.psi0[0], prof.alpha0[0]);
    std::printf("  envelope  : N=%d mu_env=%.10g rest_mass=%.10g Nenv_int=%.10g\n",
                n_env, mu_env, prof.rest_mass, nenv_integral);
    std::printf("  clumps    : n=%d sum_Ma=%.10g dipole=(%.3e,%.3e,%.3e) "
                "M_ADM=%.10g\n",
                ncl, sum_ma, dip1, dip2, dip3, m0 + sum_ma);
    for (int a = 0; a < ncl; ++a) {
      std::printf("  clump %d   : Ma=%.6g X=(%.4g,%.4g,%.4g) sigma=%.4g s=%.4g "
                  "bulk_vy=%.8g Na=%d Ia=%.10g <W>=%.10g mu=%.10g\n",
                  a+1, active[a].mass, active[a].x1, active[a].x2, active[a].x3,
                  active[a].sigma, active[a].svel, active[a].bulk_vy, active[a].npart,
                  active[a].ia, active[a].wavg, active[a].mu);
    }
    std::printf("  sampling  : seed=%d k_env=%.6g psi_bound=%.6g env_accept=%.4g "
                "clump_accept=%.4g mom_accept=%.4g\n",
                seed, k_env, psi_bound,
                static_cast<Real>(n_env/2)/static_cast<Real>(env_pos_trials),
                (clump_pos_trials > 0)
                    ? static_cast<Real>(n_clump_sum/2)/
                      static_cast<Real>(clump_pos_trials) : 0.0,
                (env_mom_trials > 0)
                    ? static_cast<Real>(n_env/2)/static_cast<Real>(env_mom_trials)
                    : 0.0);
    for (int comp = 0; comp < ncomponents; ++comp) {
      const Real *sum = &ledger[comp*nledger];
      const char *name = (comp == 0) ? "envelope" : "clump";
      std::printf("  ledger %-8s %d: N=%lld sum(mW)=%.10g P_cov=(%.6e,%.6e,%.6e) "
                  "P_hat=(%.6e,%.6e,%.6e) Lz_cov=%.10g\n",
                  name, comp, static_cast<long long>(ledger_count[comp]), sum[0], sum[1],
                  sum[2], sum[3], sum[4], sum[5], sum[6], sum[7]);
    }
    std::printf("  particles : total=%d (envelope %d + clumps %d), antithetic pairs\n",
                npart_total, n_env, n_clump_sum);

    if (!dump_profile.empty()) {
      FILE *fp = std::fopen(dump_profile.c_str(), "w");
      if (fp != nullptr) {
        std::fprintf(fp, "# gi_cluster profile: R r psi0 alpha0 rho0 n0 q0 y\n");
        for (int i = 0; i < nprof; ++i) {
          std::fprintf(fp, "%.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e\n",
                       prof.riso[i], prof.rareal[i], prof.psi0[i], prof.alpha0[i],
                       prof.rho0[i], prof.n0[i], prof.q0[i], prof.yv[i]);
        }
        std::fclose(fp);
        std::cout << "  profile table written to " << dump_profile << std::endl;
      } else {
        std::cout << "### WARNING: gi_cluster could not open gi_dump_profile file '"
                  << dump_profile << "'" << std::endl;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn RefinementCondition
//! \brief set the per-MeshBlock refine/derefine flags from <z4c_amr>.

void RefinementCondition(MeshBlockPack *pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}
