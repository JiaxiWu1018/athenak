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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
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
#include "particles/particles.hpp"
#include "pgen/pgen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

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

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
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
  if (npart_total <= 0 || yc <= 0.0 || yc >= 1.0 || total_mass <= 0.0 ||
      profile_dx <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Require cluster_n>0, 0<cluster_yc<1, cluster_mass>0, "
              << "and cluster_profile_dx>0." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  ClusterProfile profile = ConstructProfile(yc, profile_dx);

  // Copy the isotropic radial metric profile to the device.
  int nprof = static_cast<int>(profile.x.size());
  DvceArray1D<Real> radius_d("cluster_radius", nprof);
  DvceArray1D<Real> conf_d("cluster_conformal_a", nprof);
  DvceArray1D<Real> lapse_d("cluster_lapse", nprof);
  auto radius_h = Kokkos::create_mirror_view(radius_d);
  auto conf_h = Kokkos::create_mirror_view(conf_d);
  auto lapse_h = Kokkos::create_mirror_view(lapse_d);
  for (int i = 0; i < nprof; ++i) {
    radius_h(i) = total_mass*profile.riso_over_m[i];
    conf_h(i) = profile.conformal_a[i];
    lapse_h(i) = profile.lapse[i];
  }
  Kokkos::deep_copy(radius_d, radius_h);
  Kokkos::deep_copy(conf_d, conf_h);
  Kokkos::deep_copy(lapse_d, lapse_h);

  // Smooth equilibrium metric in isotropic Cartesian coordinates. The exterior is the
  // standard isotropic Schwarzschild solution with mass M.
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
    } else {
      Real q = 0.5*mass/riso;
      conformal_a = (1.0 + q)*(1.0 + q);
      alpha = (1.0 - q)/(1.0 + q);
    }
    Real gamma_diag = conformal_a*conformal_a;
    adm.psi4(m,k,j,i) = gamma_diag;
    adm.alpha(m,k,j,i) = alpha;
    for (int a = 0; a < 3; ++a) {
      adm.beta_u(m,a,k,j,i) = 0.0;
      for (int b = a; b < 3; ++b) {
        adm.g_dd(m,a,b,k,j,i) = (a == b) ? gamma_diag : 0.0;
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
                << std::endl << "relativistic_cluster supports nghost=2,3,4."
                << std::endl;
      std::exit(EXIT_FAILURE);
  }
  pmbp->pz4c->Z4cToADM(pmbp);

  // Every rank reproduces the same seeded global draw and retains only particles owned
  // by its local MeshBlocks. Tags therefore remain rank-count and decomposition
  // invariant.
  particles::Particles *ppart = pmbp->ppart;
  std::mt19937_64 generator(static_cast<std::uint64_t>(seed));
  std::uniform_real_distribution<Real> uniform(0.0, 1.0);
  PrtclStage stage;
  Real particle_mass = total_mass*profile.rest_mass_over_m/
                       static_cast<Real>(npart_total);
  for (int tag = 0; tag < npart_total; ++tag) {
    Real ur = uniform(generator);
    std::size_t hi = CDFIndex(profile.cdf, ur);
    Real riso = total_mass*InterpolateCDF(
        profile.cdf, profile.riso_over_m, hi, ur);
    Real y = InterpolateCDF(profile.cdf, profile.y, hi, ur);
    Real conformal_a = InterpolateCDF(
        profile.cdf, profile.conformal_a, hi, ur);

    Real cos_theta = 2.0*uniform(generator) - 1.0;
    Real sin_theta = std::sqrt(std::max(static_cast<Real>(1.0) - cos_theta*cos_theta,
                                        static_cast<Real>(0.0)));
    Real phi = 2.0*M_PI*uniform(generator);
    Real px = center[0] + riso*sin_theta*std::cos(phi);
    Real py = center[1] + riso*sin_theta*std::sin(phi);
    Real pz = center[2] + riso*cos_theta;

    Real cos_vtheta = 2.0*uniform(generator) - 1.0;
    Real sin_vtheta = std::sqrt(std::max(static_cast<Real>(1.0) -
                                         cos_vtheta*cos_vtheta,
                                         static_cast<Real>(0.0)));
    Real vphi = 2.0*M_PI*uniform(generator);
    Real phat_over_m =
        std::sqrt(std::max(static_cast<Real>(1.0)/y - static_cast<Real>(1.0),
                           static_cast<Real>(0.0)));
    // Appendix A gives p^i = p^(hat i)/A. AthenaK stores covariant u_i, so for
    // gamma_ij=A^2 delta_ij: u_i = gamma_ij p^j/m0 = A p^(hat i)/m0.
    Real umag = conformal_a*phat_over_m;
    Real ux = umag*sin_vtheta*std::cos(vphi);
    Real uy = umag*sin_vtheta*std::sin(vphi);
    Real uz = umag*cos_vtheta;

    int m = ppart->FindContainingMeshBlock(px, py, pz);
    if (m >= 0) {
      stage.Add(px, py, pz, ux, uy, uz, particle_mass, pmbp->gids + m, tag);
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
  }
}
