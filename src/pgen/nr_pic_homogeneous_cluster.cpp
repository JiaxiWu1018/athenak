//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file nr_pic_homogeneous_cluster.cpp
//! \brief Homogeneous circular/tangential-orbit clusters for the NRPIC reproduction of
//! Shapiro & Teukolsky, ApJ 298, 34 (1985), Section VIIb and Appendix B.
//!
//! The initial slice is the uniform-density, time-symmetric sphere in isotropic
//! Cartesian coordinates.  M is the ADM mass, R is the areal surface radius, r is the
//! isotropic radius, and r_s is the areal radius:
//!
//!   m(r_s) = M r_s^3/R^3,       rho = 3M/(4 pi R^3),
//!   v_perp = xi sqrt[m/(r_s - 2m)],
//!   alpha = (1-2M/R)^(3/4) (1-2M r_s^2/R^3)^(-1/4)       (r_s <= R).
//!
//! The spatial geometry is the analytic constant-density geometry used by nr_pic_os:
//! gamma_ij = A(r)^2 delta_ij, K_ij=0, beta^i=0, with A=psi^2.  Outside the surface
//! the data are Schwarzschild in isotropic coordinates.
//!
//! Equal-rest-mass particles use a quiet start.  Radial shell locations are quantiles of
//!   dM0 proportional to rho r_s^2 dr_s/[W sqrt(1-2m/r_s)].
//! Each shell has an independently rotated Fibonacci angular set.  The optional
//! octahedral quiet start expands Fibonacci seed points through the 24 proper cube
//! rotations, exactly cancelling shell l=1,2,3 moments.  Four co-located particles at
//! every angular site carry velocities +e_theta, -e_theta, +e_phi, -e_phi.  This
//! cancels local momentum and net angular momentum pairwise while representing
//! isotropic transverse stress.  AthenaK stores covariant spatial four-velocity
//! u_i=A W v_perp t_i.
//!
//! Supported configurations:
//!   fixed control: <adm> and <particles> feedback=false
//!   live field:    <z4c> and <particles> feedback=true
//!
//! Public <problem> parameters:
//!   cluster_mass, cluster_radius_over_mass, cluster_xi
//!   cluster_nradial, cluster_nangular, cluster_seed
//!   cluster_octahedral_quiet_start
//!   cluster_rotation_enable, cluster_rotation_axis_x, cluster_rotation_axis_y,
//!   cluster_rotation_axis_z, cluster_rotation_angle
//!   cluster_center_x1, cluster_center_x2, cluster_center_x3

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
#include "z4c/tmunu.hpp"
#include "eos/primitive-solver/geom_math.hpp"
#include "particles/particles.hpp"
#include "particles/lagrange_interp.hpp"
#include "outputs/outputs.hpp"
#include "pgen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

Real cluster_center_history[3] = {0.0, 0.0, 0.0};
Real cluster_particle_mass_history = 0.0;
int cluster_nradial_history = 0;
int cluster_nangular_history = 0;
std::string cluster_shell_fname_history;
std::string cluster_mass_radii_fname_history;

struct ClusterProfile {
  std::vector<Real> u;
  std::vector<Real> cdf;
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

[[noreturn]] void Fatal(const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

// SplitMix64 is used only as a stateless, portable hash.  It makes every shell rotation
// a function of (seed,shell), independent of MPI decomposition and iteration history.
std::uint64_t SplitMix64(std::uint64_t x) {
  x += UINT64_C(0x9e3779b97f4a7c15);
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31);
}

Real HashUnit(std::uint64_t seed, int shell, int stream) {
  std::uint64_t key = seed ^ (UINT64_C(0xd1b54a32d192ed03) *
                              static_cast<std::uint64_t>(shell + 1));
  key ^= UINT64_C(0x94d049bb133111eb) * static_cast<std::uint64_t>(stream + 1);
  // Use the upper 53 bits so the conversion is exact on IEEE double builds.
  return static_cast<Real>(SplitMix64(key) >> 11) /
         static_cast<Real>(UINT64_C(9007199254740992));
}

void ShellRotation(std::uint64_t seed, int shell, Real rot[3][3]) {
  // Uniform unit quaternion (Shoemake construction).
  Real u1 = HashUnit(seed, shell, 0);
  Real u2 = HashUnit(seed, shell, 1);
  Real u3 = HashUnit(seed, shell, 2);
  Real s1 = std::sqrt(1.0 - u1);
  Real s2 = std::sqrt(u1);
  Real qx = s1*std::sin(2.0*M_PI*u2);
  Real qy = s1*std::cos(2.0*M_PI*u2);
  Real qz = s2*std::sin(2.0*M_PI*u3);
  Real qw = s2*std::cos(2.0*M_PI*u3);

  rot[0][0] = 1.0 - 2.0*(qy*qy + qz*qz);
  rot[0][1] = 2.0*(qx*qy - qz*qw);
  rot[0][2] = 2.0*(qx*qz + qy*qw);
  rot[1][0] = 2.0*(qx*qy + qz*qw);
  rot[1][1] = 1.0 - 2.0*(qx*qx + qz*qz);
  rot[1][2] = 2.0*(qy*qz - qx*qw);
  rot[2][0] = 2.0*(qx*qz - qy*qw);
  rot[2][1] = 2.0*(qy*qz + qx*qw);
  rot[2][2] = 1.0 - 2.0*(qx*qx + qy*qy);
}

void Rotate(const Real rot[3][3], const Real a[3], Real b[3]) {
  for (int i = 0; i < 3; ++i) {
    b[i] = rot[i][0]*a[0] + rot[i][1]*a[1] + rot[i][2]*a[2];
  }
}

void AxisAngleRotation(Real axis[3], Real angle, Real rot[3][3]) {
  if (!std::isfinite(axis[0]) || !std::isfinite(axis[1]) ||
      !std::isfinite(axis[2]) || !std::isfinite(angle)) {
    Fatal("cluster rotation axis and angle must be finite.");
  }
  Real norm = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
  if (!std::isfinite(norm) || norm <= 0.0) {
    Fatal("cluster rotation axis must have finite nonzero norm.");
  }
  for (int a = 0; a < 3; ++a) { axis[a] /= norm; }

  Real nx = axis[0], ny = axis[1], nz = axis[2];
  Real cosine = std::cos(angle);
  Real sine = std::sin(angle);
  Real one_minus_cosine = 1.0 - cosine;
  rot[0][0] = cosine + nx*nx*one_minus_cosine;
  rot[0][1] = nx*ny*one_minus_cosine - nz*sine;
  rot[0][2] = nx*nz*one_minus_cosine + ny*sine;
  rot[1][0] = ny*nx*one_minus_cosine + nz*sine;
  rot[1][1] = cosine + ny*ny*one_minus_cosine;
  rot[1][2] = ny*nz*one_minus_cosine - nx*sine;
  rot[2][0] = nz*nx*one_minus_cosine - ny*sine;
  rot[2][1] = nz*ny*one_minus_cosine + nx*sine;
  rot[2][2] = cosine + nz*nz*one_minus_cosine;

  Real orthogonality_error = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      Real dot = 0.0;
      for (int c = 0; c < 3; ++c) { dot += rot[c][a]*rot[c][b]; }
      Real expected = (a == b) ? 1.0 : 0.0;
      orthogonality_error = std::max(orthogonality_error, std::abs(dot - expected));
    }
  }
  Real determinant =
      rot[0][0]*(rot[1][1]*rot[2][2] - rot[1][2]*rot[2][1]) -
      rot[0][1]*(rot[1][0]*rot[2][2] - rot[1][2]*rot[2][0]) +
      rot[0][2]*(rot[1][0]*rot[2][1] - rot[1][1]*rot[2][0]);
  Real tolerance = 128.0*std::numeric_limits<Real>::epsilon();
  if (!std::isfinite(determinant) || orthogonality_error > tolerance ||
      std::abs(determinant - 1.0) > tolerance) {
    Fatal("cluster axis-angle construction did not produce a proper rotation.");
  }
}

// Apply one of the 24 proper rotations of a cube.  Expanding every angular seed
// through this group cancels all l=1,2,3 spherical moments on each radial shell.
void OctahedralRotate(int group_index, const Real a[3], Real b[3]) {
  static constexpr int permutations[6][3] = {
    {0, 1, 2}, {0, 2, 1}, {1, 0, 2},
    {1, 2, 0}, {2, 0, 1}, {2, 1, 0}
  };
  static constexpr int parity[6] = {1, -1, -1, 1, 1, -1};
  int iperm = group_index/4;
  int isign = group_index % 4;
  Real s0 = (isign & 1) ? -1.0 : 1.0;
  Real s1 = (isign & 2) ? -1.0 : 1.0;
  Real s2 = static_cast<Real>(parity[iperm])*s0*s1;
  b[0] = s0*a[permutations[iperm][0]];
  b[1] = s1*a[permutations[iperm][1]];
  b[2] = s2*a[permutations[iperm][2]];
}

ClusterProfile ConstructProfile(Real compactness, Real xi, int nradial) {
  // A fine deterministic table makes shell radii independent of the mesh/particle
  // decomposition.  The error is far below the finite-particle radial quadrature error.
  int ntab = std::max(16384, 128*nradial);
  ClusterProfile profile;
  profile.u.resize(ntab + 1);
  profile.cdf.resize(ntab + 1);
  profile.u[0] = 0.0;
  profile.cdf[0] = 0.0;

  auto integrand = [=](Real u) {
    Real x = compactness*u*u;  // m(r_s)/r_s
    Real v2 = xi*xi*x/(1.0 - 2.0*x);
    Real inv_w = std::sqrt(std::max(static_cast<Real>(1.0) - v2,
                                    static_cast<Real>(0.0)));
    return u*u*inv_w/std::sqrt(1.0 - 2.0*x);
  };

  Real h = 1.0/static_cast<Real>(ntab);
  Real fprev = integrand(0.0);
  for (int i = 1; i <= ntab; ++i) {
    Real u = i*h;
    Real f = integrand(u);
    profile.u[i] = u;
    profile.cdf[i] = profile.cdf[i-1] + 0.5*h*(fprev + f);
    fprev = f;
  }
  Real integral = profile.cdf.back();
  profile.rest_mass_over_m = 3.0*integral;
  for (Real &value : profile.cdf) { value /= integral; }
  profile.cdf.back() = 1.0;
  return profile;
}

Real InvertCDF(const ClusterProfile &profile, Real probability) {
  auto it = std::lower_bound(profile.cdf.begin() + 1, profile.cdf.end(), probability);
  std::size_t hi = static_cast<std::size_t>(it - profile.cdf.begin());
  std::size_t lo = hi - 1;
  Real denom = profile.cdf[hi] - profile.cdf[lo];
  Real frac = (denom > 0.0) ? (probability - profile.cdf[lo])/denom : 0.0;
  return profile.u[lo] + frac*(profile.u[hi] - profile.u[lo]);
}

// Invert r_s = C r/(2 r0^3 + M r^2).  The rationalized smaller root is accurate at
// the center and maps r_s=R to r=r0.
Real IsotropicRadius(Real rs, Real mass, Real r0, Real cnum) {
  if (rs == 0.0) { return 0.0; }
  Real discriminant = cnum*cnum - 8.0*mass*rs*rs*r0*r0*r0;
  discriminant = std::max(discriminant, static_cast<Real>(0.0));
  return 4.0*rs*r0*r0*r0/(cnum + std::sqrt(discriminant));
}

struct ClusterShellHealth {
  Real radial_kinetic_energy = 0.0;
  Real inner_radial_velocity = 0.0;
  Real nonfinite_particles = 0.0;
  Real minimum_radius = 0.0;
};

Real PercentileSorted(const std::vector<Real> &values, Real fraction) {
  if (values.empty()) { return std::numeric_limits<Real>::quiet_NaN(); }
  Real index = fraction*static_cast<Real>(values.size() - 1);
  std::size_t lo = static_cast<std::size_t>(std::floor(index));
  std::size_t hi = std::min(lo + 1, values.size() - 1);
  Real weight = index - static_cast<Real>(lo);
  return (1.0 - weight)*values[lo] + weight*values[hi];
}

bool FileIsEmpty(const std::string &fname) {
  std::ifstream input(fname, std::ios::binary | std::ios::ate);
  return (!input.good() || input.tellg() == std::streampos(0));
}

// The permanent Lagrangian shell is recovered from the immutable particle tag:
// tag = 4*(shell*nangular + angular_site) + quartet_member.  At every user-history
// output, this routine records exact shell quantiles without writing a full particle
// state.  Coordinate radial velocity is evaluated with the same current-metric
// interpolation and transport velocity used by gr_boris:
//   dx^i/dt = alpha gamma^{ij} u_j/W - beta^i.
// The local areal-radius proxy is
//   r sqrt[(tr_cart gamma - gamma_rr)/2],
// which equals the exact areal radius in the spherical conformally-flat initial data
// and remains a useful angularly sampled shell median in the live spacetime.
template <int NGHOST>
ClusterShellHealth WriteClusterShellSummary(Mesh *pm) {
  ClusterShellHealth health;
  particles::Particles *ppart = pm->pmb_pack->ppart;
  int npart = ppart->nprtcl_thispack;
  auto &pr = ppart->prtcl_rdata;
  auto &pi = ppart->prtcl_idata;
  auto &size = pm->pmb_pack->pmb->mb_size;
  int gids = pm->pmb_pack->gids;
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int ncell[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
  Real cx = cluster_center_history[0];
  Real cy = cluster_center_history[1];
  Real cz = cluster_center_history[2];

  DvceArray5D<Real> adm_metric = pm->pmb_pack->padm->u_adm;
  DvceArray5D<Real> z4c_metric;
  bool use_z4c = (pm->pmb_pack->pz4c != nullptr);
  if (use_z4c) { z4c_metric = pm->pmb_pack->pz4c->u0; }

  DvceArray1D<Real> radius("cluster shell radius", npart);
  DvceArray1D<Real> radius_areal("cluster shell areal radius", npart);
  DvceArray1D<Real> radial_velocity("cluster shell radial velocity", npart);
  DvceArray1D<int> tag("cluster shell tag", npart);

  Kokkos::parallel_for("homogeneous cluster shell samples",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p) {
    Real x[3] = {pr(IPX,p) - cx, pr(IPY,p) - cy, pr(IPZ,p) - cz};
    Real xabs[3] = {pr(IPX,p), pr(IPY,p), pr(IPZ,p)};
    Real u_d[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
    Real r = std::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
    Real rsafe = (r > 1.0e-14) ? r : 1.0e-14;
    Real n[3] = {x[0]/rsafe, x[1]/rsafe, x[2]/rsafe};
    int m = pi(PGID,p) - gids;
    const Real mb_par[9] = {
      size.d_view(m).x1min, size.d_view(m).x1max, size.d_view(m).dx1,
      size.d_view(m).x2min, size.d_view(m).x2max, size.d_view(m).dx2,
      size.d_view(m).x3min, size.d_view(m).x3max, size.d_view(m).dx3};
    int interp_indcs[4] = {m, -1, -1, -1};
    particles::SetInterpIndices(xabs, mb_par, ncell, interp_indcs);
    Real Lx[2*NGHOST] = {0.0};
    Real Ly[2*NGHOST] = {0.0};
    Real Lz[2*NGHOST] = {0.0};
    particles::CalcInterpWght<NGHOST>(
        xabs, mb_par, ncell, interp_indcs, Lx, Ly, Lz);

    Real alpha = 0.0;
    Real beta[3] = {0.0, 0.0, 0.0};
    if (use_z4c) {
      alpha = particles::LagrangeInterpolator<NGHOST>(
          z4c_metric, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, Lx, Ly, Lz);
      for (int a = 0; a < 3; ++a) {
        beta[a] = particles::LagrangeInterpolator<NGHOST>(
            z4c_metric, z4c::Z4c::I_Z4C_BETAX+a, interp_indcs, Lx, Ly, Lz);
      }
    } else {
      alpha = particles::LagrangeInterpolator<NGHOST>(
          adm_metric, adm::ADM::I_ADM_ALPHA, interp_indcs, Lx, Ly, Lz);
      for (int a = 0; a < 3; ++a) {
        beta[a] = particles::LagrangeInterpolator<NGHOST>(
            adm_metric, adm::ADM::I_ADM_BETAX+a, interp_indcs, Lx, Ly, Lz);
      }
    }

    Real g3d[6] = {0.0};
    for (int a = 0; a < 6; ++a) {
      g3d[a] = particles::LagrangeInterpolator<NGHOST>(
          adm_metric, adm::ADM::I_ADM_GXX+a, interp_indcs, Lx, Ly, Lz);
    }
    Real g3u[6] = {0.0};
    Primitive::InvertMatrix(g3u, g3d, Primitive::GetDeterminant(g3d));
    Real u_u[3] = {0.0};
    Primitive::RaiseForm(u_u, u_d, g3u);
    Real lorentz = std::sqrt(1.0 + Primitive::Contract(u_u, u_d));
    Real dxdt[3] = {
      alpha*u_u[0]/lorentz - beta[0],
      alpha*u_u[1]/lorentz - beta[1],
      alpha*u_u[2]/lorentz - beta[2]
    };
    Real vr = n[0]*dxdt[0] + n[1]*dxdt[1] + n[2]*dxdt[2];
    Real grr = g3d[0]*n[0]*n[0] + g3d[3]*n[1]*n[1] +
               g3d[5]*n[2]*n[2] +
               2.0*(g3d[1]*n[0]*n[1] + g3d[2]*n[0]*n[2] +
                    g3d[4]*n[1]*n[2]);
    Real gtangent = 0.5*(g3d[0] + g3d[3] + g3d[5] - grr);
    radius(p) = r;
    radius_areal(p) = r*std::sqrt(
        (gtangent > static_cast<Real>(0.0)) ? gtangent : static_cast<Real>(0.0));
    radial_velocity(p) = vr;
    tag(p) = pi(PTAG,p);
  });

  auto hradius = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), radius);
  auto hareal = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), radius_areal);
  auto hvr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), radial_velocity);
  auto htag = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tag);

  std::vector<Real> local_radius(npart), local_areal(npart), local_vr(npart);
  std::vector<int> local_tag(npart);
  for (int p = 0; p < npart; ++p) {
    local_radius[p] = hradius(p);
    local_areal[p] = hareal(p);
    local_vr[p] = hvr(p);
    local_tag[p] = htag(p);
  }

  std::vector<int> counts(global_variable::nranks, 0);
  std::vector<int> displacements(global_variable::nranks, 0);
  for (int rank = 0; rank < global_variable::nranks; ++rank) {
    counts[rank] = pm->nprtcl_eachrank[rank];
    if (rank > 0) { displacements[rank] = displacements[rank-1] + counts[rank-1]; }
  }
  int ntotal = displacements.back() + counts.back();
  std::vector<Real> global_radius, global_areal, global_vr;
  std::vector<int> global_tag;
  if (global_variable::my_rank == 0) {
    global_radius.resize(ntotal);
    global_areal.resize(ntotal);
    global_vr.resize(ntotal);
    global_tag.resize(ntotal);
  }
#if MPI_PARALLEL_ENABLED
  MPI_Gatherv(local_radius.data(), npart, MPI_ATHENA_REAL,
              global_radius.data(), counts.data(), displacements.data(), MPI_ATHENA_REAL,
              0, MPI_COMM_WORLD);
  MPI_Gatherv(local_areal.data(), npart, MPI_ATHENA_REAL,
              global_areal.data(), counts.data(), displacements.data(), MPI_ATHENA_REAL,
              0, MPI_COMM_WORLD);
  MPI_Gatherv(local_vr.data(), npart, MPI_ATHENA_REAL,
              global_vr.data(), counts.data(), displacements.data(), MPI_ATHENA_REAL,
              0, MPI_COMM_WORLD);
  MPI_Gatherv(local_tag.data(), npart, MPI_INT,
              global_tag.data(), counts.data(), displacements.data(), MPI_INT,
              0, MPI_COMM_WORLD);
#else
  global_radius = std::move(local_radius);
  global_areal = std::move(local_areal);
  global_vr = std::move(local_vr);
  global_tag = std::move(local_tag);
#endif

  if (global_variable::my_rank != 0) { return health; }

  std::vector<std::vector<Real>> shell_radius(cluster_nradial_history);
  std::vector<std::vector<Real>> shell_areal(cluster_nradial_history);
  std::vector<std::vector<Real>> shell_vr(cluster_nradial_history);
  std::vector<Real> all_radius;
  all_radius.reserve(ntotal);
  int particles_per_shell = 4*cluster_nangular_history;
  int tag_limit = particles_per_shell*cluster_nradial_history;
  int nonfinite = 0;
  for (int p = 0; p < ntotal; ++p) {
    int shell = (global_tag[p] >= 0 && global_tag[p] < tag_limit)
              ? global_tag[p]/particles_per_shell : -1;
    bool finite = std::isfinite(global_radius[p]) &&
                  std::isfinite(global_areal[p]) &&
                  std::isfinite(global_vr[p]);
    if (shell < 0 || !finite) {
      ++nonfinite;
      continue;
    }
    shell_radius[shell].push_back(global_radius[p]);
    shell_areal[shell].push_back(global_areal[p]);
    shell_vr[shell].push_back(global_vr[p]);
    all_radius.push_back(global_radius[p]);
  }

  bool shell_header = FileIsEmpty(cluster_shell_fname_history);
  std::ofstream shell_file(cluster_shell_fname_history, std::ios::app);
  if (!shell_file.good()) {
    Fatal("could not append Lagrangian shell summary '" +
          cluster_shell_fname_history + "'.");
  }
  shell_file << std::setprecision(17);
  if (shell_header) {
    shell_file
      << "# areal_radius_proxy = median[r*sqrt((tr_cart(gamma)-gamma_rr)/2)]\n"
      << "# radial_velocity = mean[n_i*(alpha*gamma^ij*u_j/W-beta^i)]\n"
      << "time,cycle,shell,median_riso,median_rareal,p10_riso,p90_riso,"
         "mean_vrad,count,enclosed_rest_mass\n";
  }

  Real enclosed_mass = 0.0;
  Real krad = 0.0;
  Real inner_vr_sum = 0.0;
  std::size_t inner_count = 0;
  int inner_shell_limit =
      static_cast<int>(std::ceil(0.10*cluster_nradial_history));
  for (int shell = 0; shell < cluster_nradial_history; ++shell) {
    std::sort(shell_radius[shell].begin(), shell_radius[shell].end());
    std::sort(shell_areal[shell].begin(), shell_areal[shell].end());
    Real vr_sum = 0.0;
    for (Real vr : shell_vr[shell]) {
      vr_sum += vr;
      krad += 0.5*cluster_particle_mass_history*vr*vr;
    }
    if (shell < inner_shell_limit) {
      inner_vr_sum += vr_sum;
      inner_count += shell_vr[shell].size();
    }
    enclosed_mass += cluster_particle_mass_history*shell_radius[shell].size();
    Real mean_vr = shell_vr[shell].empty()
                 ? std::numeric_limits<Real>::quiet_NaN()
                 : vr_sum/static_cast<Real>(shell_vr[shell].size());
    shell_file << pm->time << ',' << pm->ncycle << ',' << shell << ','
               << PercentileSorted(shell_radius[shell], 0.50) << ','
               << PercentileSorted(shell_areal[shell], 0.50) << ','
               << PercentileSorted(shell_radius[shell], 0.10) << ','
               << PercentileSorted(shell_radius[shell], 0.90) << ','
               << mean_vr << ',' << shell_radius[shell].size() << ','
               << enclosed_mass << '\n';
  }
  shell_file.close();

  std::sort(all_radius.begin(), all_radius.end());
  bool mass_header = FileIsEmpty(cluster_mass_radii_fname_history);
  std::ofstream mass_file(cluster_mass_radii_fname_history, std::ios::app);
  if (!mass_file.good()) {
    Fatal("could not append Lagrangian mass radii '" +
          cluster_mass_radii_fname_history + "'.");
  }
  mass_file << std::setprecision(17);
  if (mass_header) {
    mass_file << "time,cycle,R1,R5,R10,R25,R50,R75,R90,R95,R99\n";
  }
  static constexpr Real fractions[9] =
      {0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99};
  mass_file << pm->time << ',' << pm->ncycle;
  for (Real fraction : fractions) {
    mass_file << ',' << PercentileSorted(all_radius, fraction);
  }
  mass_file << '\n';
  mass_file.close();

  health.radial_kinetic_energy = krad;
  health.inner_radial_velocity =
      (inner_count == 0) ? std::numeric_limits<Real>::quiet_NaN()
                         : inner_vr_sum/static_cast<Real>(inner_count);
  health.nonfinite_particles = static_cast<Real>(nonfinite);
  health.minimum_radius =
      all_radius.empty() ? std::numeric_limits<Real>::quiet_NaN() : all_radius.front();
  return health;
}

struct ClusterFieldHealth {
  Real alpha_center = 0.0;
  Real alpha_min = 0.0;
  Real density_center = 0.0;
  Real density_max = 0.0;
};

ClusterFieldHealth MeasureClusterFieldHealth(Mesh *pm) {
  ClusterFieldHealth result;
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nkji = nx3*nx2*nx1;
  int nji = nx2*nx1;
  int nmkji = pm->pmb_pack->nmb_thispack*nkji;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &adm = pm->pmb_pack->padm->adm;
  auto &matter = pm->pmb_pack->ptmunu->tmunu;
  Real cx = cluster_center_history[0];
  Real cy = cluster_center_history[1];
  Real cz = cluster_center_history[2];

  Real density_max = -std::numeric_limits<Real>::max();
  Real alpha_min = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce("homogeneous cluster field extrema",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int idx, Real &local_density_max, Real &local_alpha_min) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = idx - m*nkji - k*nji - j*nx1;
    k += ks; j += js; i += is;
    local_density_max = fmax(local_density_max, matter.E(m,k,j,i));
    local_alpha_min = fmin(local_alpha_min, adm.alpha(m,k,j,i));
  }, Kokkos::Max<Real>(density_max), Kokkos::Min<Real>(alpha_min));

  array_sum::GlobalSum center_sum;
  Kokkos::parallel_reduce("homogeneous cluster central fields",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int idx, array_sum::GlobalSum &values) {
    int m = idx/nkji;
    int k0 = (idx - m*nkji)/nji;
    int j0 = (idx - m*nkji - k0*nji)/nx1;
    int i0 = idx - m*nkji - k0*nji - j0*nx1;
    int k = k0 + ks, j = j0 + js, i = i0 + is;
    Real x = CellCenterX(i0, nx1, size.d_view(m).x1min, size.d_view(m).x1max) - cx;
    Real y = CellCenterX(j0, nx2, size.d_view(m).x2min, size.d_view(m).x2max) - cy;
    Real z = CellCenterX(k0, nx3, size.d_view(m).x3min, size.d_view(m).x3max) - cz;
    Real dx = size.d_view(m).dx1;
    if (x*x + y*y + z*z <= 0.80*dx*dx) {
      array_sum::GlobalSum sample;
      sample.the_array[0] = adm.alpha(m,k,j,i);
      sample.the_array[1] = matter.E(m,k,j,i);
      sample.the_array[2] = 1.0;
      values += sample;
    }
  }, Kokkos::Sum<array_sum::GlobalSum>(center_sum));

  Real center_values[3] = {
    center_sum.the_array[0], center_sum.the_array[1], center_sum.the_array[2]};
#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &density_max, 1, MPI_ATHENA_REAL, MPI_MAX,
               0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN,
               0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, center_values, 3, MPI_ATHENA_REAL, MPI_SUM,
               0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&density_max, &density_max, 1, MPI_ATHENA_REAL, MPI_MAX,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&alpha_min, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN,
               0, MPI_COMM_WORLD);
    MPI_Reduce(center_values, center_values, 3, MPI_ATHENA_REAL, MPI_SUM,
               0, MPI_COMM_WORLD);
    density_max = 0.0;
    alpha_min = 0.0;
    center_values[0] = center_values[1] = center_values[2] = 0.0;
  }
#endif
  if (global_variable::my_rank == 0 && center_values[2] > 0.0) {
    result.alpha_center = center_values[0]/center_values[2];
    result.density_center = center_values[1]/center_values[2];
  }
  if (global_variable::my_rank == 0) {
    result.alpha_min = alpha_min;
    result.density_max = density_max;
  }
  return result;
}

}  // namespace

// User history keeps the inexpensive conservation ledger at the simulation cadence.
// Captured/excised contributions are joined from the death CSV by the analysis script.
void HomogeneousClusterHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 20;
  pdata->label[0] = "N_alive";
  pdata->label[1] = "M0_alive";
  pdata->label[2] = "E_part";
  pdata->label[3] = "Jpart_x";
  pdata->label[4] = "Jpart_y";
  pdata->label[5] = "Jpart_z";
  pdata->label[6] = "L_scalar";
  pdata->label[7] = "L2_mass";
  pdata->label[8] = "r_mass";
  pdata->label[9] = "geo_fbacks";
  pdata->label[10] = "M0_dev_l1";
  pdata->label[11] = "alpha_ctr";
  pdata->label[12] = "alpha_min";
  pdata->label[13] = "rho_ctr";
  pdata->label[14] = "rho_max";
  pdata->label[15] = "K_radial";
  pdata->label[16] = "vr_inner10";
  pdata->label[17] = "N_nonfinite";
  pdata->label[18] = "r_min";
  pdata->label[19] = "shell_rows";

  particles::Particles *ppart = pm->pmb_pack->ppart;
  auto &pr = ppart->prtcl_rdata;
  int npart = ppart->nprtcl_thispack;
  Real cx = cluster_center_history[0];
  Real cy = cluster_center_history[1];
  Real cz = cluster_center_history[2];
  Real particle_mass_reference = cluster_particle_mass_history;

  array_sum::GlobalSum sum;
  Kokkos::parallel_reduce("homogeneous cluster history",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, array_sum::GlobalSum &values) {
    array_sum::GlobalSum hvars;
    for (int n = 0; n < NHISTORY_VARIABLES; ++n) {
      hvars.the_array[n] = 0.0;
    }
    Real mp = pr(IPM,p);
    Real x = pr(IPX,p) - cx;
    Real y = pr(IPY,p) - cy;
    Real z = pr(IPZ,p) - cz;
    Real ux = pr(IPVX,p);
    Real uy = pr(IPVY,p);
    Real uz = pr(IPVZ,p);
    Real lx = y*uz - z*uy;
    Real ly = z*ux - x*uz;
    Real lz = x*uy - y*ux;
    Real labs = std::sqrt(lx*lx + ly*ly + lz*lz);
    Real radius = std::sqrt(x*x + y*y + z*z);
    hvars.the_array[0] = 1.0;
    // All particles have the same immutable rest mass. Centering the explicit
    // sum around that reference avoids O(N epsilon) loss in a serial reduction:
    // sum(m_p) = N*m_ref + sum(m_p-m_ref).  The L1 residual independently
    // catches any particle whose stored mass differs from the reference.
    Real mass_deviation = mp - particle_mass_reference;
    hvars.the_array[1] = mass_deviation;
    hvars.the_array[2] = mp*pr(IPEN,p);
    hvars.the_array[3] = mp*lx;
    hvars.the_array[4] = mp*ly;
    hvars.the_array[5] = mp*lz;
    hvars.the_array[6] = mp*labs;
    hvars.the_array[7] = mp*labs*labs;
    hvars.the_array[8] = mp*radius;
    hvars.the_array[10] = std::abs(mass_deviation);
    values += hvars;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum));

  for (int n = 0; n < pdata->nhist; ++n) {
    pdata->hdata[n] = (n < 9 || n == 10) ? sum.the_array[n] : 0.0;
  }
  pdata->hdata[1] = sum.the_array[0]*cluster_particle_mass_history +
                    sum.the_array[1];
  pdata->hdata[9] = static_cast<Real>(ppart->boris_nfail_cum);

  ClusterShellHealth shell_health;
  switch (pm->mb_indcs.ng) {
    case 2: shell_health = WriteClusterShellSummary<2>(pm); break;
    case 3: shell_health = WriteClusterShellSummary<3>(pm); break;
    case 4: shell_health = WriteClusterShellSummary<4>(pm); break;
    default: Fatal("Lagrangian shell summary supports nghost=2,3,4.");
  }
  ClusterFieldHealth field_health = MeasureClusterFieldHealth(pm);
  pdata->hdata[11] = field_health.alpha_center;
  pdata->hdata[12] = field_health.alpha_min;
  pdata->hdata[13] = field_health.density_center;
  pdata->hdata[14] = field_health.density_max;
  pdata->hdata[15] = shell_health.radial_kinetic_energy;
  pdata->hdata[16] = shell_health.inner_radial_velocity;
  pdata->hdata[17] = shell_health.nonfinite_particles;
  pdata->hdata[18] = shell_health.minimum_radius;
  pdata->hdata[19] = (global_variable::my_rank == 0)
                   ? static_cast<Real>(cluster_nradial_history) : 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  user_hist_func = HomogeneousClusterHistory;
  if (pmbp->padm == nullptr) {
    Fatal("nr_pic_homogeneous_cluster requires <adm> or <z4c> ADM variables.");
  }
  if (pmbp->ppart == nullptr) {
    Fatal("nr_pic_homogeneous_cluster requires a <particles> block.");
  }
  if (!pmy_mesh_->three_d) {
    Fatal("nr_pic_homogeneous_cluster is 3D-only.");
  }
  if (pmbp->ppart->pusher != ParticlesPusher::gr_boris) {
    Fatal("nr_pic_homogeneous_cluster requires <particles> pusher=gr_boris.");
  }

  bool live = (pmbp->pz4c != nullptr);
  if (live != pmbp->ppart->feedback) {
    Fatal("Use <adm> with feedback=false for a fixed field, or <z4c> with "
          "feedback=true for a live self-consistent field.");
  }

  Real mass = pin->GetOrAddReal("problem", "cluster_mass", 1.0);
  Real radius_over_mass =
      pin->GetOrAddReal("problem", "cluster_radius_over_mass", 6.1);
  Real xi = pin->GetOrAddReal("problem", "cluster_xi", 1.0);
  int nradial = pin->GetOrAddInteger("problem", "cluster_nradial", 32);
  int nangular = pin->GetOrAddInteger("problem", "cluster_nangular", 64);
  cluster_nradial_history = nradial;
  cluster_nangular_history = nangular;
  std::string basename = pin->GetString("job", "basename");
  cluster_shell_fname_history = basename + ".lagrangian_shells.csv";
  cluster_mass_radii_fname_history = basename + ".mass_radii.csv";
  bool octahedral_quiet_start =
      pin->GetOrAddBoolean("problem", "cluster_octahedral_quiet_start", false);
  bool rotation_enable =
      pin->GetOrAddBoolean("problem", "cluster_rotation_enable", false);
  Real rotation_axis[3] = {
    pin->GetOrAddReal("problem", "cluster_rotation_axis_x", 1.0),
    pin->GetOrAddReal("problem", "cluster_rotation_axis_y", 2.0),
    pin->GetOrAddReal("problem", "cluster_rotation_axis_z", 3.0)
  };
  Real rotation_angle =
      pin->GetOrAddReal("problem", "cluster_rotation_angle", 0.37);
  Real cluster_rotation[3][3] = {
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0}
  };
  if (rotation_enable) {
    AxisAngleRotation(rotation_axis, rotation_angle, cluster_rotation);
  }
  int seed = pin->GetOrAddInteger("problem", "cluster_seed", 1);
  Real center[3] = {
    pin->GetOrAddReal("problem", "cluster_center_x1", 0.0),
    pin->GetOrAddReal("problem", "cluster_center_x2", 0.0),
    pin->GetOrAddReal("problem", "cluster_center_x3", 0.0)
  };
  for (int a = 0; a < 3; ++a) { cluster_center_history[a] = center[a]; }

  if (mass <= 0.0 || radius_over_mass <= 2.0 || xi < 0.0 ||
      nradial <= 0 || nangular <= 0) {
    Fatal("Require cluster_mass>0, cluster_radius_over_mass>2, cluster_xi>=0, "
          "cluster_nradial>0, and cluster_nangular>0.");
  }
  if (octahedral_quiet_start && (nangular % 24 != 0)) {
    Fatal("cluster_octahedral_quiet_start=true requires "
          "cluster_nangular to be divisible by 24.");
  }
  Real compactness = 1.0/radius_over_mass;
  Real surface_v2 = xi*xi*compactness/(1.0 - 2.0*compactness);
  if (surface_v2 >= 1.0) {
    Fatal("cluster_xi gives a non-timelike surface velocity (v_perp^2 >= 1).");
  }
  std::int64_t nsites = static_cast<std::int64_t>(nradial)*nangular;
  std::int64_t npart64 = 4*nsites;
  if (npart64 > INT_MAX) {
    Fatal("4*cluster_nradial*cluster_nangular exceeds the 32-bit particle-tag range.");
  }
  int npart_total = static_cast<int>(npart64);

  Real radius = radius_over_mass*mass;
  Real sq = std::sqrt(1.0 - 2.0/radius_over_mass);
  Real r0 = 0.5*radius*(1.0 - 1.0/radius_over_mass + sq);
  Real cnum = (1.0 + sq)*r0*radius*radius;
  Real lapse_prefactor = std::pow(1.0 - 2.0/radius_over_mass, 0.75);
  ClusterProfile profile = ConstructProfile(compactness, xi, nradial);
  cluster_particle_mass_history =
      mass*profile.rest_mass_over_m/static_cast<Real>(npart_total);

  auto &indcs = pmbp->pmesh->mb_indcs;
  auto SeedSnapshots = [&]() {
    Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->adm_last, pmbp->padm->u_adm);
    if (live) {
      Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->z4c_last, pmbp->pz4c->u0);
    }
  };

  // A live restart restores metric and particles.  A fixed-background restart refreshes
  // the analytic ADM field below and leaves the restored particles untouched.
  bool initialize_metric = (!restart || !live);
  if (initialize_metric) {
    auto &size = pmbp->pmb->mb_size;
    auto &adm = pmbp->padm->adm;
    int is = indcs.is, js = indcs.js, ks = indcs.ks;
    int isg = is - indcs.ng, ieg = indcs.ie + indcs.ng;
    int jsg = js - indcs.ng, jeg = indcs.je + indcs.ng;
    int ksg = ks - indcs.ng, keg = indcs.ke + indcs.ng;
    int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    int nmb = pmbp->nmb_thispack;
    Real cx = center[0], cy = center[1], cz = center[2];
    Real m = mass, r_surface = r0, c = cnum, R = radius;
    Real alpha0 = lapse_prefactor;

    par_for("pgen homogeneous cluster metric", DevExeSpace(), 0, nmb-1,
            ksg, keg, jsg, jeg, isg, ieg,
    KOKKOS_LAMBDA(const int mb, const int k, const int j, const int i) {
      Real x = CellCenterX(i-is, nx1, size.d_view(mb).x1min,
                           size.d_view(mb).x1max) - cx;
      Real y = CellCenterX(j-js, nx2, size.d_view(mb).x2min,
                           size.d_view(mb).x2max) - cy;
      Real z = CellCenterX(k-ks, nx3, size.d_view(mb).x3min,
                           size.d_view(mb).x3max) - cz;
      Real riso = std::sqrt(x*x + y*y + z*z);
      Real conformal_a, alpha;
      if (riso <= r_surface) {
        conformal_a = c/(2.0*r_surface*r_surface*r_surface + m*riso*riso);
        Real rs = conformal_a*riso;
        Real interior = 1.0 - 2.0*m*rs*rs/(R*R*R);
        alpha = alpha0*std::pow(interior, -0.25);
      } else {
        Real q = 0.5*m/riso;
        conformal_a = (1.0 + q)*(1.0 + q);
        alpha = (1.0 - q)/(1.0 + q);
      }
      Real gamma_diag = conformal_a*conformal_a;
      adm.psi4(mb,k,j,i) = gamma_diag;
      adm.alpha(mb,k,j,i) = alpha;
      for (int a = 0; a < 3; ++a) {
        adm.beta_u(mb,a,k,j,i) = 0.0;
        for (int b = a; b < 3; ++b) {
          adm.g_dd(mb,a,b,k,j,i) = (a == b) ? gamma_diag : 0.0;
          adm.vK_dd(mb,a,b,k,j,i) = 0.0;
        }
      }
    });
    Kokkos::fence();

    if (live && !restart) {
      switch (indcs.ng) {
        case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin); break;
        case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin); break;
        case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin); break;
        default: Fatal("nr_pic_homogeneous_cluster supports nghost=2,3,4.");
      }
      pmbp->pz4c->Z4cToADM(pmbp);
      switch (indcs.ng) {
        case 2: pmbp->pz4c->ADMConstraints<2>(pmbp); break;
        case 3: pmbp->pz4c->ADMConstraints<3>(pmbp); break;
        case 4: pmbp->pz4c->ADMConstraints<4>(pmbp); break;
      }
    }
  }

  if (restart) {
    SeedSnapshots();
    return;
  }

  std::string init = pin->GetOrAddString("particles", "init", "ppc");
  if (init.compare("pgen") != 0) {
    Fatal("nr_pic_homogeneous_cluster requires <particles> init=pgen.");
  }

  particles::Particles *ppart = pmbp->ppart;
  PrtclStage stage;
  Real particle_mass = cluster_particle_mass_history;
  Real golden = 0.5*(std::sqrt(5.0) - 1.0);
  Real global_j[3] = {0.0, 0.0, 0.0};
  Real scalar_l = 0.0;
  Real energy0 = 0.0;
  Real max_quartet_p = 0.0;
  Real max_quartet_j = 0.0;

  for (int ir = 0; ir < nradial; ++ir) {
    Real probability = (ir + 0.5)/static_cast<Real>(nradial);
    Real u = InvertCDF(profile, probability);
    Real rs = radius*u;
    Real riso = IsotropicRadius(rs, mass, r0, cnum);
    Real conformal_a = (riso > 0.0) ? rs/riso :
        cnum/(2.0*r0*r0*r0);
    Real x = compactness*u*u;  // m(r_s)/r_s
    Real v = xi*std::sqrt(x/(1.0 - 2.0*x));
    Real w = 1.0/std::sqrt(1.0 - v*v);
    Real umag = conformal_a*w*v;
    Real alpha = lapse_prefactor*std::pow(1.0 - 2.0*x, -0.25);
    Real rot[3][3];
    ShellRotation(static_cast<std::uint64_t>(static_cast<std::uint32_t>(seed)),
                  ir, rot);

    int nfibonacci = octahedral_quiet_start ? nangular/24 : nangular;
    for (int ia = 0; ia < nangular; ++ia) {
      int ifibonacci = octahedral_quiet_start ? ia/24 : ia;
      int igroup = octahedral_quiet_start ? ia % 24 : 0;
      Real cth = 1.0 - 2.0*(ifibonacci + 0.5)/
                           static_cast<Real>(nfibonacci);
      Real sth = std::sqrt(std::max(static_cast<Real>(1.0) - cth*cth,
                                    static_cast<Real>(0.0)));
      Real phi = 2.0*M_PI*std::fmod(golden*ifibonacci, 1.0);
      Real cph = std::cos(phi), sph = std::sin(phi);
      Real n0[3] = {sth*cph, sth*sph, cth};
      Real eth0[3] = {cth*cph, cth*sph, -sth};
      Real eph0[3] = {-sph, cph, 0.0};
      Real nsym[3], ethsym[3], ephsym[3];
      if (octahedral_quiet_start) {
        OctahedralRotate(igroup, n0, nsym);
        OctahedralRotate(igroup, eth0, ethsym);
        OctahedralRotate(igroup, eph0, ephsym);
      } else {
        for (int a = 0; a < 3; ++a) {
          nsym[a] = n0[a];
          ethsym[a] = eth0[a];
          ephsym[a] = eph0[a];
        }
      }
      Real n[3], eth[3], eph[3];
      Rotate(rot, nsym, n);
      Rotate(rot, ethsym, eth);
      Rotate(rot, ephsym, eph);
      Real pos[3] = {
        center[0] + riso*n[0],
        center[1] + riso*n[1],
        center[2] + riso*n[2]
      };
      if (rotation_enable) {
        Real rel_unrotated[3] = {riso*n[0], riso*n[1], riso*n[2]};
        Real rel_rotated[3];
        Rotate(cluster_rotation, rel_unrotated, rel_rotated);
        for (int a = 0; a < 3; ++a) {
          pos[a] = center[a] + rel_rotated[a];
        }
      }

      Real qmomentum[3] = {0.0, 0.0, 0.0};
      Real qangular[3] = {0.0, 0.0, 0.0};
      for (int idir = 0; idir < 4; ++idir) {
        const Real *basis = (idir < 2) ? eth : eph;
        Real sign = ((idir % 2) == 0) ? 1.0 : -1.0;
        Real vel[3] = {
          sign*umag*basis[0], sign*umag*basis[1], sign*umag*basis[2]
        };
        if (rotation_enable) {
          Real vel_rotated[3];
          Rotate(cluster_rotation, vel, vel_rotated);
          for (int a = 0; a < 3; ++a) {
            vel[a] = vel_rotated[a];
          }
        }
        int tag = 4*(ir*nangular + ia) + idir;
        int mb = ppart->FindContainingMeshBlock(pos[0], pos[1], pos[2]);
        if (mb >= 0) {
          stage.Add(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2],
                    particle_mass, pmbp->gids + mb, tag);
        }

        Real rel[3] = {
          pos[0] - center[0], pos[1] - center[1], pos[2] - center[2]
        };
        Real lvec[3] = {
          rel[1]*vel[2] - rel[2]*vel[1],
          rel[2]*vel[0] - rel[0]*vel[2],
          rel[0]*vel[1] - rel[1]*vel[0]
        };
        for (int a = 0; a < 3; ++a) {
          qmomentum[a] += particle_mass*vel[a];
          qangular[a] += particle_mass*lvec[a];
          global_j[a] += particle_mass*lvec[a];
        }
        scalar_l += particle_mass*
            std::sqrt(lvec[0]*lvec[0] + lvec[1]*lvec[1] + lvec[2]*lvec[2]);
        energy0 += particle_mass*alpha*w;
      }
      Real qp = std::sqrt(qmomentum[0]*qmomentum[0] +
                          qmomentum[1]*qmomentum[1] +
                          qmomentum[2]*qmomentum[2]);
      Real qj = std::sqrt(qangular[0]*qangular[0] +
                          qangular[1]*qangular[1] +
                          qangular[2]*qangular[2]);
      max_quartet_p = std::max(max_quartet_p, qp);
      max_quartet_j = std::max(max_quartet_j, qj);
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
    Fatal("Placed " + std::to_string(pmy_mesh_->nprtcl_total) + " of " +
          std::to_string(npart_total) +
          " particles; enlarge or recenter the mesh.");
  }

  SeedSnapshots();

  if (global_variable::my_rank == 0) {
    Real period = 2.0*M_PI*std::sqrt(radius*radius*radius/mass);
    Real jnorm = std::sqrt(global_j[0]*global_j[0] +
                           global_j[1]*global_j[1] +
                           global_j[2]*global_j[2]);
    std::cout << std::setprecision(16)
              << "Homogeneous tangential-orbit cluster initialized\n"
              << "  mode=" << (live ? "live-z4c" : "fixed-adm")
              << "  M=" << mass << "  R/M=" << radius_over_mass
              << "  xi=" << xi << "\n"
              << "  N=" << npart_total << " (nradial=" << nradial
              << ", nangular=" << nangular << ", quartet=4)"
              << "  angular_start="
              << (octahedral_quiet_start ? "octahedral-fibonacci" : "fibonacci")
              << "  seed=" << seed << "\n"
              << "  rigid_rotation="
              << (rotation_enable ? "enabled" : "disabled") << "\n";
    if (rotation_enable) {
      std::cout << "  rotation_axis_unit=(" << rotation_axis[0] << ","
                << rotation_axis[1] << "," << rotation_axis[2] << ")"
                << "  rotation_angle=" << rotation_angle << "\n"
                << "  rotation_matrix=((" << cluster_rotation[0][0] << ","
                << cluster_rotation[0][1] << "," << cluster_rotation[0][2]
                << "),(" << cluster_rotation[1][0] << ","
                << cluster_rotation[1][1] << "," << cluster_rotation[1][2]
                << "),(" << cluster_rotation[2][0] << ","
                << cluster_rotation[2][1] << "," << cluster_rotation[2][2]
                << "))\n";
    }
    std::cout
              << "  M0/M=" << profile.rest_mass_over_m
              << "  m_p=" << particle_mass
              << "  r_surface_iso/M=" << r0/mass << "\n"
              << "  T_surface/M=" << period/mass
              << "  E0/M=" << energy0/mass << "\n"
              << "  J_part/M^2=(" << global_j[0]/(mass*mass) << ","
              << global_j[1]/(mass*mass) << ","
              << global_j[2]/(mass*mass) << ")"
              << "  |J_part|/M^2=" << jnorm/(mass*mass) << "\n"
              << "  L_scalar/M^2=" << scalar_l/(mass*mass)
              << "  max_quartet_|P|=" << max_quartet_p
              << "  max_quartet_|J|=" << max_quartet_j << "\n"
              << "  ADM initial budgets: M_ADM/M=1, J_ADM/M^2=(0,0,0)"
              << std::endl;
    if (std::abs(xi - 1.0) <= 16.0*std::numeric_limits<Real>::epsilon()) {
      Real stable = std::sqrt(radius*radius*radius/(6.0*mass));
      Real bound = std::sqrt(radius*radius*radius/(4.0*mass));
      std::cout << "  circular-orbit classes (areal): stable r_s/M < "
                << stable/mass << ", bound-unstable to " << bound/mass
                << ", unbound-unstable to R/M=" << radius_over_mass << std::endl;
    } else {
      Real xi_pred = 4.0*std::sqrt(mass/radius);
      std::cout << "  Appendix-B predictor: xi_crit=4 sqrt(M/R)=" << xi_pred
                << ", xi/xi_pred=" << xi/xi_pred << std::endl;
    }
  }
}
