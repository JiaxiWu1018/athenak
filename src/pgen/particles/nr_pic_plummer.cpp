//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file nr_pic_plummer.cpp
//! \brief Relativistic Plummer Einstein cluster: a static, spherical, purely
//! tangential-pressure cluster supported entirely by circular geodesics with uniformly
//! distributed orbital-plane orientations, zero net rotation and zero initial radial
//! velocity dispersion.  The NRPIC control for the homogeneous tangential-orbit cluster:
//! same physics, Plummer density profile instead of a uniform sphere.
//!
//! CONTINUUM MODEL (G = c = 1; see src/pgen/particles/plummer_profile.hpp).  The
//! prescribed quantity is the energy density measured by the STATIC (Eulerian) observer,
//! eps = T_{mu nu} n^mu n^nu, as a function of AREAL radius r -- NOT the particle number
//! density and NOT the rest mass per proper volume.  It is hard-truncated at r = r_t
//! (vacuum outside, no taper) and renormalised so that the ADM mass is exactly M:
//!
//!   f_t   = r_t^3/(r_t^2+b^2)^{3/2},        M_P = M/f_t
//!   eps   = 3 M_P/(4 pi b^3) (1+r^2/b^2)^{-5/2}   (r < r_t),   0 (r > r_t)
//!   m(r)  = M_P r^3/(r^2+b^2)^{3/2}               (r <= r_t),  M (r >= r_t)
//!   B     = (1-2m/r)^{-1/2},  v_c^2 = m/(r-2m),  W = sqrt((r-2m)/(r-3m))
//!   p_r   = 0,   p_t = eps v_c^2/2
//!
//! The metric is  ds^2 = -alpha^2 dt^2 + B^2 dr^2 + r^2 dOmega^2  with Phi = ln alpha and
//! j = ln(R/r) obtained by integrating Phi' = m/(r(r-2m)) and j' = (B-1)/r INWARD from
//! the exact Schwarzschild match at r_t.  On the Cartesian isotropic mesh
//!
//!   gamma_ij = psi^4 delta_ij,  psi = e^{-j/2},  K_ij = 0,  beta^i = 0,
//!   alpha    = e^{Phi(r(R))},   R = |x|,  r = r(R) the inverse of R = r e^{j}.
//!
//! NOTE the interior lapse is NOT sqrt(1-2m/r); that form is the vacuum exterior only.
//! Because K_ij = 0 and S_i = 0 the momentum constraint is satisfied identically, and
//! the construction satisfies  Lap_flat psi = -2 pi psi^5 eps  and
//! Lap_flat(alpha psi) = 2 pi alpha psi^5 (eps + 2 S) with S = 2 p_t; no elliptic solve
//! is required.  The circular ensemble supplies nonzero tangential stress even though
//! the geometry is static.
//!
//! PARTICLE SAMPLER ("stratified antithetic", the Plummer analogue of the homogeneous
//! campaign's sampler G).  N = 2 N_pair equal-rest-mass particles.  The radial measure
//! is the relativistic REST-mass measure dM0/dr = 4 pi r^2 eps B/W, so mu = M0/N and
//! N mu = M0 != M_ADM (binding energy).  For pair k:
//!
//!   q_k = (k + xi_k)/N_pair,  xi_k ~ U(0,1)     stratified radial quantile
//!   r_k = F0^{-1}(q_k),   F0 = M0(r)/M0
//!   n   = (sqrt(1-z^2) cos phi, sqrt(1-z^2) sin phi, z),  z ~ U(-1,1), phi ~ U(0,2pi)
//!   e1  = (khat x n)/|khat x n| with khat the Cartesian axis LEAST aligned with n,
//!   e2  = n x e1,   t = cos(zeta) e1 + sin(zeta) e2,   zeta ~ U(0,2pi)
//!   x   = R(r_k) n,   u_i^(+) = psi^2 W v_c t_i,   u_i^(-) = -u_i^(+)
//!
//! Both members of a pair sit at the SAME Cartesian position, so the initial deposited
//! momentum and total angular momentum cancel to roundoff while angular density noise is
//! retained.  Directions are independent across pairs: NO antipodal mirroring and NO
//! octant symmetry, so the odd (l = 1) density modes keep their natural sampling seed.
//! Because the stratum index equals the pair index equals tag/2, and q_k is monotone in
//! k, the particle TAG encodes the initial radial group exactly: cohort ledgers and
//! movie colouring by initial radius need no extra per-particle field.
//!
//! The finite-N angular noise floor is set by the number of INDEPENDENT angular
//! positions, N_pair = N/2, NOT by N: A_l^shot = N_pair^{-1/2}.
//!
//! All draws come from a stateless counter-based hash (SplitMix64) keyed by
//! (plummer_seed, pair id, stream), bit-identical to nr_pic_homogeneous_cluster.cpp, so
//! every realization is independent of the MPI decomposition and construction order.
//!
//! DIAGNOSTICS.  Besides the 20 history columns this pgen writes, at the history
//! cadence, two CSVs reduced over ALL particles in double precision:
//!   <basename>.plummer_shells.csv   current-radius bins: mass, <r_areal>, <v_r>,
//!                                   dispersions, and Sum m Y_lm(nhat) for l <= lmax,
//!                                   about the grid origin AND about the instantaneous
//!                                   centre of mass;
//!   <basename>.plummer_cohorts.csv  bins by INITIAL radial group (tag/2): mass,
//!                                   <r_areal>, <r_areal^2>, <v_r>, dispersions.
//! Radii are geometric: r_areal = R_iso sqrt(g_tangent) from the EVOLVED spatial metric,
//! not the initial r(R) map.  v_r is the coordinate radial velocity dx^i/dt . nhat.
//!
//! Public <problem> parameters:
//!   plummer_mass, plummer_b, plummer_rt, plummer_npair, plummer_seed,
//!   plummer_ntable, plummer_nmetric, plummer_center_x1/x2/x3,
//!   plummer_shell_nbin, plummer_shell_rmin, plummer_shell_rmax, plummer_lmax,
//!   plummer_cohort_nbin, plummer_diag_every
//!
//! Requires <particles> pusher=gr_boris, init=pgen, and either <z4c> with feedback=true
//! (live, self-consistent) or <adm> with feedback=false (frozen-metric orbit test).

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "z4c/z4c.hpp"
#include "eos/primitive-solver/geom_math.hpp"
#include "particles/particles.hpp"
#include "particles/lagrange_interp.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"
#include "plummer_profile.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

// ------------------------------------------------------------------ module state
// Set once by UserProblem and read by the history hook.  All are host-side scalars.
Real plummer_center[3] = {0.0, 0.0, 0.0};
Real plummer_particle_mass = 0.0;
Real plummer_M0 = 0.0;
Real plummer_MADM = 1.0;
Real plummer_bscale = 20.0;
Real plummer_rt = 400.0;
Real plummer_Rt = 0.0;
Real plummer_Phalf = 0.0;
Real plummer_rhalf = 0.0;
int plummer_npair = 0;
int plummer_ntotal = 0;
int plummer_shell_nbin = 48;
int plummer_cohort_nbin = 32;
int plummer_lmax = 4;
Real plummer_shell_rmin = 0.5;
Real plummer_shell_rmax = 400.0;
std::string plummer_shell_fname;
std::string plummer_cohort_fname;
bool plummer_shell_header_written = false;
bool plummer_cohort_header_written = false;

constexpr int NYLM_MAX = 25;      // (lmax+1)^2 with lmax = 4
constexpr int NQ_SHELL_BASE = 8;  // count,mass,r,vr,vr2,vt2,energy,|L|
constexpr int NQ_COHORT = 8;

[[noreturn]] void Fatal(const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

// ------------------------------------------------------------------ RNG (host only)
// SplitMix64 used purely as a stateless, portable hash.  Identical to the homogeneous
// cluster pgen so the two campaigns' samplers are directly comparable.
std::uint64_t SplitMix64(std::uint64_t x) {
  x += UINT64_C(0x9e3779b97f4a7c15);
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31);
}

//! Counter-based uniform variate on [0,1) keyed by (seed, id, stream).  Two SplitMix64
//! rounds decorrelate sequential ids and streams.  The upper 53 bits are used so the
//! conversion to double is exact.
Real HashUnitId(std::uint64_t seed, std::uint64_t id, std::uint64_t stream) {
  std::uint64_t key = SplitMix64(seed + UINT64_C(0x9e3779b97f4a7c15)*(id + 1));
  key = SplitMix64(key ^ (UINT64_C(0xbf58476d1ce4e5b9)*(stream + 1)));
  return static_cast<Real>(key >> 11)/static_cast<Real>(UINT64_C(9007199254740992));
}

// fixed stream ids -- DO NOT renumber, they define the realization
constexpr std::uint64_t STREAM_XI = 0;    // stratified radial jitter
constexpr std::uint64_t STREAM_Z = 1;     // cos(theta) of the position direction
constexpr std::uint64_t STREAM_PHI = 2;   // azimuth of the position direction
constexpr std::uint64_t STREAM_ZETA = 3;  // tangent-plane angle

// ------------------------------------------------------------------ staging
struct PrtclStage {
  std::vector<Real> x, y, z, ux, uy, uz;
  std::vector<int> gid, tag;
  void Add(Real x_, Real y_, Real z_, Real ux_, Real uy_, Real uz_, int gid_, int tag_) {
    x.push_back(x_); y.push_back(y_); z.push_back(z_);
    ux.push_back(ux_); uy.push_back(uy_); uz.push_back(uz_);
    gid.push_back(gid_); tag.push_back(tag_);
  }
};

// ------------------------------------------------------- device metric lookup table
// Rows of the DualArray2D<Real> metric table, sampled on a grid uniform in
// u = log1p(R/R0) so the device index is a single log1p and a multiply.
enum MetricRow {MT_R = 0, MT_PSI, MT_DPSI, MT_ALPHA, MT_DALPHA, MT_NROW};

//! \fn PlummerMetricAt
//! \brief cubic-Hermite interpolation of {psi, alpha} at isotropic radius R, using the
//! exact analytic Schwarzschild-isotropic exterior beyond R_t.  Runs on device.
KOKKOS_INLINE_FUNCTION
void PlummerMetricAt(const DualArray2D<Real>::t_dev &tab, int ntab, Real R0, Real inv_du,
                     Real Rt, Real M, Real R, Real *psi, Real *alpha) {
  if (R >= Rt) {
    const Real q = 0.5*M/R;
    *psi = 1.0 + q;
    *alpha = (1.0 - q)/(1.0 + q);
    return;
  }
  Real u = Kokkos::log1p(R/R0)*inv_du;
  int idx = static_cast<int>(u);
  if (idx < 0) { idx = 0; }
  if (idx > ntab - 1) { idx = ntab - 1; }
  const Real Ra = tab(MT_R, idx), Rb = tab(MT_R, idx + 1);
  const Real h = Rb - Ra;
  Real t = (h > 0.0) ? (R - Ra)/h : 0.0;
  if (t < 0.0) { t = 0.0; }
  if (t > 1.0) { t = 1.0; }
  const Real t2 = t*t, t3 = t2*t;
  const Real h00 = 2.0*t3 - 3.0*t2 + 1.0;
  const Real h10 = t3 - 2.0*t2 + t;
  const Real h01 = -2.0*t3 + 3.0*t2;
  const Real h11 = t3 - t2;
  *psi = h00*tab(MT_PSI, idx) + h10*h*tab(MT_DPSI, idx)
       + h01*tab(MT_PSI, idx + 1) + h11*h*tab(MT_DPSI, idx + 1);
  *alpha = h00*tab(MT_ALPHA, idx) + h10*h*tab(MT_DALPHA, idx)
         + h01*tab(MT_ALPHA, idx + 1) + h11*h*tab(MT_DALPHA, idx + 1);
}

// --------------------------------------------------------- real spherical harmonics
//! \fn RealYlm
//! \brief orthonormal REAL spherical harmonics Y_lm(nhat) for l = 0..4, written out in
//! Cartesian form on the unit sphere and indexed as  i = l^2 + (l + m).  Verified
//! against scipy.special.sph_harm_y (analysis/sph_real_check.py) to 1.5e-14.
KOKKOS_INLINE_FUNCTION
void RealYlm(Real x, Real y, Real z, int lmax, Real *Y) {
  const Real P = M_PI;
  const Real x2 = x*x, y2 = y*y, z2 = z*z;
  Y[0] = 0.5/Kokkos::sqrt(P);
  if (lmax < 1) { return; }
  const Real c1 = Kokkos::sqrt(3.0/(4.0*P));
  Y[1] = c1*y;  Y[2] = c1*z;  Y[3] = c1*x;
  if (lmax < 2) { return; }
  const Real c15 = 0.5*Kokkos::sqrt(15.0/P);
  Y[4] = c15*x*y;
  Y[5] = c15*y*z;
  Y[6] = 0.25*Kokkos::sqrt(5.0/P)*(3.0*z2 - 1.0);
  Y[7] = c15*x*z;
  Y[8] = 0.25*Kokkos::sqrt(15.0/P)*(x2 - y2);
  if (lmax < 3) { return; }
  const Real a3 = 0.25*Kokkos::sqrt(35.0/(2.0*P));
  const Real b3 = 0.5*Kokkos::sqrt(105.0/P);
  const Real d3 = 0.25*Kokkos::sqrt(21.0/(2.0*P));
  Y[9]  = a3*y*(3.0*x2 - y2);
  Y[10] = b3*x*y*z;
  Y[11] = d3*y*(5.0*z2 - 1.0);
  Y[12] = 0.25*Kokkos::sqrt(7.0/P)*z*(5.0*z2 - 3.0);
  Y[13] = d3*x*(5.0*z2 - 1.0);
  Y[14] = 0.25*Kokkos::sqrt(105.0/P)*z*(x2 - y2);
  Y[15] = a3*x*(x2 - 3.0*y2);
  if (lmax < 4) { return; }
  const Real e4 = 0.75*Kokkos::sqrt(35.0/P);
  const Real f4 = 0.75*Kokkos::sqrt(35.0/(2.0*P));
  const Real g4 = 0.75*Kokkos::sqrt(5.0/P);
  const Real h4 = 0.75*Kokkos::sqrt(5.0/(2.0*P));
  Y[16] = e4*x*y*(x2 - y2);
  Y[17] = f4*y*z*(3.0*x2 - y2);
  Y[18] = g4*x*y*(7.0*z2 - 1.0);
  Y[19] = h4*y*z*(7.0*z2 - 3.0);
  Y[20] = (3.0/16.0)*Kokkos::sqrt(1.0/P)*(35.0*z2*z2 - 30.0*z2 + 3.0);
  Y[21] = h4*x*z*(7.0*z2 - 3.0);
  Y[22] = 0.375*Kokkos::sqrt(5.0/P)*(x2 - y2)*(7.0*z2 - 1.0);
  Y[23] = f4*x*z*(x2 - 3.0*y2);
  Y[24] = (3.0/16.0)*Kokkos::sqrt(35.0/P)*(x2*(x2 - 3.0*y2) - y2*(3.0*x2 - y2));
}

// ------------------------------------------------------------------ field health
struct PlummerFieldHealth {
  Real alpha_min = 0.0;
  Real alpha_center = 0.0;
  Real ham_l2 = 0.0;      // sqrt(sum H^2 dV / sum dV), coordinate volume
  Real mom_l2 = 0.0;
  Real ham_max = 0.0;
};

PlummerFieldHealth MeasurePlummerFieldHealth(Mesh *pm) {
  PlummerFieldHealth health;
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pz4c == nullptr) { return health; }
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto u0 = pmbp->pz4c->u0;
  auto ucon = pmbp->pz4c->u_con;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int ncells = nmb*nx3*nx2*nx1;

  Real amin = std::numeric_limits<Real>::max();
  Real hmax = 0.0;
  Kokkos::parallel_reduce("plummer field minmax",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells),
  KOKKOS_LAMBDA(const int idx, Real &lamin, Real &lhmax) {
    const int i = idx % nx1;
    const int j = (idx/nx1) % nx2;
    const int k = (idx/(nx1*nx2)) % nx3;
    const int m = idx/(nx1*nx2*nx3);
    const Real a = u0(m, z4c::Z4c::I_Z4C_ALPHA, k+ks, j+js, i+is);
    const Real h = Kokkos::fabs(ucon(m, z4c::Z4c::I_CON_H, k+ks, j+js, i+is));
    lamin = Kokkos::fmin(lamin, a);
    lhmax = Kokkos::fmax(lhmax, h);
  }, Kokkos::Min<Real>(amin), Kokkos::Max<Real>(hmax));

  Real sums[3] = {0.0, 0.0, 0.0};  // sum H^2 dV, sum M^2 dV, sum dV
  Kokkos::parallel_reduce("plummer constraint norms",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells),
  KOKKOS_LAMBDA(const int idx, Real &lh, Real &lm2, Real &lv) {
    const int i = idx % nx1;
    const int j = (idx/nx1) % nx2;
    const int k = (idx/(nx1*nx2)) % nx3;
    const int m = idx/(nx1*nx2*nx3);
    const Real dv = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    const Real hh = ucon(m, z4c::Z4c::I_CON_H, k+ks, j+js, i+is);
    const Real mm = ucon(m, z4c::Z4c::I_CON_M, k+ks, j+js, i+is);
    lh += hh*hh*dv;
    lm2 += mm*mm*dv;
    lv += dv;
  }, sums[0], sums[1], sums[2]);

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, sums, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &amin, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &hmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  health.alpha_min = amin;
  health.ham_max = hmax;
  health.ham_l2 = (sums[2] > 0.0) ? std::sqrt(sums[0]/sums[2]) : 0.0;
  health.mom_l2 = (sums[2] > 0.0) ? std::sqrt(sums[1]/sums[2]) : 0.0;
  health.alpha_center = 0.0;   // filled by the caller from the central-cell probe
  return health;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn PlummerClusterDiagnostics
//! \brief per-particle reduction into (a) current-radius shell bins with real spherical
//! harmonic moments about the origin and about the instantaneous centre of mass, and
//! (b) initial-radial-cohort bins.  Reduced over ALL particles in double precision, then
//! Allreduced; rank 0 appends one block of rows to each CSV.  Returns a few scalars for
//! the history file.

namespace {

struct PlummerParticleHealth {
  Real com[3] = {0.0, 0.0, 0.0};
  Real mass_total = 0.0;
  Real A1_raw = 0.0;
  Real A1_com = 0.0;
  Real A2_com = 0.0;
  Real A3_com = 0.0;
  Real A4_com = 0.0;
  Real r_q50 = 0.0;
  Real sigma_r = 0.0;
  Real sigma_t = 0.0;
  Real r_min = 0.0;
  Real nonfinite = 0.0;
  Real energy = 0.0;
};

template <int NGHOST>
PlummerParticleHealth PlummerParticleDiagnostics(Mesh *pm, Real time, int ncycle,
                                                 bool write_csv) {
  PlummerParticleHealth H;
  MeshBlockPack *pmbp = pm->pmb_pack;
  particles::Particles *ppart = pmbp->ppart;
  const int npart = ppart->nprtcl_thispack;
  auto &pr = ppart->prtcl_rdata;
  auto &pi = ppart->prtcl_idata;
  auto &size = pmbp->pmb->mb_size;
  const int gids = pmbp->gids;
  auto &indcs = pm->mb_indcs;
  int ncell[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
  const Real cx = plummer_center[0], cy = plummer_center[1], cz = plummer_center[2];
  const int nbin = plummer_shell_nbin;
  const int ncoh = plummer_cohort_nbin;
  const int lmax = plummer_lmax;
  const int nylm = (lmax + 1)*(lmax + 1);
  const int nq = NQ_SHELL_BASE + nylm;
  const Real rmin = plummer_shell_rmin, rmax = plummer_shell_rmax;
  const Real lrmin = std::log(rmin);
  const Real inv_dlr = nbin/(std::log(rmax) - lrmin);
  const int npair = plummer_npair;

  DvceArray5D<Real> adm_metric = pmbp->padm->u_adm;
  DvceArray5D<Real> z4c_metric;
  const bool use_z4c = (pmbp->pz4c != nullptr);
  if (use_z4c) { z4c_metric = pmbp->pz4c->u0; }

  // ---- pass 1: per-particle kinematics + shell/cohort accumulation about the ORIGIN
  DvceArray1D<Real> acc("plummer shell acc", static_cast<std::size_t>(nbin)*nq);
  DvceArray1D<Real> coh("plummer cohort acc", static_cast<std::size_t>(ncoh)*NQ_COHORT);
  DvceArray1D<Real> glob("plummer global acc", 16);
  Kokkos::deep_copy(acc, 0.0);
  Kokkos::deep_copy(coh, 0.0);
  Kokkos::deep_copy(glob, 0.0);
  // cached per-particle quantities reused by pass 2 (COM-subtracted moments)
  DvceArray2D<Real> cache("plummer particle cache", 5, (npart > 0) ? npart : 1);

  Kokkos::parallel_for("plummer particle diagnostics",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p) {
    const Real mp = pr(IPM, p);
    const Real xa = pr(IPX, p), ya = pr(IPY, p), za = pr(IPZ, p);
    const Real x = xa - cx, y = ya - cy, z = za - cz;
    Real u_d[3] = {pr(IPVX, p), pr(IPVY, p), pr(IPVZ, p)};
    const Real riso = Kokkos::sqrt(x*x + y*y + z*z);
    const Real rs = (riso > 1.0e-14) ? riso : 1.0e-14;
    const Real n[3] = {x/rs, y/rs, z/rs};

    const int m = pi(PGID, p) - gids;
    const Real xabs[3] = {xa, ya, za};
    const Real mb_par[9] = {
      size.d_view(m).x1min, size.d_view(m).x1max, size.d_view(m).dx1,
      size.d_view(m).x2min, size.d_view(m).x2max, size.d_view(m).dx2,
      size.d_view(m).x3min, size.d_view(m).x3max, size.d_view(m).dx3};
    int interp_indcs[4] = {m, -1, -1, -1};
    particles::SetInterpIndices(xabs, mb_par, ncell, interp_indcs);
    Real Lx[2*NGHOST] = {0.0}, Ly[2*NGHOST] = {0.0}, Lz[2*NGHOST] = {0.0};
    particles::CalcInterpWght<NGHOST>(xabs, mb_par, ncell, interp_indcs, Lx, Ly, Lz);

    Real alpha = 1.0;
    Real beta[3] = {0.0, 0.0, 0.0};
    if (use_z4c) {
      alpha = particles::LagrangeInterpolator<NGHOST>(
          z4c_metric, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, Lx, Ly, Lz);
      for (int a = 0; a < 3; ++a) {
        beta[a] = particles::LagrangeInterpolator<NGHOST>(
            z4c_metric, z4c::Z4c::I_Z4C_BETAX + a, interp_indcs, Lx, Ly, Lz);
      }
    } else {
      alpha = particles::LagrangeInterpolator<NGHOST>(
          adm_metric, adm::ADM::I_ADM_ALPHA, interp_indcs, Lx, Ly, Lz);
      for (int a = 0; a < 3; ++a) {
        beta[a] = particles::LagrangeInterpolator<NGHOST>(
            adm_metric, adm::ADM::I_ADM_BETAX + a, interp_indcs, Lx, Ly, Lz);
      }
    }
    Real g3d[6] = {0.0};
    for (int a = 0; a < 6; ++a) {
      g3d[a] = particles::LagrangeInterpolator<NGHOST>(
          adm_metric, adm::ADM::I_ADM_GXX + a, interp_indcs, Lx, Ly, Lz);
    }
    Real g3u[6] = {0.0};
    Primitive::InvertMatrix(g3u, g3d, Primitive::GetDeterminant(g3d));
    Real u_u[3] = {0.0};
    Primitive::RaiseForm(u_u, u_d, g3u);
    const Real Wlor = Kokkos::sqrt(1.0 + Primitive::Contract(u_u, u_d));
    const Real dxdt[3] = {alpha*u_u[0]/Wlor - beta[0],
                          alpha*u_u[1]/Wlor - beta[1],
                          alpha*u_u[2]/Wlor - beta[2]};
    const Real vr = n[0]*dxdt[0] + n[1]*dxdt[1] + n[2]*dxdt[2];
    const Real v2 = dxdt[0]*dxdt[0] + dxdt[1]*dxdt[1] + dxdt[2]*dxdt[2];
    const Real vt2 = Kokkos::fmax(v2 - vr*vr, 0.0);
    // geometric (areal) radius from the EVOLVED metric: the mean tangential metric
    // coefficient on the sphere through the particle
    const Real grr = g3d[0]*n[0]*n[0] + g3d[3]*n[1]*n[1] + g3d[5]*n[2]*n[2]
                   + 2.0*(g3d[1]*n[0]*n[1] + g3d[2]*n[0]*n[2] + g3d[4]*n[1]*n[2]);
    const Real gtan = 0.5*(g3d[0] + g3d[3] + g3d[5] - grr);
    const Real rareal = riso*Kokkos::sqrt(Kokkos::fmax(gtan, 0.0));
    const Real lx = y*u_d[2] - z*u_d[1];
    const Real ly = z*u_d[0] - x*u_d[2];
    const Real lz = x*u_d[1] - y*u_d[0];
    const Real labs = Kokkos::sqrt(lx*lx + ly*ly + lz*lz);

    const bool finite = Kokkos::isfinite(riso) && Kokkos::isfinite(vr)
                     && Kokkos::isfinite(rareal) && Kokkos::isfinite(Wlor);
    cache(0, p) = rareal;
    cache(1, p) = riso;
    cache(2, p) = mp;
    cache(3, p) = finite ? 1.0 : 0.0;
    cache(4, p) = vr;

    Real Y[NYLM_MAX];
    RealYlm(n[0], n[1], n[2], lmax, Y);

    if (finite) {
      // current-radius shell bin (log-spaced in areal radius, clamped at both ends)
      int ib = static_cast<int>((Kokkos::log(Kokkos::fmax(rareal, 1.0e-12)) - lrmin)
                                *inv_dlr);
      if (ib < 0) { ib = 0; }
      if (ib > nbin - 1) { ib = nbin - 1; }
      const std::size_t o = static_cast<std::size_t>(ib)*nq;
      Kokkos::atomic_add(&acc(o + 0), 1.0);
      Kokkos::atomic_add(&acc(o + 1), mp);
      Kokkos::atomic_add(&acc(o + 2), mp*rareal);
      Kokkos::atomic_add(&acc(o + 3), mp*vr);
      Kokkos::atomic_add(&acc(o + 4), mp*vr*vr);
      Kokkos::atomic_add(&acc(o + 5), mp*vt2);
      Kokkos::atomic_add(&acc(o + 6), mp*alpha*Wlor);
      Kokkos::atomic_add(&acc(o + 7), mp*labs);
      for (int q = 0; q < nylm; ++q) {
        Kokkos::atomic_add(&acc(o + NQ_SHELL_BASE + q), mp*Y[q]);
      }
      // initial-radial cohort bin: the tag encodes the stratum, tag/2 = pair index
      const int pair = pi(PTAG, p)/2;
      int ic = (npair > 0) ? static_cast<int>(
          (static_cast<std::int64_t>(pair)*ncoh)/npair) : 0;
      if (ic < 0) { ic = 0; }
      if (ic > ncoh - 1) { ic = ncoh - 1; }
      const std::size_t oc = static_cast<std::size_t>(ic)*NQ_COHORT;
      Kokkos::atomic_add(&coh(oc + 0), 1.0);
      Kokkos::atomic_add(&coh(oc + 1), mp);
      Kokkos::atomic_add(&coh(oc + 2), mp*rareal);
      Kokkos::atomic_add(&coh(oc + 3), mp*rareal*rareal);
      Kokkos::atomic_add(&coh(oc + 4), mp*vr);
      Kokkos::atomic_add(&coh(oc + 5), mp*vr*vr);
      Kokkos::atomic_add(&coh(oc + 6), mp*vt2);
      Kokkos::atomic_add(&coh(oc + 7), mp*alpha*Wlor);
      // globals: mass, COM (isotropic coords), raw dipole, dispersions, energy
      Kokkos::atomic_add(&glob(0), mp);
      Kokkos::atomic_add(&glob(1), mp*x);
      Kokkos::atomic_add(&glob(2), mp*y);
      Kokkos::atomic_add(&glob(3), mp*z);
      Kokkos::atomic_add(&glob(4), Y[1]);
      Kokkos::atomic_add(&glob(5), Y[2]);
      Kokkos::atomic_add(&glob(6), Y[3]);
      Kokkos::atomic_add(&glob(7), 1.0);
      Kokkos::atomic_add(&glob(8), mp*vr*vr);
      Kokkos::atomic_add(&glob(9), mp*vt2);
      Kokkos::atomic_add(&glob(10), mp*alpha*Wlor);
    } else {
      Kokkos::atomic_add(&glob(11), 1.0);
    }
  });
  Kokkos::fence();

  auto hacc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), acc);
  auto hcoh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coh);
  auto hglob = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), glob);
  std::vector<Real> vacc(hacc.data(), hacc.data() + hacc.extent(0));
  std::vector<Real> vcoh(hcoh.data(), hcoh.data() + hcoh.extent(0));
  std::vector<Real> vglob(hglob.data(), hglob.data() + hglob.extent(0));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, vacc.data(), static_cast<int>(vacc.size()),
                MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, vcoh.data(), static_cast<int>(vcoh.size()),
                MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, vglob.data(), static_cast<int>(vglob.size()),
                MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  const Real mtot = vglob[0];
  const Real nalive = vglob[7];
  Real com[3] = {0.0, 0.0, 0.0};
  if (mtot > 0.0) {
    for (int a = 0; a < 3; ++a) { com[a] = vglob[1 + a]/mtot; }
  }
  // A_1 raw: unweighted mean of Y_1m over particles (the established definition; all
  // particles carry the same rest mass, so unweighted == mass weighted)
  Real a1sq = 0.0;
  if (nalive > 0.0) {
    for (int a = 0; a < 3; ++a) {
      const Real c = vglob[4 + a]/nalive;
      a1sq += c*c;
    }
  }
  H.A1_raw = std::sqrt(4.0*M_PI/3.0*a1sq);

  // ---- pass 2: multipoles about the instantaneous centre of mass
  DvceArray1D<Real> glob2("plummer com moments", NYLM_MAX + 4);
  Kokkos::deep_copy(glob2, 0.0);
  const Real comx = com[0], comy = com[1], comz = com[2];
  DvceArray1D<Real> accc("plummer shell acc com", static_cast<std::size_t>(nbin)*nylm);
  Kokkos::deep_copy(accc, 0.0);
  Kokkos::parallel_for("plummer com moments",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p) {
    if (cache(3, p) == 0.0) { return; }
    const Real x = pr(IPX, p) - cx - comx;
    const Real y = pr(IPY, p) - cy - comy;
    const Real z = pr(IPZ, p) - cz - comz;
    const Real rr = Kokkos::sqrt(x*x + y*y + z*z);
    const Real rs = (rr > 1.0e-14) ? rr : 1.0e-14;
    Real Y[NYLM_MAX];
    RealYlm(x/rs, y/rs, z/rs, lmax, Y);
    for (int q = 0; q < nylm; ++q) { Kokkos::atomic_add(&glob2(q), Y[q]); }
    Kokkos::atomic_add(&glob2(NYLM_MAX), 1.0);
    // per-shell COM-subtracted moments, binned by the SAME areal-radius bin as pass 1
    int ib = static_cast<int>((Kokkos::log(Kokkos::fmax(cache(0, p), 1.0e-12)) - lrmin)
                              *inv_dlr);
    if (ib < 0) { ib = 0; }
    if (ib > nbin - 1) { ib = nbin - 1; }
    const std::size_t o = static_cast<std::size_t>(ib)*nylm;
    for (int q = 0; q < nylm; ++q) { Kokkos::atomic_add(&accc(o + q), Y[q]); }
  });
  Kokkos::fence();
  auto hg2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), glob2);
  auto hac = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), accc);
  std::vector<Real> vg2(hg2.data(), hg2.data() + hg2.extent(0));
  std::vector<Real> vac(hac.data(), hac.data() + hac.extent(0));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, vg2.data(), static_cast<int>(vg2.size()),
                MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, vac.data(), static_cast<int>(vac.size()),
                MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  const Real n2 = vg2[NYLM_MAX];
  Real Acom[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  if (n2 > 0.0) {
    for (int l = 1; l <= lmax; ++l) {
      Real s = 0.0;
      for (int mm = -l; mm <= l; ++mm) {
        const Real c = vg2[l*l + (l + mm)]/n2;
        s += c*c;
      }
      Acom[l] = std::sqrt(4.0*M_PI/(2.0*l + 1.0)*s);
    }
  }
  H.A1_com = Acom[1];
  H.A2_com = Acom[2];
  H.A3_com = Acom[3];
  H.A4_com = Acom[4];
  for (int a = 0; a < 3; ++a) { H.com[a] = com[a]; }
  H.mass_total = mtot;
  H.nonfinite = vglob[11];
  H.energy = vglob[10];
  H.sigma_r = (mtot > 0.0) ? std::sqrt(std::max(vglob[8]/mtot, 0.0)) : 0.0;
  H.sigma_t = (mtot > 0.0) ? std::sqrt(std::max(vglob[9]/mtot, 0.0)) : 0.0;

  // enclosed rest-mass median areal radius, from the cumulative shell mass
  {
    Real cum = 0.0;
    const Real half = 0.5*mtot;
    H.r_q50 = plummer_shell_rmax;
    for (int ib = 0; ib < nbin; ++ib) {
      const Real mb = vacc[static_cast<std::size_t>(ib)*nq + 1];
      if (cum + mb >= half && mb > 0.0) {
        const Real lo = std::exp(lrmin + ib/inv_dlr);
        const Real hi = std::exp(lrmin + (ib + 1)/inv_dlr);
        H.r_q50 = lo + (hi - lo)*(half - cum)/mb;
        break;
      }
      cum += mb;
    }
  }

  // ---- CSV output (rank 0)
  if (write_csv && global_variable::my_rank == 0) {
    std::ofstream fs(plummer_shell_fname, std::ios::app);
    if (fs.good()) {
      if (!plummer_shell_header_written) {
        fs << "# Plummer cluster radial shell ledger.  Bins are uniform in "
              "log(r_areal) over [" << rmin << ", " << rmax << "] M with "
           << nbin << " bins; r_areal = R_iso sqrt(g_tangent) from the EVOLVED "
              "spatial metric.  v_r is the coordinate radial velocity dx^i/dt . nhat.\n"
           << "# Y_lm are orthonormal REAL spherical harmonics of the position "
              "direction, index l*l+(l+m); c_lm columns are Sum m_p Y_lm (origin) and "
              "Sum Y_lm (about the instantaneous centre of mass, unweighted).\n"
           << "time,cycle,bin,r_lo,r_hi,count,mass,m_r,m_vr,m_vr2,m_vt2,m_alphaW,m_absL";
        for (int q = 0; q < nylm; ++q) { fs << ",c" << q; }
        for (int q = 0; q < nylm; ++q) { fs << ",d" << q; }
        fs << "\n";
        plummer_shell_header_written = true;
      }
      fs << std::setprecision(12);
      for (int ib = 0; ib < nbin; ++ib) {
        const std::size_t o = static_cast<std::size_t>(ib)*nq;
        fs << time << "," << ncycle << "," << ib << ","
           << std::exp(lrmin + ib/inv_dlr) << ","
           << std::exp(lrmin + (ib + 1)/inv_dlr);
        for (int q = 0; q < NQ_SHELL_BASE; ++q) { fs << "," << vacc[o + q]; }
        for (int q = 0; q < nylm; ++q) { fs << "," << vacc[o + NQ_SHELL_BASE + q]; }
        for (int q = 0; q < nylm; ++q) {
          fs << "," << vac[static_cast<std::size_t>(ib)*nylm + q];
        }
        fs << "\n";
      }
    }
    std::ofstream fc(plummer_cohort_fname, std::ios::app);
    if (fc.good()) {
      if (!plummer_cohort_header_written) {
        fc << "# Plummer cluster Lagrangian cohort ledger.  Cohort = contiguous band of "
              "the stratified radial quantile, cohort = floor((tag/2)*ncohort/N_pair); "
              "because the stratum index equals the pair index and the quantile is "
              "monotone in it, cohort is an exact equal-rest-mass initial radial band.\n"
           << "time,cycle,cohort,count,mass,m_r,m_r2,m_vr,m_vr2,m_vt2,m_alphaW\n";
        plummer_cohort_header_written = true;
      }
      fc << std::setprecision(12);
      for (int ic = 0; ic < ncoh; ++ic) {
        const std::size_t o = static_cast<std::size_t>(ic)*NQ_COHORT;
        fc << time << "," << ncycle << "," << ic;
        for (int q = 0; q < NQ_COHORT; ++q) { fc << "," << vcoh[o + q]; }
        fc << "\n";
      }
    }
  }
  return H;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn PlummerClusterHistory
//! \brief 20 history columns, all reduced over every particle / cell in double precision.

void PlummerClusterHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 20;
  pdata->label[0]  = "N_alive";
  pdata->label[1]  = "M0_alive";
  pdata->label[2]  = "E_part";
  pdata->label[3]  = "A1_raw";
  pdata->label[4]  = "A1_com";
  pdata->label[5]  = "A2_com";
  pdata->label[6]  = "A3_com";
  pdata->label[7]  = "A4_com";
  pdata->label[8]  = "Rcom";
  pdata->label[9]  = "com_x";
  pdata->label[10] = "com_y";
  pdata->label[11] = "com_z";
  pdata->label[12] = "r_q50";
  pdata->label[13] = "sigma_r";
  pdata->label[14] = "sigma_t";
  pdata->label[15] = "alpha_min";
  pdata->label[16] = "Ham_L2";
  pdata->label[17] = "Mom_L2";
  pdata->label[18] = "boris_nfail";
  pdata->label[19] = "N_nonfinite";

  PlummerParticleHealth H;
  switch (pm->mb_indcs.ng) {
    case 2: H = PlummerParticleDiagnostics<2>(pm, pm->time, pm->ncycle, true); break;
    case 3: H = PlummerParticleDiagnostics<3>(pm, pm->time, pm->ncycle, true); break;
    case 4: H = PlummerParticleDiagnostics<4>(pm, pm->time, pm->ncycle, true); break;
    default: Fatal("nr_pic_plummer diagnostics support nghost=2,3,4.");
  }
  PlummerFieldHealth F = MeasurePlummerFieldHealth(pm);

  const Real nalive = (plummer_particle_mass > 0.0)
                    ? H.mass_total/plummer_particle_mass : 0.0;
  pdata->hdata[0]  = nalive;
  pdata->hdata[1]  = H.mass_total;
  pdata->hdata[2]  = H.energy;
  pdata->hdata[3]  = H.A1_raw;
  pdata->hdata[4]  = H.A1_com;
  pdata->hdata[5]  = H.A2_com;
  pdata->hdata[6]  = H.A3_com;
  pdata->hdata[7]  = H.A4_com;
  pdata->hdata[8]  = std::sqrt(H.com[0]*H.com[0] + H.com[1]*H.com[1]
                             + H.com[2]*H.com[2]);
  pdata->hdata[9]  = H.com[0];
  pdata->hdata[10] = H.com[1];
  pdata->hdata[11] = H.com[2];
  pdata->hdata[12] = H.r_q50;
  pdata->hdata[13] = H.sigma_r;
  pdata->hdata[14] = H.sigma_t;
  pdata->hdata[15] = F.alpha_min;
  pdata->hdata[16] = F.ham_l2;
  pdata->hdata[17] = F.mom_l2;
  pdata->hdata[18] = static_cast<Real>(pm->pmb_pack->ppart->boris_nfail_cum);
  pdata->hdata[19] = H.nonfinite;
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  user_hist_func = PlummerClusterHistory;

  if (pmbp->padm == nullptr) {
    Fatal("nr_pic_plummer requires <adm> or <z4c> ADM variables.");
  }
  if (pmbp->ppart == nullptr) {
    Fatal("nr_pic_plummer requires a <particles> block.");
  }
  if (!pmy_mesh_->three_d) {
    Fatal("nr_pic_plummer is 3D-only.");
  }
  if (pmbp->ppart->pusher != ParticlesPusher::gr_boris) {
    Fatal("nr_pic_plummer requires <particles> pusher=gr_boris.");
  }
  const bool live = (pmbp->pz4c != nullptr);
  if (live != pmbp->ppart->feedback) {
    Fatal("Use <adm> with feedback=false for a frozen metric, or <z4c> with "
          "feedback=true for the live self-consistent evolution.");
  }

  // ---------------------------------------------------------------- parameters
  const Real M = pin->GetOrAddReal("problem", "plummer_mass", 1.0);
  const Real bscale = pin->GetOrAddReal("problem", "plummer_b", 20.0);
  const Real rt = pin->GetOrAddReal("problem", "plummer_rt", 400.0);
  const int npair = pin->GetOrAddInteger("problem", "plummer_npair", 1056768);
  const int seed = pin->GetOrAddInteger("problem", "plummer_seed", 1985);
  const int ntable = pin->GetOrAddInteger("problem", "plummer_ntable", 20000);
  const int nmetric = pin->GetOrAddInteger("problem", "plummer_nmetric", 16384);
  plummer_center[0] = pin->GetOrAddReal("problem", "plummer_center_x1", 0.0);
  plummer_center[1] = pin->GetOrAddReal("problem", "plummer_center_x2", 0.0);
  plummer_center[2] = pin->GetOrAddReal("problem", "plummer_center_x3", 0.0);
  plummer_shell_nbin = pin->GetOrAddInteger("problem", "plummer_shell_nbin", 48);
  plummer_shell_rmin = pin->GetOrAddReal("problem", "plummer_shell_rmin", 0.5);
  plummer_shell_rmax = pin->GetOrAddReal("problem", "plummer_shell_rmax", 800.0);
  plummer_cohort_nbin = pin->GetOrAddInteger("problem", "plummer_cohort_nbin", 32);
  plummer_lmax = pin->GetOrAddInteger("problem", "plummer_lmax", 4);
  const std::string basename = pin->GetString("job", "basename");
  plummer_shell_fname = basename + ".plummer_shells.csv";
  plummer_cohort_fname = basename + ".plummer_cohorts.csv";

  if (M <= 0.0 || bscale <= 0.0 || rt <= 3.0*M || npair <= 0) {
    Fatal("Require plummer_mass>0, plummer_b>0, plummer_rt>3M, plummer_npair>0.");
  }
  if (plummer_lmax < 0 || plummer_lmax > 4) {
    Fatal("plummer_lmax must be in 0..4 (the Cartesian Y_lm table stops at l=4).");
  }
  const std::int64_t ntotal64 = 2LL*npair;
  if (ntotal64 > INT_MAX) {
    Fatal("2*plummer_npair exceeds the 32-bit particle-tag range.");
  }
  if (ntotal64 > (1LL << 24)) {
    std::cout << "### WARNING in " << __FILE__ << std::endl
              << "N = " << ntotal64 << " exceeds 2^24: particle tags are stored as "
              << "float32 in pvtk output and are no longer exact." << std::endl;
  }
  const int ntotal = static_cast<int>(ntotal64);

  // ------------------------------------------------- 1D continuum construction
  plummer::PlummerProfile prof(M, bscale, rt, ntable);
  if (prof.rt <= 0.0 || !(prof.M0 > 0.0)) { Fatal("Plummer profile construction failed."); }
  const Real mu = prof.M0/static_cast<Real>(ntotal);
  const Real rhalf = prof.HalfMassRadius();
  const Real Phalf = prof.HalfMassPeriod();
  plummer_particle_mass = mu;
  plummer_M0 = prof.M0;
  plummer_MADM = M;
  plummer_bscale = bscale;
  plummer_rt = rt;
  plummer_Rt = prof.Rt;
  plummer_rhalf = rhalf;
  plummer_Phalf = Phalf;
  plummer_npair = npair;
  plummer_ntotal = ntotal;

  // circular orbits must exist everywhere particles can be placed
  {
    Real worst_mr = 0.0, worst_stab = std::numeric_limits<Real>::max();
    for (int k = 1; k <= 4000; ++k) {
      const Real r = rt*static_cast<Real>(k)/4000.0;
      worst_mr = std::max(worst_mr, prof.m(r)/r);
      worst_stab = std::min(worst_stab, prof.RadialStability(r));
    }
    if (worst_mr >= 1.0/3.0) {
      Fatal("m/r >= 1/3 somewhere inside r_t: circular geodesics do not exist there.");
    }
    if (global_variable::my_rank == 0) {
      std::cout << "nr_pic_plummer: max m/r = " << worst_mr
                << " (< 1/3 required), min(r^2 m' + r m - 6 m^2) = " << worst_stab
                << " (> 0 => individually radially stable circular orbits)" << std::endl;
    }
  }

  // ---------------------------------------------- device metric lookup table
  const Real R0tab = pin->GetOrAddReal("problem", "plummer_metric_R0", 0.5);
  const Real utop = std::log1p(prof.Rt/R0tab);
  const Real du = utop/static_cast<Real>(nmetric);
  DualArray2D<Real> mtab("plummer metric table", MT_NROW, nmetric + 1);
  for (int k = 0; k <= nmetric; ++k) {
    const Real Rk = (k == nmetric) ? prof.Rt : R0tab*std::expm1(k*du);
    const Real rk = prof.AreaFromIso(Rk);
    const Real jk = (rk > 0.0) ? prof.j(rk) : prof.j_tab[0];
    const Real Pk = (rk > 0.0) ? prof.Phi(rk) : prof.Phi_tab[0];
    const Real Bk = prof.B(std::max(rk, 1.0e-30));
    const Real jpk = prof.djdr(rk);
    const Real Ppk = prof.dPhidr(rk);
    const Real psik = std::exp(-0.5*jk);
    const Real alphak = std::exp(Pk);
    mtab.h_view(MT_R, k) = Rk;
    mtab.h_view(MT_PSI, k) = psik;
    // dpsi/dR = (dpsi/dr)/(dR/dr),  dR/dr = B e^{j}
    mtab.h_view(MT_DPSI, k) = -0.5*jpk*psik/(Bk*std::exp(jk));
    mtab.h_view(MT_ALPHA, k) = alphak;
    mtab.h_view(MT_DALPHA, k) = Ppk*alphak/(Bk*std::exp(jk));
  }
  mtab.template modify<HostMemSpace>();
  mtab.template sync<DevExeSpace>();

  auto &indcs = pmbp->pmesh->mb_indcs;
  auto SeedSnapshots = [&]() {
    Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->adm_last, pmbp->padm->u_adm);
    if (live) {
      Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->z4c_last, pmbp->pz4c->u0);
    }
  };

  // A live restart restores metric and particles from the restart file.  A frozen-metric
  // restart refreshes the analytic ADM field and leaves the restored particles alone.
  const bool initialize_metric = (!restart || !live);
  if (initialize_metric) {
    auto &size = pmbp->pmb->mb_size;
    auto &adm = pmbp->padm->adm;
    const int is = indcs.is, js = indcs.js, ks = indcs.ks;
    const int isg = is - indcs.ng, ieg = indcs.ie + indcs.ng;
    const int jsg = js - indcs.ng, jeg = indcs.je + indcs.ng;
    const int ksg = ks - indcs.ng, keg = indcs.ke + indcs.ng;
    const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    const int nmb = pmbp->nmb_thispack;
    const Real cx = plummer_center[0], cy = plummer_center[1], cz = plummer_center[2];
    const Real Rt = prof.Rt, Mass = M, R0 = R0tab, inv_du = 1.0/du;
    const int ntab = nmetric;
    auto tabd = mtab.d_view;

    par_for("pgen plummer metric", DevExeSpace(), 0, nmb-1,
            ksg, keg, jsg, jeg, isg, ieg,
    KOKKOS_LAMBDA(const int mb, const int k, const int j, const int i) {
      const Real x = CellCenterX(i-is, nx1, size.d_view(mb).x1min,
                                 size.d_view(mb).x1max) - cx;
      const Real y = CellCenterX(j-js, nx2, size.d_view(mb).x2min,
                                 size.d_view(mb).x2max) - cy;
      const Real z = CellCenterX(k-ks, nx3, size.d_view(mb).x3min,
                                 size.d_view(mb).x3max) - cz;
      const Real R = Kokkos::sqrt(x*x + y*y + z*z);
      Real psi = 1.0, alpha = 1.0;
      PlummerMetricAt(tabd, ntab, R0, inv_du, Rt, Mass, R, &psi, &alpha);
      const Real psi2 = psi*psi;
      const Real psi4 = psi2*psi2;
      adm.psi4(mb,k,j,i) = psi4;
      adm.alpha(mb,k,j,i) = alpha;
      for (int a = 0; a < 3; ++a) {
        adm.beta_u(mb,a,k,j,i) = 0.0;
        for (int b = a; b < 3; ++b) {
          adm.g_dd(mb,a,b,k,j,i) = (a == b) ? psi4 : 0.0;
          adm.vK_dd(mb,a,b,k,j,i) = 0.0;    // K_ij = 0 on this slice
        }
      }
    });
    Kokkos::fence();

    if (live && !restart) {
      switch (indcs.ng) {
        case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin); break;
        case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin); break;
        case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin); break;
        default: Fatal("nr_pic_plummer supports nghost=2,3,4.");
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

  const std::string init = pin->GetOrAddString("particles", "init", "ppc");
  if (init.compare("pgen") != 0) {
    Fatal("nr_pic_plummer requires <particles> init=pgen.");
  }

  // ------------------------------------------------------------- particle sampler
  particles::Particles *ppart = pmbp->ppart;
  PrtclStage stage;
  const std::uint64_t seed64 =
      static_cast<std::uint64_t>(static_cast<std::uint32_t>(seed));
  Real gp[3] = {0.0, 0.0, 0.0};      // Sum mu u_i
  Real gj[3] = {0.0, 0.0, 0.0};      // Sum mu (x x u)
  Real gscalarL = 0.0, genergy = 0.0;
  Real max_uerr = 0.0, max_tangent = 0.0;
  Real rmin_sample = std::numeric_limits<Real>::max(), rmax_sample = 0.0;

  for (int k = 0; k < npair; ++k) {
    const std::uint64_t kid = static_cast<std::uint64_t>(k);
    const Real q = (static_cast<Real>(k) + HashUnitId(seed64, kid, STREAM_XI))
                 /static_cast<Real>(npair);
    const Real r = prof.InvertF0(q);
    const Real Riso = prof.Riso(r);
    const Real psi = prof.psi(r);
    const Real Wlor = prof.W(r);
    const Real vc = std::sqrt(prof.vc2(r));
    const Real alpha = prof.alpha(r);
    const Real umag = psi*psi*Wlor*vc;
    rmin_sample = std::min(rmin_sample, r);
    rmax_sample = std::max(rmax_sample, r);

    const Real z = 2.0*HashUnitId(seed64, kid, STREAM_Z) - 1.0;
    const Real phi = 2.0*M_PI*HashUnitId(seed64, kid, STREAM_PHI);
    const Real zeta = 2.0*M_PI*HashUnitId(seed64, kid, STREAM_ZETA);
    const Real st = std::sqrt(std::max(1.0 - z*z, 0.0));
    const Real n[3] = {st*std::cos(phi), st*std::sin(phi), z};

    // orthonormal tangent basis from the Cartesian axis LEAST aligned with n
    int axis = 0;
    for (int a = 1; a < 3; ++a) {
      if (std::fabs(n[a]) < std::fabs(n[axis])) { axis = a; }
    }
    Real khat[3] = {0.0, 0.0, 0.0};
    khat[axis] = 1.0;
    Real e1[3] = {khat[1]*n[2] - khat[2]*n[1],
                  khat[2]*n[0] - khat[0]*n[2],
                  khat[0]*n[1] - khat[1]*n[0]};
    const Real e1n = std::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
    for (int a = 0; a < 3; ++a) { e1[a] /= e1n; }
    const Real e2[3] = {n[1]*e1[2] - n[2]*e1[1],
                        n[2]*e1[0] - n[0]*e1[2],
                        n[0]*e1[1] - n[1]*e1[0]};
    const Real cz_ = std::cos(zeta), sz_ = std::sin(zeta);
    const Real tvec[3] = {cz_*e1[0] + sz_*e2[0],
                          cz_*e1[1] + sz_*e2[1],
                          cz_*e1[2] + sz_*e2[2]};

    const Real pos[3] = {plummer_center[0] + Riso*n[0],
                         plummer_center[1] + Riso*n[1],
                         plummer_center[2] + Riso*n[2]};
    // construction checks (host, exact):  gamma^ij u_i u_j = W^2 - 1  and  x.u = 0
    {
      const Real u2 = umag*umag/(psi*psi*psi*psi);
      const Real target = Wlor*Wlor - 1.0;
      max_uerr = std::max(max_uerr, std::fabs(u2 - target)
                                    /std::max(target, 1.0e-300));
      max_tangent = std::max(max_tangent, std::fabs(n[0]*tvec[0] + n[1]*tvec[1]
                                                  + n[2]*tvec[2]));
    }
    for (int s = 0; s < 2; ++s) {
      const Real sign = (s == 0) ? 1.0 : -1.0;
      const Real vel[3] = {sign*umag*tvec[0], sign*umag*tvec[1], sign*umag*tvec[2]};
      const int tag = 2*k + s;
      const int mb = ppart->FindContainingMeshBlock(pos[0], pos[1], pos[2]);
      if (mb >= 0) {
        stage.Add(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2],
                  pmbp->gids + mb, tag);
      }
      const Real rel[3] = {pos[0] - plummer_center[0], pos[1] - plummer_center[1],
                           pos[2] - plummer_center[2]};
      const Real lv[3] = {rel[1]*vel[2] - rel[2]*vel[1],
                          rel[2]*vel[0] - rel[0]*vel[2],
                          rel[0]*vel[1] - rel[1]*vel[0]};
      for (int a = 0; a < 3; ++a) { gp[a] += mu*vel[a]; gj[a] += mu*lv[a]; }
      gscalarL += mu*std::sqrt(lv[0]*lv[0] + lv[1]*lv[1] + lv[2]*lv[2]);
      genergy += mu*alpha*Wlor;
    }
  }

  // ------------------------------------------------------------- device transfer
  const int nlocal = static_cast<int>(stage.x.size());
  Kokkos::realloc(ppart->prtcl_rdata, ppart->nrdata, nlocal);
  Kokkos::realloc(ppart->prtcl_idata, ppart->nidata, nlocal);
  auto hr = Kokkos::create_mirror_view(ppart->prtcl_rdata);
  auto hi = Kokkos::create_mirror_view(ppart->prtcl_idata);
  for (int p = 0; p < nlocal; ++p) {
    hi(PGID, p) = stage.gid[p];
    hi(PTAG, p) = stage.tag[p];
    hr(IPM, p)  = mu;
    hr(IPEN, p) = 0.0;
    hr(IPX, p)  = stage.x[p];   hr(IPVX, p) = stage.ux[p];
    hr(IPY, p)  = stage.y[p];   hr(IPVY, p) = stage.uy[p];
    hr(IPZ, p)  = stage.z[p];   hr(IPVZ, p) = stage.uz[p];
  }
  Kokkos::deep_copy(ppart->prtcl_rdata, hr);
  Kokkos::deep_copy(ppart->prtcl_idata, hi);
  ppart->nprtcl_thispack = nlocal;
  ppart->mass = mu;
  pmy_mesh_->nprtcl_thisrank = nlocal;
  pmy_mesh_->nprtcl_eachrank[global_variable::my_rank] = nlocal;
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&nlocal, 1, MPI_INT, pmy_mesh_->nprtcl_eachrank, 1, MPI_INT,
                MPI_COMM_WORLD);
#endif
  pmy_mesh_->nprtcl_total = 0;
  for (int nr = 0; nr < global_variable::nranks; ++nr) {
    pmy_mesh_->nprtcl_total += pmy_mesh_->nprtcl_eachrank[nr];
  }
  if (pmy_mesh_->nprtcl_total != ntotal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
              << "placed " << pmy_mesh_->nprtcl_total << " particles but expected "
              << ntotal << ": some sampled positions fell outside every MeshBlock."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  SeedSnapshots();

  if (global_variable::my_rank == 0) {
    const Real pabs = std::sqrt(gp[0]*gp[0] + gp[1]*gp[1] + gp[2]*gp[2]);
    const Real jabs = std::sqrt(gj[0]*gj[0] + gj[1]*gj[1] + gj[2]*gj[2]);
    std::cout << std::setprecision(12)
      << "=================== nr_pic_plummer initial data ===================\n"
      << "  M_ADM              = " << M << "\n"
      << "  b                  = " << bscale << "\n"
      << "  r_t                = " << rt << " = " << rt/bscale << " b\n"
      << "  f_t                = " << prof.ft << "  (removed mass fraction "
      << (1.0 - prof.ft) << ")\n"
      << "  M_P = M/f_t        = " << prof.MP << "\n"
      << "  M_0 (rest mass)    = " << prof.M0 << "   M_0/M = " << prof.M0/M << "\n"
      << "  mu = M_0/N         = " << mu << "\n"
      << "  N, N_pair          = " << ntotal << ", " << npair << "\n"
      << "  R_t (isotropic)    = " << prof.Rt << "\n"
      << "  r_1/2              = " << rhalf << "\n"
      << "  R_1/2              = " << prof.Riso(rhalf) << "\n"
      << "  alpha(r_1/2)       = " << prof.alpha(rhalf) << "\n"
      << "  v_c(r_1/2)         = " << std::sqrt(prof.vc2(rhalf)) << "\n"
      << "  P_1/2              = " << Phalf << "   3 P_1/2 = " << 3.0*Phalf << "\n"
      << "  alpha(0), psi(0)   = " << prof.alpha(0.0) << ", "
      << std::exp(-0.5*prof.j_tab[0]) << "\n"
      << "  sampled r range    = [" << rmin_sample << ", " << rmax_sample << "]\n"
      << "  max rel err in gamma^ij u_i u_j - (W^2-1) = " << max_uerr << "\n"
      << "  max |nhat . that| (tangency)              = " << max_tangent << "\n"
      << "  |Sum mu u_i|       = " << pabs << "\n"
      << "  |Sum mu (x x u)|   = " << jabs << "\n"
      << "  Sum mu |L|         = " << gscalarL << "\n"
      << "  Sum mu alpha W     = " << genergy << "\n"
      << "  A_l shot floor     = " << 1.0/std::sqrt(static_cast<Real>(npair))
      << "  (N_pair^-1/2; co-located pairs, NOT N^-1/2)\n"
      << "  seed               = " << seed << "  (stratified antithetic sampler)\n"
      << "===================================================================\n"
      << std::flush;
  }
  return;
}
