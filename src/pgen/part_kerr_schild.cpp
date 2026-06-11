//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file part_kerr_schild.cpp
//! \brief particles on an analytic Cartesian Kerr-Schild ADM background (M=1, spin from
//! <coord> a; <coord> and <adm> blocks required). Two init paths:
//!   <particles> init=file (NRPIC Stage 1/2): load particles from an HDF5 table -- the
//!     single-particle geodesic validation runs (gen_geodesic_orbit.py ICs);
//!   <particles> init=pgen (Stage 3c): in-pgen deterministic ensembles for the horizon-
//!     capture tests, built from <problem> keys:
//!       shell_np/shell_r0 : rest shell (u_i = 0) at KS radius r0 on a Fibonacci sphere
//!                           (zero angular momentum -> every particle plunges; in Kerr
//!                           the infall is frame-dragged). Tags [0, shell_np).
//!       ring_np/ring_r0/ring_sign : equatorial circular geodesic at KS radius r0
//!                           (sign=+1 prograde, -1 retrograde), phi-equally-spaced --
//!                           the survivor population (r, E=-u_t, L=u_phi all const).
//!                           Tags [shell_np, shell_np + ring_np).
//!     Setting one count to 0 runs the other population alone (the doomed-only /
//!     survivors-only control runs of the Stage-3c(c) campaign). Constant-r surfaces in
//!     Cartesian KS are the ellipsoids x^2+y^2 = (r^2+a^2) sin^2(th), z = r cos(th),
//!     which the placement uses exactly; the circular-orbit 4-velocity construction
//!     mirrors scripts/particles/gen_geodesic_orbit.py (Omega = +-sqrt(M)/(r^1.5 +-
//!     a sqrt(M)), normalized with the local CKS metric, indices lowered to u_i).
//! On restart the particles are restored by the restart reader; this pgen then only
//! refreshes the analytic metric (incl. ghost zones) and re-seeds the GR-pusher
//! previous-step snapshots before evolution continues.

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "particles/particles.hpp"
#include "pgen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

// Cartesian Kerr-Schild 4-metric g_{mu nu} (covariant) and KS radius at (x,y,z); M=1
void CKSMetric(Real x, Real y, Real z, Real a, Real g[4][4], Real *r_out) {
  Real rho2 = x*x + y*y + z*z;
  Real r2 = 0.5*((rho2 - a*a) + std::sqrt(SQR(rho2 - a*a) + 4.0*a*a*z*z));
  Real r = std::sqrt(r2);
  Real H = r2*r/(r2*r2 + a*a*z*z);          // M r^3 / (r^4 + a^2 z^2), M = 1
  Real l[4] = {1.0, (r*x + a*y)/(r2 + a*a), (r*y - a*x)/(r2 + a*a), z/r};
  for (int mu=0; mu<4; ++mu) {
    for (int nu=0; nu<4; ++nu) {
      Real eta = (mu == nu) ? ((mu == 0) ? -1.0 : 1.0) : 0.0;
      g[mu][nu] = eta + 2.0*H*l[mu]*l[nu];
    }
  }
  *r_out = r;
}

// staged particle data accumulated on the host before the device fill
struct PrtclStage {
  std::vector<Real> x, y, z, vx, vy, vz;
  std::vector<int> gid, tag;
  void Add(Real x_, Real y_, Real z_, Real ux_, Real uy_, Real uz_, int gid_, int tag_) {
    x.push_back(x_); y.push_back(y_); z.push_back(z_);
    vx.push_back(ux_); vy.push_back(uy_); vz.push_back(uz_);
    gid.push_back(gid_); tag.push_back(tag_);
  }
};

} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->padm == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "part_kerr_schild requires an <adm> block (analytic ADM background)."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->ppart == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "part_kerr_schild requires a <particles> block (init=file or init=pgen)."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // Populate the ADM variables (analytic Kerr-Schild incl. ghost zones; spin from
  // <coord> a, Minkowski flag from <coord> minkowski, M=1). Safe and exact on restart
  // too -- the background is static-analytic, so this reproduces the restored values.
  pmbp->padm->SetADMVariables(pmbp);

  // Seed the GR Boris pusher's previous-step metric snapshot with this (static)
  // background so the first geodesic substep has a valid step-n metric. Required on
  // BOTH fresh starts and restarts (the snapshots are not part of the restart file).
  if (pmbp->ppart->pusher == ParticlesPusher::gr_boris) {
    Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->adm_last, pmbp->padm->u_adm);
    if (pmbp->pz4c != nullptr) {
      Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->z4c_last, pmbp->pz4c->u0);
    }
  }

  if (restart) return;   // particles themselves were restored by the restart reader

  std::string init = pin->GetOrAddString("particles","init","ppc");
  if (init.compare("file") == 0) {
    return;   // particles already loaded by the HDF5 reader (Stage-1/2 path)
  }
  if (init.compare("pgen") != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "part_kerr_schild requires <particles> init = file or pgen" << std::endl;
    exit(EXIT_FAILURE);
  }

  // ---- init=pgen ensembles (Stage 3c horizon-capture tests) ----
  particles::Particles *ppart = pmbp->ppart;
  Mesh *pm = pmy_mesh_;
  bool three_d = pm->three_d;
  if (!three_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "part_kerr_schild init=pgen ensembles are 3D-only" << std::endl;
    exit(EXIT_FAILURE);
  }
  Real a = pmbp->pcoord->coord_data.bh_spin;

  int shell_np  = pin->GetOrAddInteger("problem","shell_np",0);
  Real shell_r0 = pin->GetOrAddReal("problem","shell_r0",5.0);
  int ring_np   = pin->GetOrAddInteger("problem","ring_np",0);
  Real ring_r0  = pin->GetOrAddReal("problem","ring_r0",8.0);
  Real ring_sgn = pin->GetOrAddReal("problem","ring_sign",1.0);
  // first ring tag: default = shell_np (contiguous). A survivors-only control run
  // (shell_np=0) must pass the MIXED run's shell_np here so the ring tags match and
  // per-tag comparisons work across the two runs.
  int ring_tag0 = pin->GetOrAddInteger("problem","ring_tag0",shell_np);
  if (shell_np + ring_np <= 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "init=pgen needs <problem> shell_np and/or ring_np > 0" << std::endl;
    exit(EXIT_FAILURE);
  }

  PrtclStage st;

  // rest shell at KS radius shell_r0: Fibonacci-sphere directions (deterministic in the
  // tag, decomposition-invariant), placed on the exact constant-r ellipsoid; u_i = 0
  // (normal-observer rest: E = -u_t = alpha(r0), zero angular momentum -> plunge)
  const Real golden = 0.5*(std::sqrt(5.0) - 1.0);   // golden-ratio conjugate
  Real rcyl0 = std::sqrt(shell_r0*shell_r0 + a*a);
  for (int k=0; k<shell_np; ++k) {
    Real cth = 1.0 - 2.0*(k + 0.5)/static_cast<Real>(shell_np);
    Real sth = std::sqrt(1.0 - cth*cth);
    Real phi = 2.0*M_PI*std::fmod(golden*k, 1.0);
    Real px = rcyl0*sth*std::cos(phi);
    Real py = rcyl0*sth*std::sin(phi);
    Real pz = shell_r0*cth;
    int m = ppart->FindContainingMeshBlock(px, py, pz);
    if (m >= 0) {st.Add(px, py, pz, 0.0, 0.0, 0.0, pmbp->gids + m, k);}
  }

  // equatorial circular geodesic ring at KS radius ring_r0 (gen_geodesic_orbit.py
  // construction at azimuth phi): u^mu ~ (1, -x0 Omega sin phi, x0 Omega cos phi, 0),
  // normalized with the local metric, then lowered to the covariant u_i the pushers use
  Real x0 = std::sqrt(ring_r0*ring_r0 + a*a);
  Real Omega = ring_sgn/(std::pow(ring_r0, 1.5) + ring_sgn*a);   // sqrt(M) = 1
  for (int k=0; k<ring_np; ++k) {
    Real phi = 2.0*M_PI*(k + 0.5)/static_cast<Real>(ring_np);
    Real cph = std::cos(phi), sph = std::sin(phi);
    Real px = x0*cph, py = x0*sph, pz = 0.0;
    Real g[4][4], rks;
    CKSMetric(px, py, pz, a, g, &rks);
    Real udir[4] = {1.0, -x0*Omega*sph, x0*Omega*cph, 0.0};
    Real norm = 0.0;
    for (int mu=0; mu<4; ++mu) {
      for (int nu=0; nu<4; ++nu) {norm -= udir[mu]*g[mu][nu]*udir[nu];}
    }
    if (norm <= 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "no timelike circular orbit at ring_r0=" << ring_r0
                << ", a=" << a << " (norm=" << norm << "); use a larger ring_r0"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    Real ut = 1.0/std::sqrt(norm);
    Real u_dn[3];
    for (int i=1; i<4; ++i) {
      Real s = 0.0;
      for (int mu=0; mu<4; ++mu) {s += g[i][mu]*ut*udir[mu];}
      u_dn[i-1] = s;
    }
    int m = ppart->FindContainingMeshBlock(px, py, pz);
    if (m >= 0) {
      st.Add(px, py, pz, u_dn[0], u_dn[1], u_dn[2], pmbp->gids + m, ring_tag0 + k);
    }
  }

  // fill the particle arrays from the staged host data (part_crossing pattern)
  int npart = static_cast<int>(st.x.size());
  Kokkos::realloc(ppart->prtcl_rdata, ppart->nrdata, npart);
  Kokkos::realloc(ppart->prtcl_idata, ppart->nidata, npart);
  auto hr = Kokkos::create_mirror_view(ppart->prtcl_rdata);
  auto hi = Kokkos::create_mirror_view(ppart->prtcl_idata);
  for (int p=0; p<npart; ++p) {
    hi(PGID,p) = st.gid[p];
    hi(PTAG,p) = st.tag[p];
    hr(IPM,p)  = ppart->mass;
    hr(IPEN,p) = 0.0;
    hr(IPX,p)  = st.x[p];
    hr(IPVX,p) = st.vx[p];
    hr(IPY,p)  = st.y[p];
    hr(IPVY,p) = st.vy[p];
    hr(IPZ,p)  = st.z[p];
    hr(IPVZ,p) = st.vz[p];
  }
  Kokkos::deep_copy(ppart->prtcl_rdata, hr);
  Kokkos::deep_copy(ppart->prtcl_idata, hi);
  ppart->nprtcl_thispack = npart;

  // refresh the Mesh particle counts (AddCoordinatesAndPhysics counted zero particles
  // before this pgen ran; mirror its logic, cf. mesh.cpp)
  pm->nprtcl_thisrank = npart;
  pm->nprtcl_eachrank[global_variable::my_rank] = npart;
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&npart, 1, MPI_INT, pm->nprtcl_eachrank, 1, MPI_INT, MPI_COMM_WORLD);
#endif
  pm->nprtcl_total = 0;
  for (int n=0; n<global_variable::nranks; ++n) {
    pm->nprtcl_total += pm->nprtcl_eachrank[n];
  }

  if (global_variable::my_rank == 0) {
    std::cout << "part_kerr_schild: placed " << pm->nprtcl_total << " particles (shell "
              << shell_np << " at r0=" << shell_r0 << ", ring " << ring_np << " at r0="
              << ring_r0 << ", a=" << a << ")" << std::endl;
  }

  // block map for the analysis scripts (gid, owning rank, level, parity, bbox), printed
  // rank by rank behind barriers (best effort; parsers should also use --tag-output)
  auto &mbsize = pmbp->pmb->mb_size;
  auto &mblev = pmbp->pmb->mb_lev;
  int nmb = pmbp->nmb_thispack;
  int gids = pmbp->gids;
  for (int r=0; r<global_variable::nranks; ++r) {
#if MPI_PARALLEL_ENABLED
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    if (r != global_variable::my_rank) {continue;}
    for (int m=0; m<nmb; ++m) {
      int gid = gids + m;
      auto &lloc = pm->lloc_eachmb[gid];
      std::cout << "[part_kerr_schild] block gid=" << gid << " rank=" << r
                << " level=" << mblev.h_view(m)
                << " parity=(" << (lloc.lx1 & 1) << "," << (lloc.lx2 & 1) << ","
                << (lloc.lx3 & 1) << ")"
                << " x1=[" << mbsize.h_view(m).x1min << "," << mbsize.h_view(m).x1max
                << ") x2=[" << mbsize.h_view(m).x2min << "," << mbsize.h_view(m).x2max
                << ") x3=[" << mbsize.h_view(m).x3min << "," << mbsize.h_view(m).x3max
                << ")" << std::endl << std::flush;
    }
  }
  return;
}
