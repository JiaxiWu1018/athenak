//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file part_tmunu_test.cpp
//! \brief verification problem generator for the particle stress-energy deposition
//! (NRPIC Stage 4a). Sets an analytic ADM background -- exact Minkowski with
//! <coord> minkowski=true, else Cartesian Kerr-Schild with spin <coord> a (use a=0.0
//! for Schwarzschild) -- converts it to Z4c data (the <z4c> block is required: it is
//! the feedback consumer), and places a deterministic particle set from <problem> keys:
//!
//!   mode = single : one particle at (px, py, pz) with covariant velocity
//!                   (pux, puy, puz) and mass pmass (tag 0). Used for the per-cell
//!                   factor-discrimination tests and the level-interface guard demo.
//!   mode = sweep  : an 11^3 lattice of positions x = anchor + off_i per dimension,
//!                   where anchor = (sweep_x1, sweep_x2, sweep_x3), h = sweep_dx (the
//!                   cell width) and L = sweep_span (the distance to the far block
//!                   face). The 11 per-dim offsets probe every CIC edge case:
//!                     {0 (exact block face/corner), +ulp, 0.25h, 0.5h (exact cell
//!                      center), h (exact interior cell face), h-ulp, 1.5h (next
//!                      center), 1.5h-ulp (delta -> 1 clamp), 8h (a face mid-block),
//!                      L-0.25h (upper band), L-ulp (upper band, half-ulp inside)}
//!                   Velocities are common (pux, puy, puz); masses are distinct,
//!                   m_p = pmass*(1 + 1e-3*tag), so any weight-routing error breaks
//!                   the conservation identity rather than hiding in a degeneracy.
//!                   The anchor is given in ABSOLUTE coordinates so that runs with
//!                   different MeshBlock decompositions of the same domain place
//!                   bit-identical particles (the cross-block reference comparison).
//!
//! On restart only the GR-pusher metric snapshots are re-seeded; the evolved state and
//! the particles were restored from the restart file (never re-run initial data on
//! restart -- the z4c_one_puncture lesson).

#include <cmath>
#include <iostream>
#include <limits>
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
#include "pgen/pgen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

// staged particle data accumulated on the host before the device fill
struct PrtclStage {
  std::vector<Real> x, y, z, vx, vy, vz, mm;
  std::vector<int> gid, tag;
  void Add(Real x_, Real y_, Real z_, Real ux_, Real uy_, Real uz_, Real m_,
           int gid_, int tag_) {
    x.push_back(x_); y.push_back(y_); z.push_back(z_);
    vx.push_back(ux_); vy.push_back(uy_); vz.push_back(uz_);
    mm.push_back(m_);
    gid.push_back(gid_); tag.push_back(tag_);
  }
};

} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  if (pmbp->pz4c == nullptr || pmbp->padm == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "part_tmunu_test requires a <z4c> block (the feedback consumer; ADM"
              << std::endl << "storage comes with it)." << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->ppart == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "part_tmunu_test requires a <particles> block with init = pgen."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!pmy_mesh_->three_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "part_tmunu_test is 3D-only." << std::endl;
    exit(EXIT_FAILURE);
  }

  // GR-pusher previous-step snapshots: required on BOTH fresh starts and restarts (not
  // part of the restart file). On restart u_adm/u0 were restored -- ghosts included --
  // before this runs; on fresh starts they are seeded again after the ID is built below.
  auto SeedSnapshots = [&]() {
    if (pmbp->ppart->pusher == ParticlesPusher::gr_boris) {
      Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->adm_last, pmbp->padm->u_adm);
      Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->z4c_last, pmbp->pz4c->u0);
    }
  };
  if (restart) {
    SeedSnapshots();
    return;   // evolved state + particles restored by the restart reader
  }

  // ---- initial data: analytic ADM background (incl. ghosts and gauge; exact Minkowski
  // when <coord> minkowski = true) -> Z4c variables -> derived ADM (the one_puncture
  // sequence). SetADMVariables writes alpha/beta through the adm aliases, which target
  // the Z4c state vector when z4c is live.
  pmbp->padm->SetADMVariables(pmbp);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin); break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin); break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin); break;
  }
  pmbp->pz4c->Z4cToADM(pmbp);
  SeedSnapshots();

  // ---- particles ----
  particles::Particles *ppart = pmbp->ppart;
  std::string pinit = pin->GetOrAddString("particles","init","ppc");
  if (pinit.compare("pgen") != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "part_tmunu_test requires <particles> init = pgen" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string mode = pin->GetOrAddString("problem","mode","single");
  Real pux = pin->GetOrAddReal("problem","pux",0.0);
  Real puy = pin->GetOrAddReal("problem","puy",0.0);
  Real puz = pin->GetOrAddReal("problem","puz",0.0);
  Real pmass = pin->GetOrAddReal("problem","pmass",ppart->mass);

  PrtclStage st;
  if (mode.compare("single") == 0) {
    Real px = pin->GetOrAddReal("problem","px",0.0);
    Real py = pin->GetOrAddReal("problem","py",0.0);
    Real pz = pin->GetOrAddReal("problem","pz",0.0);
    int m = ppart->FindContainingMeshBlock(px, py, pz);
    if (m >= 0) {st.Add(px, py, pz, pux, puy, puz, pmass, pmbp->gids + m, 0);}
  } else if (mode.compare("sweep") == 0) {
    Real anc[3] = {pin->GetReal("problem","sweep_x1"),
                   pin->GetReal("problem","sweep_x2"),
                   pin->GetReal("problem","sweep_x3")};
    Real h = pin->GetReal("problem","sweep_dx");
    Real span = pin->GetReal("problem","sweep_span");
    // absolute per-dim positions (ulp offsets applied to the SUM -- adding a denormal
    // offset to the anchor would round away)
    constexpr int NOFF = 11;
    Real pos[3][NOFF];
    constexpr Real huge = std::numeric_limits<Real>::infinity();
    for (int d=0; d<3; ++d) {
      pos[d][0]  = anc[d];                                  // exact block face/corner
      pos[d][1]  = std::nextafter(anc[d], huge);            // half-ulp inside
      pos[d][2]  = anc[d] + 0.25*h;                         // lower band interior
      pos[d][3]  = anc[d] + 0.5*h;                          // exact first cell center
      pos[d][4]  = anc[d] + h;                              // exact interior cell face
      pos[d][5]  = std::nextafter(anc[d] + h, -huge);       // half-ulp below that face
      pos[d][6]  = anc[d] + 1.5*h;                          // exact second cell center
      pos[d][7]  = std::nextafter(anc[d] + 1.5*h, -huge);   // delta -> 1 clamp probe
      pos[d][8]  = anc[d] + 8.0*h;                          // a cell face mid-block
      pos[d][9]  = anc[d] + (span - 0.25*h);                // upper band interior
      pos[d][10] = std::nextafter(anc[d] + span, -huge);    // half-ulp inside far face
    }
    for (int kz=0; kz<NOFF; ++kz) {
      for (int ky=0; ky<NOFF; ++ky) {
        for (int kx=0; kx<NOFF; ++kx) {
          int tag = kx + NOFF*(ky + NOFF*kz);
          Real px = pos[0][kx];
          Real py = pos[1][ky];
          Real pz = pos[2][kz];
          int m = ppart->FindContainingMeshBlock(px, py, pz);
          if (m >= 0) {
            Real mp = pmass*(1.0 + 1.0e-3*static_cast<Real>(tag));
            st.Add(px, py, pz, pux, puy, puz, mp, pmbp->gids + m, tag);
          }
        }
      }
    }
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<problem> mode = '" << mode << "' not recognized (use single|sweep)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // fill the particle arrays from the staged host data (part_kerr_schild pattern)
  int npart = static_cast<int>(st.x.size());
  Kokkos::realloc(ppart->prtcl_rdata, ppart->nrdata, npart);
  Kokkos::realloc(ppart->prtcl_idata, ppart->nidata, npart);
  auto hr = Kokkos::create_mirror_view(ppart->prtcl_rdata);
  auto hi = Kokkos::create_mirror_view(ppart->prtcl_idata);
  for (int p=0; p<npart; ++p) {
    hi(PGID,p) = st.gid[p];
    hi(PTAG,p) = st.tag[p];
    hr(IPM,p)  = st.mm[p];
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
  // before this pgen ran)
  Mesh *pm = pmy_mesh_;
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
    std::cout << "part_tmunu_test: placed " << pm->nprtcl_total << " particles (mode "
              << mode << ", u=(" << pux << "," << puy << "," << puz << "))" << std::endl;
  }

  // block map for the analysis scripts (gid, owning rank, level, parity, bbox)
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
      std::cout << "[part_tmunu_test] block gid=" << gid << " rank=" << r
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
