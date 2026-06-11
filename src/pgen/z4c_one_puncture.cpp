//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_one_puncture.cpp
//  \brief Problem generator for a single puncture placed at the origin of the domain

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <memory>
#include <string>     // c_str(), string
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "particles/particles.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

void ADMOnePuncture(MeshBlockPack *pmbp, ParameterInput *pin);
void RefinementCondition(MeshBlockPack* pmbp);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for single puncture
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_ref_func  = RefinementCondition;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "One Puncture test can only be run in Z4c, but no <z4c> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  ADMOnePuncture(pmbp, pin);
  pmbp->pz4c->GaugePreCollapsedLapse(pmbp, pin);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
            break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
            break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
            break;
  }
  pmbp->pz4c->Z4cToADM(pmbp);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMConstraints<2>(pmbp);
            break;
    case 3: pmbp->pz4c->ADMConstraints<3>(pmbp);
            break;
    case 4: pmbp->pz4c->ADMConstraints<4>(pmbp);
            break;
  }
  // ---- NRPIC Stage 3c(b): optional particle rings for the puncture-lapse excision
  // smoke test (active only when the input has a <particles> block with init=pgen).
  // Up to two equatorial rest rings (u_i = 0) of <problem> prtcl_np particles at radii
  // prtcl_r1 / prtcl_r2 (0 = off; tags ring-1 first). With the pre-collapsed initial
  // lapse (alpha = psi^-2) and 1+log slicing, alpha at an inner ring (r ~ 0.5M) falls
  // below a <particles> excise_lapse ~ 0.1 threshold within a few M of evolution -- a
  // DYNAMICAL-lapse kill through the I_Z4C_ALPHA interpolation branch (the OS-collapse
  // rehearsal); an outer ring (r ~ 4M) survives. No effect on z4c runs without
  // particles.
  if (pmbp->ppart != nullptr && !restart) {
    particles::Particles *ppart = pmbp->ppart;
    std::string pinit = pin->GetOrAddString("particles","init","ppc");
    if (pinit.compare("pgen") != 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "z4c_one_puncture particles require init = pgen"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    int npr  = pin->GetOrAddInteger("problem","prtcl_np",16);
    Real rr[2] = {pin->GetOrAddReal("problem","prtcl_r1",0.0),
                  pin->GetOrAddReal("problem","prtcl_r2",0.0)};
    std::vector<Real> sx, sy, sz;
    std::vector<int> sgid, stag;
    int tag = 0;
    for (int ir=0; ir<2; ++ir) {
      for (int k=0; k<npr; ++k, ++tag) {
        if (rr[ir] <= 0.0) {continue;}
        Real phi = 2.0*M_PI*(k + 0.5)/static_cast<Real>(npr);
        Real px = rr[ir]*std::cos(phi), py = rr[ir]*std::sin(phi);
        int m = ppart->FindContainingMeshBlock(px, py, 0.0);
        if (m >= 0) {
          sx.push_back(px); sy.push_back(py); sz.push_back(0.0);
          sgid.push_back(pmbp->gids + m); stag.push_back(tag);
        }
      }
    }
    int npart = static_cast<int>(sx.size());
    Kokkos::realloc(ppart->prtcl_rdata, ppart->nrdata, npart);
    Kokkos::realloc(ppart->prtcl_idata, ppart->nidata, npart);
    auto hr = Kokkos::create_mirror_view(ppart->prtcl_rdata);
    auto hi = Kokkos::create_mirror_view(ppart->prtcl_idata);
    for (int p=0; p<npart; ++p) {
      hi(PGID,p) = sgid[p];
      hi(PTAG,p) = stag[p];
      hr(IPM,p)  = ppart->mass;
      hr(IPEN,p) = 0.0;
      hr(IPX,p)  = sx[p];  hr(IPVX,p) = 0.0;
      hr(IPY,p)  = sy[p];  hr(IPVY,p) = 0.0;
      hr(IPZ,p)  = sz[p];  hr(IPVZ,p) = 0.0;
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
    for (int n=0; n<global_variable::nranks; ++n) {
      pmy_mesh_->nprtcl_total += pmy_mesh_->nprtcl_eachrank[n];
    }
    if (global_variable::my_rank == 0) {
      std::cout << "z4c_one_puncture: placed " << pmy_mesh_->nprtcl_total
                << " particles (rings r1=" << rr[0] << " r2=" << rr[1] << ")"
                << std::endl;
    }
  }
  // seed the GR-pusher previous-step snapshots (fresh start AND restart; the restart
  // reader restores u_adm/u0 before this runs)
  if (pmbp->ppart != nullptr &&
      pmbp->ppart->pusher == ParticlesPusher::gr_boris) {
    Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->adm_last, pmbp->padm->u_adm);
    if (pmbp->pz4c != nullptr) {
      Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->z4c_last, pmbp->pz4c->u0);
    }
  }

  std::cout<<"OnePuncture initialized."<<std::endl;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ADMOnePuncture(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief Initialize ADM vars to single puncture (no spin)

void ADMOnePuncture(MeshBlockPack *pmbp, ParameterInput *pin) {
  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  // For GLOOPS
  int isg = is-indcs.ng; int ieg = ie+indcs.ng;
  int jsg = js-indcs.ng; int jeg = je+indcs.ng;
  int ksg = ks-indcs.ng; int keg = ke+indcs.ng;
  int nmb = pmbp->nmb_thispack;
  Real ADM_mass = pin->GetOrAddReal("problem", "punc_ADM_mass", 1.);
  Real center_x1 = pin->GetOrAddReal("problem", "punc_center_x1", 0.);
  Real center_x2 = pin->GetOrAddReal("problem", "punc_center_x2", 0.);
  Real center_x3 = pin->GetOrAddReal("problem", "punc_center_x3", 0.);

  adm::ADM::ADM_vars &adm = pmbp->padm->adm;

  par_for("pgen one puncture",
  DevExeSpace(),0,nmb-1,ksg,keg,jsg,jeg,isg,ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    x1v -= center_x1;
    x2v -= center_x2;
    x3v -= center_x3;

    Real r = std::sqrt(std::pow(x3v,2) + std::pow(x2v,2) + std::pow(x1v,2));

    // Minkowski spacetime
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      adm.g_dd(m,a,b,k,j,i) = (a == b ? 1. : 0.);
    }
    // admK_dd is automatically set to 0 when is initialized as Kokkos View

    // ADMOnePuncture
    adm.psi4(m,k,j,i) = std::pow(1.0 + 0.5*ADM_mass/r,4); // adm.psi4

    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      adm.g_dd(m,a,b,k,j,i) *= adm.psi4(m,k,j,i);
    }
  });
}

// how decide the refinement
void RefinementCondition(MeshBlockPack* pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}
