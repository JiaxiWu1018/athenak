//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file uni_mag_part.cpp
//! \brief Problem generator for the particle moving in a uniform magnetic field

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "particles/particles.hpp"

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if ((pmbp->pmhd == nullptr) || (pmbp->ppart == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "particle test needs to have <mhd> block and <particle> block "
              << "in input file" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int isg = is-indcs.ng; int ieg = ie+indcs.ng;
  int jsg = js-indcs.ng; int jeg = je+indcs.ng;
  int ksg = ks-indcs.ng; int keg = ke+indcs.ng;
  int nmb = pmbp->nmb_thispack;

  auto &w0_ = pmbp->pmhd->w0;
  Real dfloor = pmbp->pmhd->peos->eos_data.dfloor;
  Real pfloor = pmbp->pmhd->peos->eos_data.pfloor;
  Real gm1 = pmbp->pmhd->peos->eos_data.gamma - 1.0;
  Real fluid_energy = pmbp->pcoord->is_dynamical_relativistic ? pfloor : pfloor/gm1;

  par_for("pgen_boris_hyd", DevExeSpace(), 0, nmb-1, ksg, keg, jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    w0_(m, IDN, k, j, i) = dfloor;
    w0_(m, IVX, k, j, i) = 0.0;
    w0_(m, IVY, k, j, i) = 0.0;
    w0_(m, IVZ, k, j, i) = 0.0;
    w0_(m, IEN, k, j, i) = fluid_energy;
  });

  auto &b0_ = pmbp->pmhd->b0;
  auto &bcc_ = pmbp->pmhd->bcc0;
  Real bz = pin->GetReal("problem", "Bz");
  par_for("pgen_boris_mag", DevExeSpace(), 0, nmb-1, ksg, keg, jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    b0_.x1f(m, k, j, i) = 0.0;
    b0_.x2f(m, k, j, i) = 0.0;
    b0_.x3f(m, k, j, i) = bz;
    bcc_(m, IBX, k, j, i) = 0.0;
    bcc_(m, IBY, k, j, i) = 0.0;
    bcc_(m, IBZ, k, j, i) = bz;
  });

  if (pmbp->padm != nullptr) {
    pmbp->padm->SetADMVariables(pmbp);
  }

  if (pmbp->ppart->pusher == ParticlesPusher::gr_boris ||
      pmbp->ppart->pusher == ParticlesPusher::geo_boris ||
      pmbp->ppart->pusher == ParticlesPusher::geo_boris_fw) {
    Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->w0_last, w0_);
    Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->bcc0_last, bcc_);
    Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->adm_last, pmbp->padm->u_adm);
  }

  if (pmbp->pcoord->is_dynamical_relativistic) {
    pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
  } else {
    pmbp->pmhd->peos->PrimToCons(w0_, bcc_, pmbp->pmhd->u0, is, ie, js, je, ks, ke);
  }

  return;
}
