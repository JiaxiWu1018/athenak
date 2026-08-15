//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file uni_mag_part.cpp
//! \brief NRPIC Stage-2 validation pgen: a single charged particle in a uniform magnetic
//! field on a flat (Minkowski) background. With the default quiescent fluid (v=0) the
//! ideal-MHD electric field vanishes and the particle executes a pure relativistic Larmor
//! orbit. With <problem> fluid_vx = v_f != 0 the uniform fluid carries the field (an exact
//! equilibrium), E = -v x B = v_f*Bz yhat, and a particle starting at rest executes a cycloid
//! whose guiding centre drifts at E x B / B^2 = v_f xhat — the ExB-drift validation of the
//! w0 = utilde = W*v convention handling in the pushers. Used to validate
//! the `boris` pusher and, as a flat-space cross-check, the `gr_boris` pusher (whose tetrad
//! reduces to the identity and whose geodesic substep is force-free here).
//! Requires <mhd>, <adm>/<coord minkowski=true>, and <particles> (use init=file). B is set by
//! <problem> Bz; the particle position/velocity come from the HDF5 IC file.

#include <cmath>
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
              << "uni_mag_part needs an <mhd> block and a <particles> block" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // capture variables for the kernels
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int isg = is-indcs.ng; int ieg = ie+indcs.ng;
  int jsg = js-indcs.ng; int jeg = je+indcs.ng;
  int ksg = ks-indcs.ng; int keg = ke+indcs.ng;
  int nmb = pmbp->nmb_thispack;

  // uniform floor fluid; default at rest (E = -v x B = 0, Larmor test), optionally moving in
  // x with Valencia velocity v_f (ExB-drift test). The velocity slots take the projected
  // 4-velocity utilde^i = W v^i, the SR-MHD/dyn_grmhd primitive convention.
  auto &w0_ = pmbp->pmhd->w0;
  Real dfloor = pmbp->pmhd->peos->eos_data.dfloor;
  Real pfloor = pmbp->pmhd->peos->eos_data.pfloor;
  Real gm1 = pmbp->pmhd->peos->eos_data.gamma - 1.0;
  Real fluid_energy = pmbp->pcoord->is_dynamical_relativistic ? pfloor : pfloor/gm1;
  Real vf = pin->GetOrAddReal("problem", "fluid_vx", 0.0);
  Real fluid_utx = vf/std::sqrt(1.0 - vf*vf);

  par_for("pgen_uni_mag_hyd", DevExeSpace(), 0, nmb-1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    w0_(m, IDN, k, j, i) = dfloor;
    w0_(m, IVX, k, j, i) = fluid_utx;
    w0_(m, IVY, k, j, i) = 0.0;
    w0_(m, IVZ, k, j, i) = 0.0;
    w0_(m, IEN, k, j, i) = fluid_energy;
  });

  // uniform B = Bz zhat (face and cell-centred)
  auto &b0_ = pmbp->pmhd->b0;
  auto &bcc_ = pmbp->pmhd->bcc0;
  Real bz = pin->GetReal("problem", "Bz");
  par_for("pgen_uni_mag_mag", DevExeSpace(), 0, nmb-1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    b0_.x1f(m, k, j, i) = 0.0;
    b0_.x2f(m, k, j, i) = 0.0;
    b0_.x3f(m, k, j, i) = bz;
    bcc_(m, IBX, k, j, i) = 0.0;
    bcc_(m, IBY, k, j, i) = 0.0;
    bcc_(m, IBZ, k, j, i) = bz;
  });

  // flat ADM metric (needed by gr_boris and the conserved-energy diagnostic)
  if (pmbp->padm != nullptr) {
    pmbp->padm->SetADMVariables(pmbp);
  }

  // seed the GR pusher's previous-step snapshots with the (static) initial state
  if (pmbp->ppart->pusher == ParticlesPusher::gr_boris) {
    Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->w0_last, w0_);
    Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->bcc0_last, bcc_);
    Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->adm_last, pmbp->padm->u_adm);
  }

  // initialize conserved MHD variables
  if (pmbp->pcoord->is_dynamical_relativistic) {
    pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
  } else {
    pmbp->pmhd->peos->PrimToCons(w0_, bcc_, pmbp->pmhd->u0, is, ie, js, je, ks, ke);
  }

  return;
}
