//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file nr_pic_os.cpp
//  \brief Problem generator for Oppenheimer-Snyder spherical dust collapse

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "particles/particles.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for single puncture
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "One Puncture test can only be run in Z4c, but no <z4c> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Set OS mass
  Real R0_over_M = pin->GetReal("problem", "os_radius_over_mass");
  Real M = pin->GetReal("problem", "os_mass");
  Real R0 = R0_over_M * M;
  Real nprtcl = pmy_mesh_->nprtcl_total;
  Real &mp = pmbp->ppart->mass;
  mp = M / nprtcl;

  if (restart) {
    // Copy ADM and Z4c variables from last time step if restarting
    Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->adm_last, pmbp->padm->u_adm);
    Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->z4c_last, pmbp->pz4c->u0);
    return;
  }

  // Initialize adm variables
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int &nx1 = indcs.nx1; int &nx2 = indcs.nx2; int &nx3 = indcs.nx3;
  // For GLOOPS
  int isg = is-indcs.ng; int ieg = ie+indcs.ng;
  int jsg = js-indcs.ng; int jeg = je+indcs.ng;
  int ksg = ks-indcs.ng; int keg = ke+indcs.ng;
  int nmb = pmbp->nmb_thispack;

  adm::ADM::ADM_vars &adm = pmbp->padm->adm;
  z4c::Z4c::Z4c_vars &z4c = pmbp->pz4c->z4c;
  par_for("pgen os puncture",
  DevExeSpace(),0,nmb-1,ksg,keg,jsg,jeg,isg,ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // compute cell center coordinates
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    Real r = std::sqrt(x1v*x1v + x2v*x2v + x3v*x3v);

    // Conformally flat metric
    Real M_over_R0 = 1.0 / R0_over_M;
    Real om2MoR0 = 1.0 - 2.0 * M_over_R0;
    Real r0 = 0.5 * R0 * (1.0 - 1.0 * M_over_R0 + std::sqrt(om2MoR0));
    Real psi = 1.0 + 0.5 * M / r;
    Real psi4 = std::pow(psi, 4);
    if (r < r0) {
      Real psi2 = ((1.0 + std::sqrt(om2MoR0)) * r0 * R0 * R0 / (2.0 * r0 * r0 * r0 + M * r * r));
      psi4 = psi2 * psi2;
    }
    // Set ADM variables
    adm.psi4(m,k,j,i) = psi4;
    for (int a = 0; a < 3; a++) {
      for (int b = a; b < 3; b++) {
        adm.g_dd(m,a,b,k,j,i) = psi4 * ((a==b) ? 1.0 : 0.0);
        adm.vK_dd(m,a,b,k,j,i) = 0.0;
      }
    }

    // Set Z4c variables
    z4c.alpha(m,k,j,i) = 1.0;
    for (int a = 0; a < 3; a++) {
      z4c.beta_u(m,a,k,j,i) = 0.0;
    }
  });

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

  // Copy ADM and Z4c variables from last time step if restarting
  Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->adm_last, pmbp->padm->u_adm);
  Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->z4c_last, pmbp->pz4c->u0);

  std::cout<<"Oppenheimer Snyder initialized."<<std::endl;

  return;
}