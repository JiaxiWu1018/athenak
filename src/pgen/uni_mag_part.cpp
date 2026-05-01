//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file uni_mag_part.cpp
//! \brief Problem generator for the particle moving in a uniform magnetic field

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "eos/primitive-solver/geom_math.hpp"
#include "particles/particles.hpp"
#include "particles/lagrange_interp.hpp"
#include "particles/calc_tetrad.hpp"

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if ((pmbp->pmhd == nullptr) || (pmbp->ppart == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "particle test needs to have <mhd> block and <particle> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (pmbp->pcoord->is_dynamical_relativistic == false) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "particle test needs to be special relativitic." << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  // For GLOOPS
  int isg = is-indcs.ng; int ieg = ie+indcs.ng;
  int jsg = js-indcs.ng; int jeg = je+indcs.ng;
  int ksg = ks-indcs.ng; int keg = ke+indcs.ng;
  int nmb = pmbp->nmb_thispack;

  auto &w0_ = pmbp->pmhd->w0;
  auto &w0_prev = pmbp->ppart->w0_last;
  Real gm1 = pmbp->pmhd->peos->eos_data.gamma - 1.0;
  Real dfloor = pmbp->pmhd->peos->eos_data.dfloor;
  Real pfloor = pmbp->pmhd->peos->eos_data.pfloor;
  // Do we need to set the ghost cells in pgen?
  par_for("pgen_boris_hyd", DevExeSpace(), 0, nmb-1, ksg, keg, jsg, jeg, isg, ieg,
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

    w0_(m, IDN, k, j, i) = dfloor;
    w0_(m, IVX, k, j, i) = 0.0;
    w0_(m, IVY, k, j, i) = 0.0;
    w0_(m, IVZ, k, j, i) = 0.0;
    w0_(m, IEN, k, j, i) = pfloor / gm1;
    w0_prev(m, IDN, k, j, i) = dfloor;
    w0_prev(m, IVX, k, j, i) = 0.0;
    w0_prev(m, IVY, k, j, i) = 0.0;
    w0_prev(m, IVZ, k, j, i) = 0.0;
    w0_prev(m, IEN, k, j, i) = pfloor / gm1;
  });

  auto &b0_ = pmbp->pmhd->b0;
  auto &bcc_ = pmbp->pmhd->bcc0;
  auto &bcc_prev = pmbp->ppart->bcc0_last;
  Real bz = pin->GetReal("problem", "Bz");
  par_for("pgen_boris_mag", DevExeSpace(), 0, nmb-1, ksg, keg, jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    b0_.x1f(m, k, j, i) = 0.0;
    b0_.x2f(m, k, j, i) = 0.0;
    b0_.x3f(m, k, j, i) = bz; // How do we set the units here?
    bcc_(m, IBX, k, j, i) = 0.0;
    bcc_(m, IBY, k, j, i) = 0.0;
    bcc_(m, IBZ, k, j, i) = bz; // Be consistent with the above b0_
    bcc_prev(m, IBX, k, j, i) = 0.0;
    bcc_prev(m, IBY, k, j, i) = 0.0;
    bcc_prev(m, IBZ, k, j, i) = bz; // Be consistent with the above b0_
  });

  pmbp->pmhd->peos->PrimToCons(w0_, bcc_, pmbp->pmhd->u0, is, ie, js, je, ks, ke);

  auto &u_adm = pmbp->padm->u_adm;
  auto &u_adm_prev = pmbp->ppart->adm_last;
  par_for("pgen_adm", DevExeSpace(), 0, nmb-1, ksg, keg, jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    u_adm(m, adm::ADM::I_ADM_ALPHA, k, j, i) = 1.0;
    u_adm(m, adm::ADM::I_ADM_BETAX, k, j, i) = 0.0;
    u_adm(m, adm::ADM::I_ADM_BETAY, k, j, i) = 0.0;
    u_adm(m, adm::ADM::I_ADM_BETAZ, k, j, i) = 0.0;
    u_adm(m, adm::ADM::I_ADM_GXX, k, j, i) = 1.0;
    u_adm(m, adm::ADM::I_ADM_GXY, k, j, i) = 0.0;
    u_adm(m, adm::ADM::I_ADM_GXZ, k, j, i) = 0.0;
    u_adm(m, adm::ADM::I_ADM_GYY, k, j, i) = 1.0;
    u_adm(m, adm::ADM::I_ADM_GYZ, k, j, i) = 0.0;
    u_adm(m, adm::ADM::I_ADM_GZZ, k, j, i) = 1.0;
    u_adm(m, adm::ADM::I_ADM_KXX, k, j, i) = 0.0;
    u_adm(m, adm::ADM::I_ADM_KXY, k, j, i) = 0.0;
    u_adm(m, adm::ADM::I_ADM_KXZ, k, j, i) = 0.0;
    u_adm(m, adm::ADM::I_ADM_KYY, k, j, i) = 0.0;
    u_adm(m, adm::ADM::I_ADM_KYZ, k, j, i) = 0.0;
    u_adm(m, adm::ADM::I_ADM_KZZ, k, j, i) = 0.0;
    u_adm(m, adm::ADM::I_ADM_PSI4, k, j, i) = 1.0;
    u_adm_prev(m, adm::ADM::I_ADM_ALPHA, k, j, i) = 1.0;
    u_adm_prev(m, adm::ADM::I_ADM_BETAX, k, j, i) = 0.0;
    u_adm_prev(m, adm::ADM::I_ADM_BETAY, k, j, i) = 0.0;
    u_adm_prev(m, adm::ADM::I_ADM_BETAZ, k, j, i) = 0.0;
    u_adm_prev(m, adm::ADM::I_ADM_GXX, k, j, i) = 1.0;
    u_adm_prev(m, adm::ADM::I_ADM_GXY, k, j, i) = 0.0;
    u_adm_prev(m, adm::ADM::I_ADM_GXZ, k, j, i) = 0.0;
    u_adm_prev(m, adm::ADM::I_ADM_GYY, k, j, i) = 1.0;
    u_adm_prev(m, adm::ADM::I_ADM_GYZ, k, j, i) = 0.0;
    u_adm_prev(m, adm::ADM::I_ADM_GZZ, k, j, i) = 1.0;
    u_adm_prev(m, adm::ADM::I_ADM_KXX, k, j, i) = 0.0;
    u_adm_prev(m, adm::ADM::I_ADM_KXY, k, j, i) = 0.0;
    u_adm_prev(m, adm::ADM::I_ADM_KXZ, k, j, i) = 0.0;
    u_adm_prev(m, adm::ADM::I_ADM_KYY, k, j, i) = 0.0;
    u_adm_prev(m, adm::ADM::I_ADM_KYZ, k, j, i) = 0.0;
    u_adm_prev(m, adm::ADM::I_ADM_KZZ, k, j, i) = 0.0;
    u_adm_prev(m, adm::ADM::I_ADM_PSI4, k, j, i) = 1.0;
  });

  // Initialize particle data
  Real ptcl_x = pin->GetReal("problem", "ptcl_x");
  Real ptcl_y = pin->GetReal("problem", "ptcl_y");
  Real ptcl_z = pin->GetReal("problem", "ptcl_z");
  Real ptcl_ux = pin->GetReal("problem", "ptcl_ux");
  Real ptcl_uy = pin->GetReal("problem", "ptcl_uy");
  Real ptcl_uz = pin->GetReal("problem", "ptcl_uz");

  // Find prtcl rank
  int gids = pmbp->gids;
  int nptcl_fnd = 0, ptcl_m = -1;
  Kokkos::parallel_reduce("pgen_KerrWald_4", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb),
    KOKKOS_LAMBDA(const int &m, int &sum_ptcl, int &ptcl_mb) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;

      if ((x1min <= ptcl_x) && (x1max > ptcl_x) &&
          (x2min <= ptcl_y) && (x2max > ptcl_y) &&
          (x3min <= ptcl_z) && (x3max > ptcl_z)) {
        sum_ptcl += 1;
        ptcl_mb = m;
      }
    }, Kokkos::Sum<int>(nptcl_fnd), Kokkos::Sum<int>(ptcl_m));

  if ((nptcl_fnd != 0) && (nptcl_fnd != 1)) {
    Kokkos::printf("particle finding bug!\n");
  }

  // resize the particle data
  auto &pi_ = pmbp->ppart->prtcl_idata;
  auto &pr_ = pmbp->ppart->prtcl_rdata;
  int &nidata = pmbp->ppart->nidata;
  int &nrdata = pmbp->ppart->nrdata;
  int &npart = pmbp->ppart->nprtcl_thispack;
  int pgid = gids;
  if (nptcl_fnd == 1) {
    pgid += ptcl_m;
    Kokkos::printf("Particle found in mesh block %d.\n", pgid);
    npart = 1;
    pmy_mesh_->nprtcl_thisrank = npart;
    pmy_mesh_->nprtcl_eachrank[global_variable::my_rank] = npart;
    pmy_mesh_->nprtcl_total += npart;
    Kokkos::resize(pi_, nidata, npart);
    Kokkos::resize(pr_, nrdata, npart);
  }
  // set prtcl datga
  if (pmbp->ppart->pusher == ParticlesPusher::geo_boris) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "this test problem does not support geo_boris pusher" << std::endl;
    exit(EXIT_FAILURE);
  } else {
    par_for("pgen_ptcl", DevExeSpace(), 0, npart-1,
    KOKKOS_LAMBDA(int p) {
      pi_(PGID, p) = pgid;
      pi_(PTAG, p) = 0;
      pr_(IPX, p) = ptcl_x;
      pr_(IPY, p) = ptcl_y;
      pr_(IPZ, p) = ptcl_z;
      pr_(IPVX, p) = ptcl_ux;
      pr_(IPVY, p) = ptcl_uy;
      pr_(IPVZ, p) = ptcl_uz;
    });
  }

  return;
}