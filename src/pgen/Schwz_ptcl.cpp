//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file Schwz_ptcl.cpp
//! \brief Problem generator for particle pusher tests in Schwarzschild spacetime
//!

#include <math.h>
#include <algorithm>

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

void SetADMVariablesToIsotropic(MeshBlockPack *pmbp);

template<int NG>
KOKKOS_INLINE_FUNCTION
void GetADMVariables(Real &alp, Real *beta, Real *g3d, const Real *x_mid,
                     const int mb, const Real *mb_par, const int *ncell,
                     const DvceArray5D<Real> &adm_n, const DvceArray5D<Real> &adm_nm1);

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  Real ptcl_x = pin->GetReal("problem", "ptcl_x");
  Real ptcl_y = pin->GetReal("problem", "ptcl_y");
  Real ptcl_z = pin->GetReal("problem", "ptcl_z");
  Real ptcl_ux = pin->GetReal("problem", "ptcl_ux");
  Real ptcl_uy = pin->GetReal("problem", "ptcl_uy");
  Real ptcl_uz = pin->GetReal("problem", "ptcl_uz");

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int &nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  // Set primitive variables
  auto &w0_ = pmbp->pmhd->w0;
  Real dfloor = pmbp->pmhd->peos->eos_data.dfloor;
  Real pfloor = pmbp->pmhd->peos->eos_data.pfloor;
  par_for("pgen_Schwz_ptcl1", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k,int j,int i) {
      w0_(m, IDN, k, j, i) = dfloor;
      w0_(m, IVX, k, j, i) = 0.0;
      w0_(m, IVY, k, j, i) = 0.0;
      w0_(m, IVZ, k, j, i) = 0.0;
      w0_(m, IEN, k, j, i) = pfloor;
    });
  Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->w0_last, w0_);

  // Set magnetic field
  auto &b0 = pmbp->pmhd->b0;
  par_for("pgen_Schwz_ptcl2", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k,int j,int i) {
      b0.x1f(m, k, j, i) = 0.0;
      b0.x2f(m, k, j, i) = 0.0;
      b0.x3f(m, k, j, i) = 0.0;
      if (i == ie) {
        b0.x1f(m, k, j, i + 1) = 0.0;
      }
      if (j == je) {
        b0.x2f(m, k, j + i, i) = 0.0;
      }
      if (k == ke) {
        b0.x3f(m, k + 1, j, i) = 0.0;
      }
  });
  auto &bcc_ = pmbp->pmhd->bcc0;
  par_for("pgen_Schwz_ptcl3", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k,int j,int i) {
      bcc_(m, IBX, k, j, i) = 0.0;
      bcc_(m, IBY, k, j, i) = 0.0;
      bcc_(m, IBZ, k, j, i) = 0.0;
  });
  Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->bcc0_last, bcc_);

  // Set spacetime and perform p2c
  // pmbp->padm->SetADMVariables = &SetADMVariablesToIsotropic;
  pmbp->padm->SetADMVariables(pmbp);
  auto &adm = pmbp->padm->u_adm;
  Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->adm_last, adm);
  if (pmbp->ppart->pusher == ParticlesPusher::geo_boris){
    Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->adm_last, adm);
  }
  pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);

  // Find ptcl rank
  int gids = pmbp->gids;
  int nptcl_fnd = 0, ptcl_m = -1;
  Kokkos::parallel_reduce("pgen_Schwz_ptcl4", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb),
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
  // set particle data
  if (pmbp->ppart->pusher == ParticlesPusher::geo_boris) {
    const Real mb_par[9] = {size.h_view(ptcl_m).x1min, size.h_view(ptcl_m).x1max, size.h_view(ptcl_m).dx1,
                            size.h_view(ptcl_m).x2min, size.h_view(ptcl_m).x2max, size.h_view(ptcl_m).dx2,
                            size.h_view(ptcl_m).x3min, size.h_view(ptcl_m).x3max, size.h_view(ptcl_m).dx3};
    Real x_mid[3] = {ptcl_x, ptcl_y, ptcl_z};
    int &ng = indcs.ng;
    int ncell[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
    auto &adm_n = pmbp->ppart->adm_last;
    auto &adm_nm1 = pmbp->ppart->adm_last;
    Real dt_ = 0.0375;
    par_for("pgen_Schwz_ptcl5", DevExeSpace(), 0, npart-1,
    KOKKOS_LAMBDA(int p) {
      Real alp, beta[3], g3d[6];
      switch (ng) {
      case 2: {
        GetADMVariables<2>(alp, beta, g3d, x_mid, ptcl_m, mb_par, ncell, adm_n, adm_nm1);
        break;
      }
      case 3: {
        GetADMVariables<3>(alp, beta, g3d, x_mid, ptcl_m, mb_par, ncell, adm_n, adm_nm1);
        break;
      }
      case 4: {
        GetADMVariables<4>(alp, beta, g3d, x_mid, ptcl_m, mb_par, ncell, adm_n, adm_nm1);
        break;
      }}
      pi_(PGID, p) = pgid;
      pi_(PTAG, p) = 0;
      pr_(IPVX, p) = ptcl_ux;
      pr_(IPVY, p) = ptcl_uy;
      pr_(IPVZ, p) = ptcl_uz;
      Real det = Primitive::GetDeterminant(g3d);
      Real g3u[6] = {0.0};
      Primitive::InvertMatrix(g3u, g3d, det);
      Real u_l[3] = {ptcl_ux, ptcl_uy, ptcl_uz}, u_u[3] = {0.0};
      Primitive::RaiseForm(u_u, u_l, g3u);
      Real Lorentz = std::sqrt(1.0 + Primitive::Contract(u_u, u_l));
      Real iLorentz = 1. / Lorentz;
      Real v[3] = {0.0};
      for (int i = 0; i < 3 ; ++i) {
        v[i] = alp * iLorentz * u_u[i] - beta[i];
      }
      for (int i = 0; i < 3; ++i) {
        pr_(IPX+i, p) = x_mid[i] + 0.5 * dt_ * v[i];
        pr_(IPLX+i, p) = x_mid[i] - 0.5 * dt_ * v[i];
      }
    });
  } else {
    par_for("pgen_Schwz_ptcl5", DevExeSpace(), 0, npart-1,
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

template<int NG>
KOKKOS_INLINE_FUNCTION
void GetADMVariables(Real &alp, Real *beta, Real *g3d, const Real *x_mid,
                     const int mb, const Real *mb_par, const int *ncell,
                     const DvceArray5D<Real> &adm_n, const DvceArray5D<Real> &adm_nm1) {
  int interp_indcs[4] = {mb, -1, -1, -1};
  interp_indcs[1] = static_cast<int>(std::floor((x_mid[0] - (mb_par[0] + mb_par[2] / 2.0)) / mb_par[2]));
  interp_indcs[2] = static_cast<int>(std::floor((x_mid[1] - (mb_par[3] + mb_par[5] / 2.0)) / mb_par[5]));
  interp_indcs[3] = static_cast<int>(std::floor((x_mid[2] - (mb_par[6] + mb_par[8] / 2.0)) / mb_par[8]));
  constexpr int N = 2 * NG;
  Real Lx[N] = {0.0}, Ly[N] = {0.0}, Lz[N] = {0.0};
  particles::CalcInterpWght<NG>(x_mid, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
  Real alp_mid_n = 0.0, beta_mid_n[3] = {0.0}, g3d_mid_n[6] = {0.0};
  Real alp_mid_nm1 = 0.0, beta_mid_nm1[3] = {0.0}, g3d_mid_nm1[6] = {0.0};
  alp_mid_n = particles::LagrangeInterpolator<NG>(adm_n, adm::ADM::I_ADM_ALPHA, interp_indcs, Lx, Ly, Lz);
  alp_mid_nm1 = particles::LagrangeInterpolator<NG>(adm_nm1, adm::ADM::I_ADM_ALPHA, interp_indcs, Lx, Ly, Lz);
  alp = 0.5 * (alp_mid_n + alp_mid_nm1);
  for (int i = 0; i < 3; ++i) {
    beta_mid_n[i] = particles::LagrangeInterpolator<NG>(adm_n, adm::ADM::I_ADM_BETAX + i, interp_indcs, Lx, Ly, Lz);
    beta_mid_nm1[i] = particles::LagrangeInterpolator<NG>(adm_nm1, adm::ADM::I_ADM_BETAX + i, interp_indcs, Lx, Ly, Lz);
    beta[i] = 0.5 * (beta_mid_n[i] + beta_mid_nm1[i]);
  }
  for (int i = 0; i < 6; ++i) {
    g3d_mid_n[i] = particles::LagrangeInterpolator<NG>(adm_n, adm::ADM::I_ADM_GXX + i, interp_indcs, Lx, Ly, Lz);
    g3d_mid_nm1[i] = particles::LagrangeInterpolator<NG>(adm_nm1, adm::ADM::I_ADM_GXX + i, interp_indcs, Lx, Ly, Lz);
    g3d[i] = 0.5 * (g3d_mid_n[i] + g3d_mid_nm1[i]);
  }
}

void SetADMVariablesToIsotropic(MeshBlockPack *pmbp) {
  auto &adm = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  par_for("update_adm_vars", DevExeSpace(), 0,nmb-1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real r = std::sqrt(x1v * x1v + x2v * x2v + x3v * x3v);
    Real ir = 1. / r;

    adm.alpha(m, k, j, i) = (1. - 0.5 * ir) / (1 + 0.5 * ir);
    adm.beta_u(m, 0, k, j, i) = 0.;
    adm.beta_u(m, 1, k, j, i) = 0.;
    adm.beta_u(m, 2, k, j, i) = 0.;

    Real power4 = (1. + 0.5 * ir) * (1. + 0.5 * ir) * (1. + 0.5 * ir) * (1. + 0.5 * ir);
    adm.g_dd(m, 0, 0, k, j, i) = power4;
    adm.g_dd(m, 1, 1, k, j, i) = power4;
    adm.g_dd(m, 2, 2, k, j, i) = power4;
    adm.g_dd(m, 0, 1, k, j, i) = 0.;
    adm.g_dd(m, 0, 2, k, j, i) = 0.;
    adm.g_dd(m, 1, 2, k, j, i) = 0.;

    adm.vK_dd(m, 0, 0, k, j, i) = 0.;
    adm.vK_dd(m, 0, 1, k, j, i) = 0.;
    adm.vK_dd(m, 0, 2, k, j, i) = 0.;
    adm.vK_dd(m, 1, 1, k, j, i) = 0.;
    adm.vK_dd(m, 1, 2, k, j, i) = 0.;
    adm.vK_dd(m, 2, 2, k, j, i) = 0.;
  });
}