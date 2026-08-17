//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_hdiff_test.cpp
//! \brief Single-mode Hamiltonian-constraint perturbation for verifying the parabolic
//! H-damping term (campaign evidence/2026-08-17_bssn_parabolic_H_damping).
//!
//! Initial data: conformally flat metric g_ij = psi^4 delta_ij with
//!   psi^4 = 1 + amp*sin(k x),  k = 2*pi*nmode/Lx,  K_ij = 0, vacuum.
//! This is a pure Hamiltonian-constraint violation, H = -8 psi^-5 LapT(psi).
//! With <z4c>/hdamp_cH = c_H > 0 the perturbation obeys (linearized)
//!   d_t dchi = c_H Lap dchi, so the mode amplitude and H decay at rate
//!   gamma = c_H * sigma(k), sigma(k) the discrete Dxx symbol. Run with
//!   c_H = 0 (control), c_H > 0 (decay), and (code hacked) c_H < 0 (growth).
//!
//! History output: max|H|, rms H, rms(chi-1), and the sin(kx) mode amplitude of chi.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/adm.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "driver/driver.hpp"
#include "pgen/pgen.hpp"

static Real hdiff_kx = 0.0;   // mode wavenumber, set in UserProblem

void Z4cHdiffDiagnostics(HistoryData *pdata, Mesh *pm);

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_hist_func = &Z4cHdiffDiagnostics;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "z4c_hdiff_test requires a <z4c> block in the input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  Real amp = pin->GetOrAddReal("problem", "amp", 1.0e-3);
  int nmode = pin->GetOrAddInteger("problem", "nmode", 4);
  Real x1min = pmy_mesh_->mesh_size.x1min;
  Real x1max = pmy_mesh_->mesh_size.x1max;
  hdiff_kx = 2.0*M_PI*nmode/(x1max - x1min);

  if (restart) return;

  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto &adm = pmbp->padm->adm;
  Real kx = hdiff_kx;
  int nx1 = indcs.nx1;

  par_for("hdiff_init", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real &x1min_b = size.d_view(m).x1min;
    Real &x1max_b = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-ng, nx1, x1min_b, x1max_b);

    Real psi4 = 1.0 + amp*sin(kx*x1v);
    for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b) {
      adm.g_dd(m,a,b,k,j,i) = (a == b) ? psi4 : 0.0;
      adm.vK_dd(m,a,b,k,j,i) = 0.0;
    }
    adm.alpha(m,k,j,i) = 1.0;
    for (int a = 0; a < 3; ++a) {
      adm.beta_u(m,a,k,j,i) = 0.0;
    }
  });

  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin); break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin); break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin); break;
  }
  pmbp->pz4c->Z4cToADM(pmbp);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMConstraints<2>(pmbp); break;
    case 3: pmbp->pz4c->ADMConstraints<3>(pmbp); break;
    case 4: pmbp->pz4c->ADMConstraints<4>(pmbp); break;
  }
  if (global_variable::my_rank == 0) {
    std::cout << "# [z4c_hdiff_test] amp = " << amp << ", nmode = " << nmode
              << ", kx = " << hdiff_kx << std::endl;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Z4cHdiffDiagnostics
//! \brief history: [0] max|H|, [1] rms H, [2] rms(chi-1), [3] chi sin-mode amplitude

void Z4cHdiffDiagnostics(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 4;
  pdata->label[0] = "Hmax";
  pdata->label[1] = "Hrms";
  pdata->label[2] = "dchirms";
  pdata->label[3] = "chimode";

  auto &indcs = pm->mb_indcs;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int &ng = indcs.ng;
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &size = pmbp->pmb->mb_size;
  auto &pz4c = pmbp->pz4c;
  auto &u_con_ = pmbp->pz4c->u_con;
  auto &u0_ = pmbp->pz4c->u0;
  Real kx = hdiff_kx;

  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  array_sum::GlobalSum sum_this_mb;
  Real hmax = 0.0;
  Kokkos::parallel_reduce("hdiff-diag", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum, Real &max_h) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    Real &x1min_b = size.d_view(m).x1min;
    Real &x1max_b = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min_b, x1max_b);

    Real hh = u_con_(m, pz4c->I_CON_H, k, j, i);
    Real dchi = u0_(m, pz4c->I_Z4C_CHI, k, j, i) - 1.0;
    max_h = fmax(max_h, fabs(hh));

    array_sum::GlobalSum evars;
    evars.the_array[0] = vol*hh*hh;
    evars.the_array[1] = vol*dchi*dchi;
    evars.the_array[2] = vol*dchi*sin(kx*x1v);
    for (int n = 3; n < NREDUCTION_VARIABLES; ++n) {
      evars.the_array[n] = 0.0;
    }
    mb_sum += evars;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb), Kokkos::Max<Real>(hmax));

  Real vol = (pm->mesh_size.x1max - pm->mesh_size.x1min)
            *(pm->mesh_size.x2max - pm->mesh_size.x2min)
            *(pm->mesh_size.x3max - pm->mesh_size.x3min);
  pdata->hdata[0] = hmax;
  pdata->hdata[1] = std::sqrt(sum_this_mb.the_array[0]/vol);
  pdata->hdata[2] = std::sqrt(sum_this_mb.the_array[1]/vol);
  pdata->hdata[3] = 2.0*sum_this_mb.the_array[2]/vol;
  return;
}
