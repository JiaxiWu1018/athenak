//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_newdt.cpp
//! \brief function to compute z4c timestep across all MeshBlock(s) in a MeshBlockPack

#include <math.h>

#include <limits>
#include <iostream>
#include <algorithm>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "z4c.hpp"

namespace z4c {

//----------------------------------------------------------------------------------------
//! \fn void Z4c::NewTimeStep()
//! \brief calculate the minimum timestep within a MeshBlockPack for z4c problems

TaskStatus Z4c::NewTimeStep(Driver *pdriver, int stage) {
  if (stage != (pdriver->nexp_stages)) {
    return TaskStatus::complete; // only execute last stage
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;

  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();

  // capture class variables for kernel
  auto &mbsize = pmy_pack->pmb->mb_size;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;

  Kokkos::parallel_reduce("Z4c dt",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;

    min_dt1 = fmin((mbsize.d_view(m).dx1), min_dt1);
    min_dt2 = fmin((mbsize.d_view(m).dx2), min_dt2);
    min_dt3 = fmin((mbsize.d_view(m).dx3), min_dt3);
  }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));

  // compute minimum of dt1/dt2/dt3 for 1D/2D/3D problems
  dtnew = dt1;
  if (pmy_pack->pmesh->multi_d) { dtnew = std::min(dtnew, dt2); }
  if (pmy_pack->pmesh->three_d) { dtnew = std::min(dtnew, dt3); }

  // Parabolic timestep bound for the chi Hamiltonian-damping term:
  //   dt <= safety * s_RK * dx_min^2 / (sigma_3D * c_H * psi_max)
  // with s_RK = 2.7853 the 4-stage-RK4 real-axis stability edge, sigma_3D =
  // 3x the worst-mode symbol of the Dxx stencil (4, 16/3, 272/45 per direction
  // for nghost=2,3,4), and psi_max = chi_min^{1/p} the instantaneous maximum
  // of the conformal factor (D = c_H*psi grows in collapse). See
  // evidence/2026-08-17_bssn_parabolic_H_damping/derivation note, Sec. 5.
  auto &opt = pmy_pack->pz4c->opt;
  if (opt.hdamp_cH > 0.0) {
    Real dx_min = dtnew;   // dtnew so far is exactly min(dx), no wavespeed factor
    auto &chi_ = pmy_pack->pz4c->z4c.chi;
    int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;
    const int nji = nx2*nx1;
    Real chi_min = std::numeric_limits<Real>::max();
    Kokkos::parallel_reduce("hdamp_chimin",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &min_chi) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks; j += js;
      min_chi = fmin(chi_(m,k,j,i), min_chi);
    }, Kokkos::Min<Real>(chi_min));
    chi_min = fmax(chi_min, opt.chi_min_floor);
    Real psi_max = pow(chi_min, 1.0/opt.chi_psi_power);  // p < 0: chi_min -> psi_max
    constexpr Real s_rk = 2.7853;
    const Real sig1d = (indcs.ng == 2) ? 4.0 :
                       (indcs.ng == 3) ? 16.0/3.0 : 272.0/45.0;
    Real dt_par = opt.hdamp_par_safety * s_rk * dx_min*dx_min
                  / (3.0*sig1d * opt.hdamp_cH * psi_max);
    Real cfl = pmy_pack->pmesh->cfl_no;
    Real dt_hyp = cfl * dx_min;
    if (global_variable::my_rank == 0 && pmy_pack->pmesh->ncycle == 0) {
      std::cout << "# [z4c] hdamp dt: dt_hyp = " << dt_hyp << ", dt_par = " << dt_par
                << " (psi_max = " << psi_max
                << ", safety = " << opt.hdamp_par_safety << "); "
                << ((dt_par < dt_hyp) ? "PARABOLIC" : "hyperbolic")
                << " bound controls, dt_par/dt_hyp = " << dt_par/dt_hyp << std::endl;
    }
    if (dt_par < dt_hyp && !pmy_pack->pz4c->hdamp_dtwarned) {
      pmy_pack->pz4c->hdamp_dtwarned = true;
      std::cout << "# [z4c] NOTE (rank " << global_variable::my_rank
                << ", cycle " << pmy_pack->pmesh->ncycle
                << "): parabolic dt bound now controls (psi_max = " << psi_max
                << ", dt_par = " << dt_par << " < dt_hyp = " << dt_hyp << ")" << std::endl;
    }
    dtnew = std::min(dtnew, dt_par/cfl);
  }

  return TaskStatus::complete;
}
} // namespace z4c
