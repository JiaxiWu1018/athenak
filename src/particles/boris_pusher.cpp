//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file boris_pusher.cpp
//  \brief New Boris pusher function, which uses routines outlined in Zou+2024

#include <cmath>
#include <functional>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "particles.hpp"
#include "boris_utils.hpp"
#include "lagrange_interp.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/cell_locations.hpp"

namespace particles {

void Particles::BorisPush() {
  // Extract MHD variables
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  int gids = pmy_pack->gids;
  auto dt_ = pmy_pack->pmesh->dt;
  auto qom = q_over_m;
  auto &bcc = pmy_pack->pmhd->bcc0;
  auto &prim = pmy_pack->pmhd->w0;
  int &ng = indcs.ng;

  // Loop over all particles
  par_for("boris_push", DevExeSpace(), 0, nprtcl_thispack - 1,
  KOKKOS_LAMBDA(const int p) {
    // Retrieve particle position and velocity
    Real x_n[3] = {pr(IPX, p), pr(IPY, p), pr(IPZ, p)};
    Real u_n[3] = {pr(IPVX, p), pr(IPVY, p), pr(IPVZ, p)};

    // First flat spacetime Boris push
    Real gamma = std::sqrt(1.0 + u_n[0] * u_n[0] + u_n[1] * u_n[1] + u_n[2] * u_n[2]);
    Real x_nplushalf[3] = {0.0};
    for (int i = 0; i < 3; ++i) {
      x_nplushalf[i] = x_n[i] + 0.5 * dt_ * u_n[i] / gamma;
    }

    // Set interpolation indices
    int ncell[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
    int mb = pi(PGID, p) - gids;
    const Real mb_par[9] = {size.d_view(mb).x1min, size.d_view(mb).x1max, size.d_view(mb).dx1,
                            size.d_view(mb).x2min, size.d_view(mb).x2max, size.d_view(mb).dx2,
                            size.d_view(mb).x3min, size.d_view(mb).x3max, size.d_view(mb).dx3};
    int interp_indcs[4] = {mb, -1, -1, -1};
    SetInterpIndices(x_nplushalf, mb_par, ncell, interp_indcs);
    Real Lx[8] = {0.0}, Ly[8] = {0.0}, Lz[8] = {0.0};

    // Perform interpolation to the B field and fluid velocity
    Real B_interp[3] = {0.0}, v_interp[3] = {0.0};
    switch (ng) {
    case 2:
      CalcInterpWght<2>(x_nplushalf, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
      for (int idx = 0; idx < 3; ++idx) {
        B_interp[idx] = LagrangeInterpolator<2>(bcc, idx, interp_indcs, Lx, Ly, Lz);
        v_interp[idx] = LagrangeInterpolator<2>(prim, idx+IVX, interp_indcs, Lx, Ly, Lz);
      }
      break;
    case 3:
      CalcInterpWght<3>(x_nplushalf, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
      for (int idx = 0; idx < 3; ++idx) {
        B_interp[idx] = LagrangeInterpolator<3>(bcc, idx, interp_indcs, Lx, Ly, Lz);
        v_interp[idx] = LagrangeInterpolator<3>(prim, idx+IVX, interp_indcs, Lx, Ly, Lz);
      }
      break;
    case 4:
      CalcInterpWght<4>(x_nplushalf, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
      for (int idx = 0; idx < 3; ++idx) {
        B_interp[idx] = LagrangeInterpolator<4>(bcc, idx, interp_indcs, Lx, Ly, Lz);
        v_interp[idx] = LagrangeInterpolator<4>(prim, idx+IVX, interp_indcs, Lx, Ly, Lz);
      }
      break;
    }

    // Calculate the E field assuming Ideal MHD for now
    Real E_interp[3] = {0.0};
    E_interp[0] = B_interp[1] * v_interp[2] - B_interp[2] * v_interp[1];
    E_interp[1] = B_interp[2] * v_interp[0] - B_interp[0] * v_interp[2];
    E_interp[2] = B_interp[0] * v_interp[1] - B_interp[1] * v_interp[0];

    Real u_nplus1[3] = {0.0};
    FlatBorisPush(u_nplus1, u_n, E_interp, B_interp, qom, dt_);

    // Second flat Boris push
    Real gamma_nplus1 = std::sqrt(1.0 + u_nplus1[0] * u_nplus1[0] + u_nplus1[1] * u_nplus1[1] + u_nplus1[2] * u_nplus1[2]);
    Real x_nplus1[3] = {0.0};
    for (int i = 0; i < 3; ++i) {
      x_nplus1[i] = x_nplushalf[i] + 0.5 * dt_ / gamma_nplus1 * u_nplus1[i];
    }

    // Update particle position and speed into device memory
    pr(IPX, p) = x_nplus1[0];
    pr(IPY, p) = x_nplus1[1];
    pr(IPZ, p) = x_nplus1[2];
    pr(IPVX, p) = u_nplus1[0];
    pr(IPVY, p) = u_nplus1[1];
    pr(IPVZ, p) = u_nplus1[2];
  });
} // end BorisPush
} // end namespace particles
