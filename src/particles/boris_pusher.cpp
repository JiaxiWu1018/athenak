//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file boris_pusher.cpp
//! \brief special-relativistic Boris pusher (flat spacetime). One full-dt update per cycle:
//!   drift half-step -> interpolate (B, v_fluid) from MHD at the midpoint -> form the ideal-MHD
//!   electric field E = B x v -> relativistic Boris velocity kick -> drift half-step.
//! Velocity slots IPVX/IPVY/IPVZ hold the spatial 4-velocity (u^i == u_i in flat space).
//!
//! MHD convention: the w0 velocity slots hold the spatial 4-velocity utilde^i = gamma_f v^i
//! (both SR-MHD, see ideal_srmhd.cpp, and dyn_grmhd); the pusher divides out gamma_f before
//! forming E = -v x B. (In flat space bcc0 is the physical B for both paths.) Newtonian-MHD
//! caveat: there w0 holds v itself, so the division introduces only an O(v^2) relative error —
//! negligible exactly where Newtonian MHD is valid.

#include <cmath>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "particles.hpp"
#include "boris_utils.hpp"
#include "lagrange_interp.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/cell_locations.hpp"

namespace particles {

//----------------------------------------------------------------------------------------
//! \fn void Particles::BorisPush

void Particles::BorisPush() {
  // The SR Boris pusher sources its EM field from MHD. The constructor already requires an
  // <mhd> block when pusher==boris; guard again so a misuse fails safe instead of segfaulting.
  if (pmy_pack->pmhd == nullptr) {return;}

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

  par_for("boris_push", DevExeSpace(), 0, nprtcl_thispack - 1,
  KOKKOS_LAMBDA(const int p) {
    Real x_n[3] = {pr(IPX, p), pr(IPY, p), pr(IPZ, p)};
    Real u_n[3] = {pr(IPVX, p), pr(IPVY, p), pr(IPVZ, p)};

    // drift to the half-step position used for field interpolation
    Real gamma = std::sqrt(1.0 + u_n[0]*u_n[0] + u_n[1]*u_n[1] + u_n[2]*u_n[2]);
    Real x_half[3] = {0.0};
    for (int i = 0; i < 3; ++i) {
      x_half[i] = x_n[i] + 0.5*dt_*u_n[i]/gamma;
    }

    // stencil indices + Lagrange weights at the half-step position
    int ncell[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
    int mb = pi(PGID, p) - gids;
    const Real mb_par[9] = {size.d_view(mb).x1min, size.d_view(mb).x1max, size.d_view(mb).dx1,
                            size.d_view(mb).x2min, size.d_view(mb).x2max, size.d_view(mb).dx2,
                            size.d_view(mb).x3min, size.d_view(mb).x3max, size.d_view(mb).dx3};
    int interp_indcs[4] = {mb, -1, -1, -1};
    SetInterpIndices(x_half, mb_par, ncell, interp_indcs);
    Real Lx[8] = {0.0}, Ly[8] = {0.0}, Lz[8] = {0.0};

    // interpolate cell-centred B and fluid 3-velocity
    Real B_interp[3] = {0.0}, v_interp[3] = {0.0};
    switch (ng) {
    case 2:
      CalcInterpWght<2>(x_half, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
      for (int idx = 0; idx < 3; ++idx) {
        B_interp[idx] = LagrangeInterpolator<2>(bcc, idx, interp_indcs, Lx, Ly, Lz);
        v_interp[idx] = LagrangeInterpolator<2>(prim, idx+IVX, interp_indcs, Lx, Ly, Lz);
      }
      break;
    case 3:
      CalcInterpWght<3>(x_half, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
      for (int idx = 0; idx < 3; ++idx) {
        B_interp[idx] = LagrangeInterpolator<3>(bcc, idx, interp_indcs, Lx, Ly, Lz);
        v_interp[idx] = LagrangeInterpolator<3>(prim, idx+IVX, interp_indcs, Lx, Ly, Lz);
      }
      break;
    case 4:
      CalcInterpWght<4>(x_half, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
      for (int idx = 0; idx < 3; ++idx) {
        B_interp[idx] = LagrangeInterpolator<4>(bcc, idx, interp_indcs, Lx, Ly, Lz);
        v_interp[idx] = LagrangeInterpolator<4>(prim, idx+IVX, interp_indcs, Lx, Ly, Lz);
      }
      break;
    }

    // w0 stores the spatial 4-velocity utilde^i = gamma_f v^i (SR-MHD and dyn_grmhd
    // convention); the ideal-MHD E field needs the transport velocity v = utilde/gamma_f
    Real Wf = std::sqrt(1.0 + v_interp[0]*v_interp[0] + v_interp[1]*v_interp[1]
                            + v_interp[2]*v_interp[2]);
    for (int idx = 0; idx < 3; ++idx) { v_interp[idx] /= Wf; }

    // ideal-MHD electric field E = -v x B = B x v
    Real E_interp[3] = {0.0};
    E_interp[0] = B_interp[1]*v_interp[2] - B_interp[2]*v_interp[1];
    E_interp[1] = B_interp[2]*v_interp[0] - B_interp[0]*v_interp[2];
    E_interp[2] = B_interp[0]*v_interp[1] - B_interp[1]*v_interp[0];

    // relativistic Boris velocity update
    Real u_np1[3] = {0.0};
    FlatBorisPush(u_np1, u_n, E_interp, B_interp, qom, dt_);

    // drift the remaining half-step with the updated velocity
    Real gamma_np1 = std::sqrt(1.0 + u_np1[0]*u_np1[0] + u_np1[1]*u_np1[1] + u_np1[2]*u_np1[2]);
    Real x_np1[3] = {0.0};
    for (int i = 0; i < 3; ++i) {
      x_np1[i] = x_half[i] + 0.5*dt_/gamma_np1*u_np1[i];
    }

    pr(IPX, p) = x_np1[0];
    pr(IPY, p) = x_np1[1];
    pr(IPZ, p) = x_np1[2];
    pr(IPVX, p) = u_np1[0];
    pr(IPVY, p) = u_np1[1];
    pr(IPVZ, p) = u_np1[2];
  });
}

} // namespace particles
