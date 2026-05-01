//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_tmunu.cpp
//  \brief Calculate Tmunu of particles

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles.hpp"
#include "z4c/z4c.hpp"
#include "z4c/tmunu.hpp"
#include "coordinates/cell_locations.hpp"
#include "lagrange_interp.hpp"
#include "mesh/nghbr_index.hpp"

namespace particles {
KOKKOS_INLINE_FUNCTION
Real form_fct_calc(Real x) {
  return 1. - std::abs(x);
}

KOKKOS_INLINE_FUNCTION
void deposite_tmunu(int mb, int idx, int idy, int idz, int ng, Real ivol, Real mass,
                    Real norm_x, Real norm_y, Real norm_z, Real u_dot_n, Real *u_d,
                    Tmunu::Tmunu_vars const& tmunu) {
  int form_fct_order = 1;
  // calculate Tmunu deposition for first order form factor
  for (int i = 0; i <= form_fct_order; ++i) {
    int idxl = idx + ng + i;
    Real S_i = form_fct_calc(norm_x - i);
    for (int j = 0; j <= form_fct_order; ++j) {
      int idyl = idy + ng + j;
      Real S_j = form_fct_calc(norm_y - j);
      for(int k = 0; k <= form_fct_order; ++k) {
        int idzl = idz + ng + k;
        Real S_k = form_fct_calc(norm_z - k);
        Real S_ijk = S_i * S_j * S_k;
        tmunu.E(mb, idzl, idyl, idxl) += mass * ivol * S_ijk * u_dot_n * u_dot_n;
        for (int a = 0; a < 3; ++a) {
          tmunu.S_d(mb, a, idzl, idyl, idxl) -= mass * ivol * S_ijk * u_d[a+1] * u_dot_n;
          for (int b = 0; b < 3; ++b) {
            tmunu.S_dd(mb, a, b, idzl, idyl, idxl) += mass * ivol * S_ijk * u_d[a+1] * u_d[b+1];
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::SetPrtclTmunu
//! \brief Wrapper task list function that calculates Tmunu which enters the RHD of z4c.

TaskStatus Particles::SetPrtclTmunu(Driver *pdrive, int stage) {
  int nmb = pmy_pack->nmb_thispack;
  int gids = pmy_pack->gids;
  auto &size = pmy_pack->pmb->mb_size;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie; int &nx1 = indcs.nx1;
  int &js = indcs.js; int &je = indcs.je; int &nx2 = indcs.nx2;
  int &ks = indcs.ks; int &ke = indcs.ke; int &nx3 = indcs.nx3;
  int ncell[3] = {nx1, nx2, nx3};
  int ng = indcs.ng;

  // clean the tmunu and prepare for particle deposition
  auto &tmunu = pmy_pack->ptmunu->tmunu;
  par_for("clean_tmunu", DevExeSpace(), 0, nmb-1, ks-ng, ke+ng, js-ng, je+ng, is-ng, ie+ng,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    tmunu.E(m, k, j, i) = 0.0;
    for (int a = 0; a < 3; ++a) {
      tmunu.S_d(m,  a, k, j, i) = 0.0;
      for (int b = 0; b < 3; ++b) {
        tmunu.S_dd(m, a, b, k, j, i) = 0.0;
      }
    }
  });

  auto &z4c = pmy_pack->pz4c->u0;
  auto &pr = prtcl_rdata;
  auto &pi = prtcl_idata;
  auto &meshsize = pmy_pack->pmesh->mesh_size;
  auto &nghbr = pmy_pack->pmb->nghbr;
  // auto myrank = global_variable::my_rank; // For future use
  Real &mp = mass;
  par_for("prtcl_tmunu", DevExeSpace(), 0, nprtcl_thispack - 1, KOKKOS_LAMBDA(const int p) {
    // extract meshblock information
    int m = pi(PGID, p) - gids;
    Real &x1min = size.d_view(m).x1min; Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min; Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min; Real &x3max = size.d_view(m).x3max;
    Real &dx1 = size.d_view(m).dx1; Real &dx2 = size.d_view(m).dx2; Real &dx3 = size.d_view(m).dx3;

    // calculate nearest cell on the left side of the particle
    Real xp = pr(IPX, p); Real yp = pr(IPY, p); Real zp = pr(IPZ, p);
    int idx = LeftEdgeIndex(xp, nx1, x1min, x1max);
    int idy = LeftEdgeIndex(yp, nx2, x2min, x2max);
    int idz = LeftEdgeIndex(zp, nx3, x3min, x3max);
    Real xi = CellCenterX(idx, nx1, x1min, x1max);
    Real yi = CellCenterX(idy, nx2, x2min, x2max);
    Real zi = CellCenterX(idz, nx3, x3min, x3max);

    // calculate n^{\nu} at the particle position
    Real n_u[4] = {0.0};
    const Real mb_par[9] = {x1min, x1max, dx1, x2min, x2max, dx2, x3min, x3max, dx3};
    int interp_indcs[4] = {m, idx, idy, idz};
    Real x[3] = {xp, yp, zp};
    switch (ng) {
    case 2: {
      constexpr int NGHOST = 2;
      Real Lx[2 * NGHOST] = {0.0}, Ly[2 * NGHOST] = {0.0}, Lz[2 * NGHOST] = {0.0};
      CalcInterpWght<NGHOST>(x, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
      Real alp = LagrangeInterpolator<NGHOST>(z4c, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, Lx, Ly, Lz);
      Real betax = LagrangeInterpolator<NGHOST>(z4c, z4c::Z4c::I_Z4C_BETAX, interp_indcs, Lx, Ly, Lz);
      Real betay = LagrangeInterpolator<NGHOST>(z4c, z4c::Z4c::I_Z4C_BETAY, interp_indcs, Lx, Ly, Lz);
      Real betaz = LagrangeInterpolator<NGHOST>(z4c, z4c::Z4c::I_Z4C_BETAZ, interp_indcs, Lx, Ly, Lz);
      Real ialp = 1. / alp;
      n_u[0] = ialp; n_u[1] = -ialp * betax; n_u[2] = -ialp * betay; n_u[3]  = -ialp * betaz;
      break;
    }
    case 3: {
      constexpr int NGHOST = 3;
      Real Lx[2 * NGHOST] = {0.0}, Ly[2 * NGHOST] = {0.0}, Lz[2 * NGHOST] = {0.0};
      CalcInterpWght<NGHOST>(x, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
      Real alp = LagrangeInterpolator<NGHOST>(z4c, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, Lx, Ly, Lz);
      Real betax = LagrangeInterpolator<NGHOST>(z4c, z4c::Z4c::I_Z4C_BETAX, interp_indcs, Lx, Ly, Lz);
      Real betay = LagrangeInterpolator<NGHOST>(z4c, z4c::Z4c::I_Z4C_BETAY, interp_indcs, Lx, Ly, Lz);
      Real betaz = LagrangeInterpolator<NGHOST>(z4c, z4c::Z4c::I_Z4C_BETAZ, interp_indcs, Lx, Ly, Lz);
      Real ialp = 1. / alp;
      n_u[0] = ialp; n_u[1] = -ialp * betax; n_u[2] = -ialp * betay; n_u[3]  = -ialp * betaz;
      break;
    }
    case 4: {
      constexpr int NGHOST = 4;
      Real Lx[2 * NGHOST] = {0.0}, Ly[2 * NGHOST] = {0.0}, Lz[2 * NGHOST] = {0.0};
      CalcInterpWght<NGHOST>(x, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
      Real alp = LagrangeInterpolator<NGHOST>(z4c, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, Lx, Ly, Lz);
      Real betax = LagrangeInterpolator<NGHOST>(z4c, z4c::Z4c::I_Z4C_BETAX, interp_indcs, Lx, Ly, Lz);
      Real betay = LagrangeInterpolator<NGHOST>(z4c, z4c::Z4c::I_Z4C_BETAY, interp_indcs, Lx, Ly, Lz);
      Real betaz = LagrangeInterpolator<NGHOST>(z4c, z4c::Z4c::I_Z4C_BETAZ, interp_indcs, Lx, Ly, Lz);
      Real ialp = 1. / alp;
      n_u[0] = ialp; n_u[1] = -ialp * betax; n_u[2] = -ialp * betay; n_u[3]  = -ialp * betaz;
      break;
    }}

    // calculate u dot n
    Real u_d[4] = {-1. * pr(IPEN, p), pr(IPVX, p), pr(IPVY, p), pr(IPVZ, p)};
    Real u_dot_n = 0.0;
    for (int a = 0; a < 4; ++a) {
      u_dot_n += u_d[a] * n_u[a];
    }

    // handle particles at the meshblock boundary
    int nbx1 = 0; int nbx2 = 0; int nbx3 = 0;
    if (idx == -1 && xp - dx1 > meshsize.x1min) {
      nbx1 = -1;
    } else if (idx == nx1 - 1 && xp + dx1 < meshsize.x1max) {
      nbx1 = 1;
    }
    if (idy == -1 && yp - dx2 > meshsize.x2min) {
      nbx2 = -1;
    } else if (idy == nx1 - 1 && yp +dx2 < meshsize.x2max) {
      nbx2 = 1;
    }
    if (idz == -1 && zp - dx3 > meshsize.x3min) {
      nbx3 = -1;
    } else if (idz == nx1 - 1 && zp + dx3 < meshsize.x3max) {
      nbx3 = 1;
    }
    // calculate cell size
    Real idx1 = 1. / dx1; Real idx2 = 1. / dx2; Real idx3 = 1. / dx3;
    Real ivol = idx1 * idx2 * idx3;
    // for now we assume a uniform grid
    Real norm_x = (xp - xi) * idx1;
    Real norm_y = (yp - yi) * idx2;
    Real norm_z = (zp - zi) * idx3;
    deposite_tmunu(m, idx, idy, idz, ng, ivol, mp, norm_x, norm_y, norm_z, u_dot_n, u_d, tmunu);
    int nghbr_id = -1; int m_nghbr = -1;
    if (nbx1 != 0) {
      // x1face
      nghbr_id = NeighborIndex(nbx1, 0, 0, 0, 0);
      m_nghbr = nghbr.d_view(m, nghbr_id).gid - gids;
      int idx_nghbr = idx == -1 ? nx1 - 1 : -1;
      deposite_tmunu(m_nghbr, idx_nghbr, idy, idz, ng, ivol, mp, norm_x, norm_y, norm_z, u_dot_n, u_d, tmunu);
      if (nbx2 != 0) {
        // x1-x2 edge
        nghbr_id = NeighborIndex(nbx1, nbx2, 0, 0, 0);
        m_nghbr = nghbr.d_view(m, nghbr_id).gid - gids;
        int idy_nghbr = idy == -1 ? nx2 - 1 : -1;
        deposite_tmunu(m_nghbr, idx_nghbr, idy_nghbr, idz, ng, ivol, mp, norm_x, norm_y, norm_z, u_dot_n, u_d, tmunu);
        if (nbx3 != 0) {
          // corner
          nghbr_id = NeighborIndex(nbx1, nbx2, nbx3, 0, 0);
          m_nghbr = nghbr.d_view(m, nghbr_id).gid - gids;
          int idz_nghbr = idz == -1 ? nx3 - 1 : -1;
          deposite_tmunu(m_nghbr, idx_nghbr, idy_nghbr, idz_nghbr, ng, ivol, mp, norm_x, norm_y, norm_z, u_dot_n, u_d, tmunu);
        }
      }
      if (nbx3 != 0) {
        // x1-x3 edge
        nghbr_id = NeighborIndex(nbx1, 0, nbx3, 0, 0);
        m_nghbr = nghbr.d_view(m, nghbr_id).gid - gids;
        int idz_nghbr = idz == -1 ? nx3 - 1 : -1;
        deposite_tmunu(m_nghbr, idx_nghbr, idy, idz_nghbr, ng, ivol, mp, norm_x, norm_y, norm_z, u_dot_n, u_d, tmunu);
      }
    }
    if (nbx2 != 0) {
      // x2face
      nghbr_id = NeighborIndex(0, nbx2, 0, 0, 0);
      m_nghbr = nghbr.d_view(m, nghbr_id).gid - gids;
      int idy_nghbr = idy == -1 ? nx2 - 1 : -1;
      deposite_tmunu(m_nghbr, idx, idy_nghbr, idz, ng, ivol, mp, norm_x, norm_y, norm_z, u_dot_n, u_d, tmunu);
      if (nbx3 != 0) {
        //x2-x3edge
        nghbr_id = NeighborIndex(0, nbx2, nbx3, 0, 0);
        m_nghbr = nghbr.d_view(m, nghbr_id).gid - gids;
        int idz_nghbr = idz == -1 ? nx3 - 1 : -1;
        deposite_tmunu(m_nghbr, idx, idy_nghbr, idz_nghbr, ng, ivol, mp, norm_x, norm_y, norm_z, u_dot_n, u_d, tmunu);
      }
    }
    if (nbx3 != 0) {
      // x3face
      nghbr_id = NeighborIndex(0, 0, nbx3, 0, 0);
      m_nghbr = nghbr.d_view(m, nghbr_id).gid - gids;
      int idz_nghbr = idz == -1 ? nx3 - 1 : -1;
      deposite_tmunu(m_nghbr, idx, idy, idz_nghbr, ng, ivol, mp, norm_x, norm_y, norm_z, u_dot_n, u_d, tmunu);
    }
  });

  // TODO: ghost particles deposition, ghost particles are stored in a separate list.

  return TaskStatus::complete;
}
} // end namespace particles