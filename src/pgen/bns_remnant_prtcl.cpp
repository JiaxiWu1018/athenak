//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bns_remnant_prtcl.cpp
//! \brief Problem generator for propagating particles on the background of a binary
//!        neutron star merger remnant, including pp and p\gamma interactions.
//!

#include <math.h>
#include <algorithm>
#include <hdf5.h>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "eos/primitive-solver/geom_math.hpp"
#include "particles/particles.hpp"

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int ng = indcs.ng;
  int ncells1 = indcs.nx1 + 2 * ng;
  int ncells2 = indcs.nx2 + 2 * ng;
  int ncells3 = indcs.nx3 + 2 * ng;

  hid_t H5T_REAL = (sizeof(Real) == sizeof(float)) ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;

  std::string init_fname = pin->GetString("problem", "init_file");
  std::string init_iter = pin->GetString("problem", "iteration");
  hid_t file = H5Fopen(init_fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  hid_t g_it = H5Gopen(file, init_iter.c_str(), H5P_DEFAULT);

  int nx1, nx2, nx3, nmb_tot;
  hid_t attr;
  attr = H5Aopen(g_it, "nx1", H5P_DEFAULT);
  H5Aread(attr, H5T_NATIVE_INT, &nx1);
  H5Aclose(attr);
  attr = H5Aopen(g_it, "nx2", H5P_DEFAULT);
  H5Aread(attr, H5T_NATIVE_INT, &nx2);
  H5Aclose(attr);
  attr = H5Aopen(g_it, "nx3", H5P_DEFAULT);
  H5Aread(attr, H5T_NATIVE_INT, &nx3);
  H5Aclose(attr);
  attr = H5Aopen(g_it, "nmb", H5P_DEFAULT);
  H5Aread(attr, H5T_NATIVE_INT, &nmb_tot);
  H5Aclose(attr);

  if (indcs.nx1 != nx1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Number of cells in x1 direction doesn't match" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (indcs.nx2 != nx2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Number of cells in x2 direction doesn't match" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (indcs.nx3 != nx3) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Number of cells in x3 direction doesn't match" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (nmb_tot != pmy_mesh_->nmb_total) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Number of total meshblock doesn't match" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  hid_t mb_min_data = H5Dopen(g_it, "mb_bounds_min", H5P_DEFAULT);
  std::vector<Real> mb_min(nmb_tot * 3);
  H5Dread(mb_min_data, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, mb_min.data());
  H5Dclose(mb_min_data);

  hid_t mb_max_data = H5Dopen(g_it, "mb_bounds_max", H5P_DEFAULT);
  std::vector<Real> mb_max(nmb_tot * 3);
  H5Dread(mb_max_data, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, mb_max.data());
  H5Dclose(mb_max_data);

  hid_t mb_idx_data = H5Dopen(g_it, "mb_idx", H5P_DEFAULT);
  std::vector<int> mb_idx(nmb_tot);
  H5Dread(mb_idx_data, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, mb_idx.data());
  H5Dclose(mb_idx_data);

  // find meshblock index in dataset
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  int &nmb = pmbp->nmb_thispack;
  int &gids = pmbp->gids;
  int &gide = pmbp->gide;

  if ((gide - gids + 1) != nmb) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "gids/gide and nmb do not match" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  hid_t g_vars = H5Gopen(g_it, "vars", H5P_DEFAULT);

  {
    // Read in primitive variables
    std::vector<Real> rho_b(nmb_tot * nx1 * nx2 * nx3);
    std::vector<Real> vx(nmb_tot * nx1 * nx2 * nx3);
    std::vector<Real> vy(nmb_tot * nx1 * nx2 * nx3);
    std::vector<Real> vz(nmb_tot * nx1 * nx2 * nx3);
    std::vector<Real> W(nmb_tot * nx1 * nx2 * nx3);
    std::vector<Real> press(nmb_tot * nx1 * nx2 * nx3);
    std::vector<Real> ye(nmb_tot * nx1 * nx2 * nx3);

    hid_t ds;
    ds = H5Dopen(g_vars, "rho_b", H5P_DEFAULT);
    H5Dread(ds, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, rho_b.data());
    H5Dclose(ds);
    ds = H5Dopen(g_vars, "vx", H5P_DEFAULT);
    H5Dread(ds, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, vx.data());
    H5Dclose(ds);
    ds = H5Dopen(g_vars, "vy", H5P_DEFAULT);
    H5Dread(ds, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, vy.data());
    H5Dclose(ds);
    ds = H5Dopen(g_vars, "vz", H5P_DEFAULT);
    H5Dread(ds, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, vz.data());
    H5Dclose(ds);
    ds = H5Dopen(g_vars, "w_lorentz", H5P_DEFAULT);
    H5Dread(ds, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, W.data());
    H5Dclose(ds);
    ds = H5Dopen(g_vars, "P", H5P_DEFAULT);
    H5Dread(ds, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, press.data());
    H5Dclose(ds);
    ds = H5Dopen(g_vars, "ye", H5P_DEFAULT);
    H5Dread(ds, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, ye.data());
    H5Dclose(ds);

    auto &w0 = pmbp->pmhd->w0;
    HostArray5D<Real>::HostMirror host_w0 = create_mirror_view(w0);
    for (int m = 0; m < nmb_tot; m++) {
      if (mb_idx[m] < gids || mb_idx[m] > gide) { continue; }
      int mb = mb_idx[m] - gids;
      for (int k = 0; k < nx3; k++) {
        for (int j = 0; j < nx2; j++) {
          for(int i = 0; i < nx1; i++) {
            int idx = m * (nx1 * nx2 * nx3) + i * (nx2 * nx3) + j * nx3 + k;
            host_w0(mb, IDN, k+ng, j+ng, i+ng) = rho_b[idx];
            host_w0(mb, IVX, k+ng, j+ng, i+ng) = W[idx] * vx[idx];
            host_w0(mb, IVY, k+ng, j+ng, i+ng) = W[idx] * vy[idx];
            host_w0(mb, IVZ, k+ng, j+ng, i+ng) = W[idx] * vz[idx];
            host_w0(mb, IPR, k+ng, j+ng, i+ng) = press[idx];
            host_w0(mb, IYF, k+ng, j+ng, i+ng) = ye[idx];
          }
        }
      }
    }
    Kokkos::deep_copy(w0, host_w0);
  }

  DvceArray4D<Real> Ax, Ay, Az;
  Kokkos::realloc(Ax, nmb, ncells3+1, ncells2+1, ncells1); // A_x at (i, j+1/2, k+1/2)
  Kokkos::realloc(Ay, nmb, ncells3+1, ncells2, ncells1+1); // A_y at (i+1/2, j, k+1/2)
  Kokkos::realloc(Az, nmb, ncells3, ncells2+1, ncells1+1); // A_z at (i+1/2, j+1/2, k)
  {
    HostArray4D<Real>::HostMirror host_Ax = create_mirror_view(Ax);
    HostArray4D<Real>::HostMirror host_Ay = create_mirror_view(Ay);
    HostArray4D<Real>::HostMirror host_Az = create_mirror_view(Az);
    // Read in magnetic potential
    std::vector<Real> Ax_vec(nmb_tot * nx1 * (nx2 + 1) * (nx3 + 1));
    std::vector<Real> Ay_vec(nmb_tot * (nx1 + 1) * nx2 * (nx3 + 1));
    std::vector<Real> Az_vec(nmb_tot * (nx1 + 1) * (nx2 + 1) * nx3);

    hid_t ds;
    ds = H5Dopen(g_vars, "Ax", H5P_DEFAULT);
    H5Dread(ds, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, Ax_vec.data());
    H5Dclose(ds);
    ds = H5Dopen(g_vars, "Ay", H5P_DEFAULT);
    H5Dread(ds, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, Ay_vec.data());
    H5Dclose(ds);
    ds = H5Dopen(g_vars, "Az", H5P_DEFAULT);
    H5Dread(ds, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, Az_vec.data());
    H5Dclose(ds);
    // Ax
    for (int m = 0; m < nmb_tot; m++) {
      if (mb_idx[m] < gids || mb_idx[m] > gide) continue;
      int mb = mb_idx[m] - gids;
      for (int k = 0; k < nx3 + 1; k++) {
        for (int j = 0; j < nx2 + 1; j++) {
          for(int i = 0; i < nx1; i++) {
            int idx = m * (nx1 * (nx2 + 1) * (nx3 + 1)) + i * ((nx2 + 1) * (nx3 + 1)) + j * (nx3 + 1) + k;
            host_Ax(mb, k+ng, j+ng, i+ng) = Ax_vec[idx];
          }
        }
      }
    }
    // Ay
    for (int m = 0; m < nmb_tot; m++) {
      if (mb_idx[m] < gids || mb_idx[m] > gide) continue;
      int mb = mb_idx[m] - gids;
      for (int k = 0; k < nx3 + 1; k++) {
        for (int j = 0; j < nx2; j++) {
          for(int i = 0; i < nx1 + 1; i++) {
            int idx = m * ((nx1 + 1) * nx2 * (nx3 + 1)) + i * (nx2 * (nx3 + 1)) + j * (nx3 + 1) + k;
            host_Ay(mb, k+ng, j+ng, i+ng) = Ay_vec[idx];
          }
        }
      }
    }
    // Az
    for (int m = 0; m < nmb_tot; m++) {
      if (mb_idx[m] < gids || mb_idx[m] > gide) continue;
      int mb = mb_idx[m] - gids;
      for (int k = 0; k < nx3; k++) {
        for (int j = 0; j < nx2 + 1; j++) {
          for(int i = 0; i < nx1 + 1; i++) {
            int idx = m * ((nx1 + 1) * (nx2 + 1) * nx3) + i * ((nx2 + 1) * nx3) + j * nx3 + k;
            host_Az(mb, k+ng, j+ng, i+ng) = Az_vec[idx];
          }
        }
      }
    }
    Kokkos::deep_copy(Ax, host_Ax);
    Kokkos::deep_copy(Ay, host_Ay);
    Kokkos::deep_copy(Az, host_Az);
  }

  {
    // Read in adm variables
    std::vector<Real> alp(nmb_tot * nx1 * nx2 * nx3);
    std::vector<Real> betax(nmb_tot * nx1 * nx2 * nx3);
    std::vector<Real> betay(nmb_tot * nx1 * nx2 * nx3);
    std::vector<Real> betaz(nmb_tot * nx1 * nx2 * nx3);
    std::vector<Real> gtxx(nmb_tot * nx1 * nx2 * nx3);
    std::vector<Real> gtxy(nmb_tot * nx1 * nx2 * nx3);
    std::vector<Real> gtxz(nmb_tot * nx1 * nx2 * nx3);
    std::vector<Real> gtyy(nmb_tot * nx1 * nx2 * nx3);
    std::vector<Real> gtyz(nmb_tot * nx1 * nx2 * nx3);
    std::vector<Real> gtzz(nmb_tot * nx1 * nx2 * nx3);
    std::vector<Real> W_z4c(nmb_tot * nx1 * nx2 * nx3);

    hid_t ds;
    ds = H5Dopen(g_vars, "alp", H5P_DEFAULT);
    H5Dread(ds, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, alp.data());
    H5Dclose(ds);
    ds = H5Dopen(g_vars, "betax", H5P_DEFAULT);
    H5Dread(ds, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, betax.data());
    H5Dclose(ds);
    ds = H5Dopen(g_vars, "betay", H5P_DEFAULT);
    H5Dread(ds, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, betay.data());
    H5Dclose(ds);
    ds = H5Dopen(g_vars, "betaz", H5P_DEFAULT);
    H5Dread(ds, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, betaz.data());
    H5Dclose(ds);
    ds = H5Dopen(g_vars, "gtxx", H5P_DEFAULT);
    H5Dread(ds, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, gtxx.data());
    H5Dclose(ds);
    ds = H5Dopen(g_vars, "gtxy", H5P_DEFAULT);
    H5Dread(ds, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, gtxy.data());
    H5Dclose(ds);
    ds = H5Dopen(g_vars, "gtxz", H5P_DEFAULT);
    H5Dread(ds, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, gtxz.data());
    H5Dclose(ds);
    ds = H5Dopen(g_vars, "gtyy", H5P_DEFAULT);
    H5Dread(ds, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, gtyy.data());
    H5Dclose(ds);
    ds = H5Dopen(g_vars, "gtyz", H5P_DEFAULT);
    H5Dread(ds, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, gtyz.data());
    H5Dclose(ds);
    ds = H5Dopen(g_vars, "gtzz", H5P_DEFAULT);
    H5Dread(ds, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, gtzz.data());
    H5Dclose(ds);
    ds = H5Dopen(g_vars, "W_z4c", H5P_DEFAULT);
    H5Dread(ds, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, W_z4c.data());
    H5Dclose(ds);

    auto &u_adm = pmbp->padm->u_adm;
    HostArray5D<Real>::HostMirror host_u_adm = create_mirror_view(u_adm);

    for (int m = 0; m < nmb_tot; m++) {
      if (mb_idx[m] < gids || mb_idx[m] > gide) continue;
      int mb = mb_idx[m] - gids;
      for (int k = 0; k < nx3; k++) {
        for (int j = 0; j < nx2; j++) {
          for(int i = 0; i < nx1; i++) {
            int idx = m * (nx1 * nx2 * nx3) + i * (nx2 * nx3) + j * nx3 + k;
            host_u_adm(mb, adm::ADM::I_ADM_ALPHA, k+ng, j+ng, i+ng) = alp[idx];
            host_u_adm(mb, adm::ADM::I_ADM_BETAX, k+ng, j+ng, i+ng) = betax[idx];
            host_u_adm(mb, adm::ADM::I_ADM_BETAY, k+ng, j+ng, i+ng) = betay[idx];
            host_u_adm(mb, adm::ADM::I_ADM_BETAZ, k+ng, j+ng, i+ng) = betaz[idx];
            host_u_adm(mb, adm::ADM::I_ADM_GXX, k+ng, j+ng, i+ng) = gtxx[idx] / (W_z4c[idx] * W_z4c[idx]);
            host_u_adm(mb, adm::ADM::I_ADM_GXY, k+ng, j+ng, i+ng) = gtxy[idx] / (W_z4c[idx] * W_z4c[idx]);
            host_u_adm(mb, adm::ADM::I_ADM_GXZ, k+ng, j+ng, i+ng) = gtxz[idx] / (W_z4c[idx] * W_z4c[idx]);
            host_u_adm(mb, adm::ADM::I_ADM_GYY, k+ng, j+ng, i+ng) = gtyy[idx] / (W_z4c[idx] * W_z4c[idx]);
            host_u_adm(mb, adm::ADM::I_ADM_GYZ, k+ng, j+ng, i+ng) = gtyz[idx] / (W_z4c[idx] * W_z4c[idx]);
            host_u_adm(mb, adm::ADM::I_ADM_GZZ, k+ng, j+ng, i+ng) = gtzz[idx] / (W_z4c[idx] * W_z4c[idx]);
            // temporary choice
            host_u_adm(mb, adm::ADM::I_ADM_KXX, k+ng, j+ng, i+ng) = 0.0;
            host_u_adm(mb, adm::ADM::I_ADM_KXY, k+ng, j+ng, i+ng) = 0.0;
            host_u_adm(mb, adm::ADM::I_ADM_KXZ, k+ng, j+ng, i+ng) = 0.0;
            host_u_adm(mb, adm::ADM::I_ADM_KYY, k+ng, j+ng, i+ng) = 0.0;
            host_u_adm(mb, adm::ADM::I_ADM_KYZ, k+ng, j+ng, i+ng) = 0.0;
            host_u_adm(mb, adm::ADM::I_ADM_KZZ, k+ng, j+ng, i+ng) = 0.0;
          }
        }
      }
    }
    Kokkos::deep_copy(u_adm, host_u_adm);
  }

  H5Gclose(g_vars);
  H5Gclose(g_it);
  H5Fclose(file);

  // calculate the face centered b field from vector potential
  auto &size = pmbp->pmb->mb_size;
  auto &b0 = pmbp->pmhd->b0;
  par_for("pgen_b0", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;
    b0.x1f(m, k, j, i) = ((Az(m, k, j+1, i) - Az(m, k, j, i)) / dx2 -
                          (Ay(m, k+1, j, i) - Ay(m, k, j, i)) / dx3);
    b0.x2f(m, k, j, i) = ((Ax(m, k+1, j, i) - Ax(m, k, j, i)) / dx3 -
                          (Az(m, k, j, i+1) - Az(m, k, j, i)) / dx1);
    b0.x3f(m, k, j, i) = ((Ay(m, k, j, i+1) - Ay(m, k, j, i)) / dx1 -
                          (Ax(m, k, j+1, i) - Ax(m, k, j, i)) / dx2);

    if (i == ie) {
      b0.x1f(m, k, j, i+1) = ((Az(m, k, j+1, i+1) - Az(m, k, j, i+1)) / dx2 -
                              (Ay(m, k+1, j, i+1) - Ay(m, k, j, i+1)) / dx3);
    }
    if (j == je) {
      b0.x2f(m, k, j+1, i) = ((Ax(m, k+1, j+1, i) - Ax(m, k, j+1, i)) / dx3 -
                              (Az(m, k, j+1, i+1) - Az(m, k, j+1, i)) / dx1);
    }
    if (k == ke) {
      b0.x3f(m, k+1, j, i) = ((Ay(m, k+1, j, i+1) - Ay(m, k+1, j, i)) / dx1 -
                              (Ax(m, k+1, j+1, i) - Ax(m, k+1, j, i)) / dx2);
    }
  });

  // calculate the cell centered bcc from face centered b
  auto &bcc0 = pmbp->pmhd->bcc0;
  par_for("pgen_bcc0", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
    bcc0(m, IBX, k, j, i) = 0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i+1));
    bcc0(m, IBY, k, j, i) = 0.5 * (b0.x2f(m, k, j, i) + b0.x2f(m, k, j+1, i));
    bcc0(m, IBZ, k, j, i) = 0.5 * (b0.x3f(m, k, j, i) + b0.x3f(m, k+1, j, i));
  });

  // Initialize conservative
  pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);

  return;
}