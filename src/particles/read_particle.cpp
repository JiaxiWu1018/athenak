//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file read_prtcl_data.cpp
//! \brief function to read and initialize particle data

#include <hdf5.h>
#include <cmath>
#include <limits>
#include <iostream>
#include <algorithm>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles.hpp"

namespace particles {
  void Particles::read_prtcl_table(const char* fname) {
    hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "particle H5Fopen failed" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    auto open1d = [&](const char* name)->std::pair<hid_t, hid_t>{
      hid_t dset = H5Dopen(file, name, H5P_DEFAULT);
      if (dset < 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "particle H5Dopen " << name << " failed" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      hid_t space = H5Dget_space(dset);
      if (space < 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "particle H5Dget_space " << name << " failed" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (H5Sget_simple_extent_ndims(space) != 1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "particle dataset " << name << " not 1D" << std::endl;
        std::exit(EXIT_FAILURE);
      }

      return {dset, space};
    };

    auto [dx, sx] = open1d("x");
    auto [dy, sy] = open1d("y");
    auto [dz, sz] = open1d("z");
    auto [dux, sux] = open1d("ux");
    auto [duy, suy] = open1d("uy");
    auto [duz, suz] = open1d("uz");

    // get length N and sanity-check all three datasets agree
    hsize_t dims_x[1]; H5Sget_simple_extent_dims(sx, dims_x, nullptr);
    hsize_t dims_y[1]; H5Sget_simple_extent_dims(sy, dims_y, nullptr);
    hsize_t dims_z[1]; H5Sget_simple_extent_dims(sz, dims_z, nullptr);
    hsize_t dims_ux[1]; H5Sget_simple_extent_dims(sux, dims_ux, nullptr);
    hsize_t dims_uy[1]; H5Sget_simple_extent_dims(suy, dims_uy, nullptr);
    hsize_t dims_uz[1]; H5Sget_simple_extent_dims(suz, dims_uz, nullptr);
    if (!(dims_x[0]==dims_y[0] && dims_x[0]==dims_z[0]) ||
        !(dims_ux[0]==dims_uy[0] && dims_ux[0]==dims_uz[0]) ||
        !(dims_x[0]==dims_ux[0])) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "particle x/y/z or ux/uy/uz dataset lengths differ" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const hsize_t N = dims_x[0];

    // Host particle data structure
    struct HostPrtclData {double x, y, z, ux, uy, uz; int gid;};
    std::vector<HostPrtclData> prtcl_h;
    const hsize_t CHUNK = 1<<20;
    prtcl_h.reserve(std::min<hsize_t>(N, 8*CHUNK)); // heuristic choice

    // Functions that process each chunk and store the data in this rank
    int &nmb = pmy_pack->nmb_thispack;
    auto &size = pmy_pack->pmb->mb_size;
    int &gids = pmy_pack->gids;
    struct MBBox { double x1min,x1max,x2min,x2max,x3min,x3max;};
    std::vector<MBBox> boxes(nmb);
    for (int m = 0; m < nmb; ++m) {
      boxes[m] = {size.h_view(m).x1min, size.h_view(m).x1max,
                  size.h_view(m).x2min, size.h_view(m).x2max,
                  size.h_view(m).x3min, size.h_view(m).x3max};
    }
    auto process_chunk = [&](const Real* xbuf, const Real* ybuf, const Real* zbuf,
                             const Real* uxbuf, const Real* uybuf, const Real* uzbuf, hsize_t n){
      for (hsize_t i = 0; i < n; ++i) {
        const Real xi = xbuf[i], yi = ybuf[i], zi = zbuf[i];
        const Real uxi = uxbuf[i], uyi = uybuf[i], uzi = uzbuf[i];
        for (int m = 0; m < nmb; ++m) {
          const MBBox& b = boxes[m];
          if (xi >= b.x1min && xi < b.x1max && yi >= b.x2min && yi < b.x2max &&
              zi >= b.x3min && zi < b.x3max) {
            int gid = gids + m;
            prtcl_h.push_back(HostPrtclData{xi, yi, zi, uxi, uyi, uzi, gid});
            break;
          }
        }
      }
    };

    // read data
    std::vector<Real> xb(CHUNK), yb(CHUNK), zb(CHUNK), uxb(CHUNK), uyb(CHUNK), uzb(CHUNK);
    hid_t mem_space = H5Screate_simple(1, (const hsize_t[1]){CHUNK}, nullptr);
    hid_t H5T_REAL = (sizeof(Real) == sizeof(float)) ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;
    for (hsize_t off = 0; off < N; off += CHUNK) {
      hsize_t this_chunk = std::min(CHUNK, N - off);
      H5Sset_extent_simple(mem_space, 1, &this_chunk, nullptr);

      hsize_t start[1] = {off}, count[1] = {this_chunk};
      H5Sselect_hyperslab(sx, H5S_SELECT_SET, start, nullptr, count, nullptr);
      H5Sselect_hyperslab(sy, H5S_SELECT_SET, start, nullptr, count, nullptr);
      H5Sselect_hyperslab(sz, H5S_SELECT_SET, start, nullptr, count, nullptr);
      H5Sselect_hyperslab(sux, H5S_SELECT_SET, start, nullptr, count, nullptr);
      H5Sselect_hyperslab(suy, H5S_SELECT_SET, start, nullptr, count, nullptr);
      H5Sselect_hyperslab(suz, H5S_SELECT_SET, start, nullptr, count, nullptr);

      hsize_t mstart[1] = {0}, mcount[1] = {this_chunk};
      H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, mstart, nullptr, mcount, nullptr);

      H5Dread(dx, H5T_REAL, mem_space, sx, H5P_DEFAULT, xb.data());
      H5Dread(dy, H5T_REAL, mem_space, sy, H5P_DEFAULT, yb.data());
      H5Dread(dz, H5T_REAL, mem_space, sz, H5P_DEFAULT, zb.data());
      H5Dread(dux, H5T_REAL, mem_space, sux, H5P_DEFAULT, uxb.data());
      H5Dread(duy, H5T_REAL, mem_space, suy, H5P_DEFAULT, uyb.data());
      H5Dread(duz, H5T_REAL, mem_space, suz, H5P_DEFAULT, uzb.data());

      process_chunk(xb.data(), yb.data(), zb.data(), uxb.data(), uyb.data(), uzb.data(), this_chunk);
    }

    H5Sclose(mem_space);

    // Close HDF5 handles
    H5Sclose(sx); H5Sclose(sy); H5Sclose(sz);
    H5Dclose(dx); H5Dclose(dy); H5Dclose(dz);
    H5Sclose(sux); H5Sclose(suy); H5Sclose(suz);
    H5Dclose(dux); H5Dclose(duy); H5Dclose(duz);
    H5Fclose(file);

    nprtcl_thispack = prtcl_h.size();
    std::cout << "Loaded " << nprtcl_thispack << " particles on rank " << global_variable::my_rank << std::endl;
    HostArray2D<Real> prtcl_rdata_h("prtcl_rdata_h", nrdata, nprtcl_thispack);
    HostArray2D<int> prtcl_idata_h("prtcl_idata_h", nidata, nprtcl_thispack);
    for (size_t p = 0; p < nprtcl_thispack; ++p) {
      prtcl_rdata_h(IPX, p) = prtcl_h[p].x;
      prtcl_rdata_h(IPY, p) = prtcl_h[p].y;
      prtcl_rdata_h(IPZ, p) = prtcl_h[p].z;
      prtcl_rdata_h(IPVX, p) = prtcl_h[p].ux;
      prtcl_rdata_h(IPVY, p) = prtcl_h[p].uy;
      prtcl_rdata_h(IPVZ, p) = prtcl_h[p].uz;
      prtcl_idata_h(PGID, p) = prtcl_h[p].gid;
      prtcl_idata_h(PTAG, p) = 0;
    }

    Kokkos::realloc(prtcl_rdata, nrdata, nprtcl_thispack);
    Kokkos::realloc(prtcl_idata, nidata, nprtcl_thispack);
    if (nprtcl_thispack > 0) {
      Kokkos::deep_copy(prtcl_rdata, prtcl_rdata_h);
      Kokkos::deep_copy(prtcl_idata, prtcl_idata_h);
    }

    return;
  }
} // namespace particles
