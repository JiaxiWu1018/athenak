//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file read_particle.cpp
//! \brief Particles::read_prtcl_table — load particle initial conditions from an HDF5
//! file.
//!
//! Required 1-D datasets (all length N): x,y,z (position), ux,uy,uz (the COVARIANT
//! spatial 4-velocity u_i). Optional 1-D dataset mass (per-particle rest mass); if
//! absent, the scalar <particles> mass is used for every particle. Each particle is
//! assigned to the local MeshBlock whose bounding box contains it; particles outside all
//! local MeshBlocks belong to another rank and are dropped. The particle tag is its
//! global row index in the file, so tags are unique and identical regardless of the MPI
//! decomposition (so a serial run and an N-rank run can be compared per-tag — see the
//! stage README).
//!
//! Guarded by ATHENA_HAVE_HDF5: in a non-HDF5 build this is a clean fatal error at
//! runtime.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "config.hpp"
#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles.hpp"

#if ATHENA_HAVE_HDF5
#include <hdf5.h>
#endif

namespace particles {

//----------------------------------------------------------------------------------------
//! \fn void Particles::read_prtcl_table(const char *fname)

void Particles::read_prtcl_table(const char *fname) {
#if !(ATHENA_HAVE_HDF5)
  (void)fname;
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "<particles> init=file requires an HDF5-enabled build; reconfigure with "
            << "-D Athena_ENABLE_HDF5=ON." << std::endl;
  std::exit(EXIT_FAILURE);
#else
  hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "could not open particle initial-data file '" << fname << "'."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // open a 1-D dataset, returning its length; fatal on missing dataset or wrong rank
  auto open1d = [&](const char *name, hid_t &dset, hid_t &space) -> hsize_t {
    dset = H5Dopen2(file, name, H5P_DEFAULT);
    if (dset < 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "particle file '" << fname << "' is missing dataset '" << name << "'."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    space = H5Dget_space(dset);
    if (H5Sget_simple_extent_ndims(space) != 1) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "particle dataset '" << name << "' must be 1-D." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    hsize_t dim = 0;
    H5Sget_simple_extent_dims(space, &dim, NULL);
    return dim;
  };

  hid_t dx, dy, dz, dux, duy, duz, sx, sy, sz, sux, suy, suz;
  hsize_t N = open1d("x", dx, sx);
  if (open1d("y", dy, sy)   != N || open1d("z", dz, sz)   != N ||
      open1d("ux", dux, sux) != N || open1d("uy", duy, suy) != N ||
      open1d("uz", duz, suz) != N) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "particle datasets x,y,z,ux,uy,uz must all have the same length."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // optional per-particle mass
  bool have_mass = (H5Lexists(file, "mass", H5P_DEFAULT) > 0);
  hid_t dmass = -1, smass = -1;
  if (have_mass && open1d("mass", dmass, smass) != N) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "particle dataset 'mass' length must match x,y,z,ux,uy,uz." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  hid_t real_type = (sizeof(Real) == sizeof(double)) ? H5T_NATIVE_DOUBLE
                                                     : H5T_NATIVE_FLOAT;
  bool multi_d = pmy_pack->pmesh->multi_d;
  bool three_d = pmy_pack->pmesh->three_d;

  // kept (local) particle records, in the new contiguous slot order
  std::vector<Real> kx, ky, kz, kux, kuy, kuz, km;
  std::vector<int>  ktag, kgid;

  // read in chunks (robust for very large files)
  const hsize_t CHUNK = (1 << 20);
  std::vector<Real> bx(CHUNK), by(CHUNK), bz(CHUNK), bux(CHUNK), buy(CHUNK), buz(CHUNK),
                    bm(CHUNK);
  for (hsize_t off = 0; off < N; off += CHUNK) {
    hsize_t count = std::min(CHUNK, N - off);
    hid_t mem = H5Screate_simple(1, &count, NULL);
    auto read_chunk = [&](hid_t dset, hid_t space, std::vector<Real> &buf) {
      H5Sselect_hyperslab(space, H5S_SELECT_SET, &off, NULL, &count, NULL);
      H5Dread(dset, real_type, mem, space, H5P_DEFAULT, buf.data());
    };
    read_chunk(dx, sx, bx);   read_chunk(dy, sy, by);   read_chunk(dz, sz, bz);
    read_chunk(dux, sux, bux); read_chunk(duy, suy, buy); read_chunk(duz, suz, buz);
    if (have_mass) {read_chunk(dmass, smass, bm);}

    for (hsize_t i = 0; i < count; ++i) {
      int m = FindContainingMeshBlock(bx[i], by[i], bz[i]);
      if (m < 0) {continue;}   // belongs to another rank
      kx.push_back(bx[i]);   ky.push_back(by[i]);   kz.push_back(bz[i]);
      kux.push_back(bux[i]); kuy.push_back(buy[i]); kuz.push_back(buz[i]);
      km.push_back(have_mass ? bm[i] : this->mass);
      ktag.push_back(static_cast<int>(off + i));     // global file-row index
      kgid.push_back(pmy_pack->gids + m);
    }
    H5Sclose(mem);
  }

  // build host mirror arrays in the new layout and copy to device
  nprtcl_thispack = static_cast<int>(kx.size());
  Kokkos::realloc(prtcl_rdata, nrdata, nprtcl_thispack);
  Kokkos::realloc(prtcl_idata, nidata, nprtcl_thispack);
  auto rh = Kokkos::create_mirror_view(prtcl_rdata);
  auto ih = Kokkos::create_mirror_view(prtcl_idata);
  for (int p = 0; p < nprtcl_thispack; ++p) {
    rh(IPX,p) = kx[p];   rh(IPVX,p) = kux[p];
    rh(IPY,p) = ky[p];   rh(IPVY,p) = kuy[p];   // IPY/IPVY always allocated (>=2D)
    if (three_d) {rh(IPZ,p) = kz[p]; rh(IPVZ,p) = kuz[p];}
    rh(IPEN,p) = 0.0;                            // set by EnergyCalculation
    rh(IPM,p)  = km[p];
    ih(PGID,p) = kgid[p];
    ih(PTAG,p) = ktag[p];
  }
  Kokkos::deep_copy(prtcl_rdata, rh);
  Kokkos::deep_copy(prtcl_idata, ih);

  // close HDF5 handles
  if (have_mass) {H5Sclose(smass); H5Dclose(dmass);}
  H5Sclose(sx); H5Sclose(sy); H5Sclose(sz);
  H5Sclose(sux); H5Sclose(suy); H5Sclose(suz);
  H5Dclose(dx); H5Dclose(dy); H5Dclose(dz);
  H5Dclose(dux); H5Dclose(duy); H5Dclose(duz);
  H5Fclose(file);

  if (global_variable::my_rank == 0) {
    std::cout << "Particles: loaded '" << fname << "' (" << N << " particles in file)"
              << std::endl;
  }
#endif
}

} // namespace particles
