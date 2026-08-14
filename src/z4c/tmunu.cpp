//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tmunu.hpp
//! \brief implementation of Tmunu class
#include <algorithm>

#include <iostream>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "parameter_input.hpp"
#include "z4c/tmunu.hpp"
#include "mesh/mesh.hpp"
#include "globals.hpp"
#include "bvals/bvals.hpp"

char const * const Tmunu::Tmunu_names[Tmunu::N_Tmunu] = {
  "tmunu_Sxx", "tmunu_Sxy", "tmunu_Sxz", "tmunu_Syy", "tmunu_Syz", "tmunu_Szz",
  "tmunu_E", "tmunu_Sx", "tmunu_Sy", "tmunu_Sz",
};

//----------------------------------------------------------------------------------------
// constructor: initializes data structures and parameters
Tmunu::Tmunu(MeshBlockPack *ppack, ParameterInput *pin):
  pmy_pack(ppack),
  u_tmunu("u_tmunu",1,1,1,1,1),
  coarse_u_tmunu("coarse_u_tmunu",1,1,1,1,1),
  u_filt_scratch("u_filt_scratch",1,1,1,1),
  filt_sums("filt_sums",1),
  pbval_tmunu(nullptr) {
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*(indcs.ng)) : 1;

  Kokkos::realloc(u_tmunu, nmb, N_Tmunu, ncells3, ncells2, ncells1);
  tmunu.S_dd.InitWithShallowSlice(u_tmunu, I_Tmunu_Sxx, I_Tmunu_Szz);
  tmunu.E.InitWithShallowSlice(u_tmunu, I_Tmunu_E);
  tmunu.S_d.InitWithShallowSlice(u_tmunu, I_Tmunu_Sx, I_Tmunu_Sz);

  // ---- Entity-style digital filter (tmunu_filter.cpp). Default 0 = OFF: nothing below
  // allocates or communicates, preserving the pre-filter behavior bit-for-bit.
  nfilter_passes = pin->GetOrAddInteger("particles", "tmunu_filter_passes", 0);
  filter_selftest = pin->GetOrAddBoolean("particles", "tmunu_filter_selftest", false);
  filter_diag_cadence = std::max(1, pin->GetOrAddInteger("time", "ndiag", 1));
  if (filter_selftest && nfilter_passes <= 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "tmunu_filter_selftest=true requires "
              << "tmunu_filter_passes >= 1" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (nfilter_passes > 0) {
    if (!(pmy_pack->pmesh->three_d)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "tmunu_filter_passes > 0 requires a 3D mesh "
                << "(Eq. (11) stencil is 3D)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    Kokkos::realloc(u_filt_scratch, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(filt_sums, 20);
    if (pmy_pack->pmesh->multilevel) {
      int nccells1 = indcs.cnx1 + 2*(indcs.ng);
      int nccells2 = (indcs.cnx2 > 1) ? (indcs.cnx2 + 2*(indcs.ng)) : 1;
      int nccells3 = (indcs.cnx3 > 1) ? (indcs.cnx3 + 2*(indcs.ng)) : 1;
      Kokkos::realloc(coarse_u_tmunu, nmb, N_Tmunu, nccells3, nccells2, nccells1);
    }
    // dedicated CC boundary machinery for u_tmunu (non-z4c mode: standard buffers,
    // 2nd-order prolongation at SMR seams). Own duplicated MPI communicator, so it
    // cannot collide with z4c/hydro traffic.
    pbval_tmunu = new MeshBoundaryValuesCC(ppack, pin, false);
    pbval_tmunu->InitializeBuffers(N_Tmunu);
    // one-time memory report: everything the filter allocates beyond the base build
    if (global_variable::my_rank == 0) {
      std::size_t bytes = u_filt_scratch.size()*sizeof(Real)
                        + coarse_u_tmunu.size()*sizeof(Real)
                        + filt_sums.size()*sizeof(Real);
      std::size_t buf_bytes = 0;
      for (int n=0; n<56; ++n) {
        buf_bytes += (pbval_tmunu->sendbuf[n].vars.size()
                    + pbval_tmunu->recvbuf[n].vars.size()
                    + pbval_tmunu->sendbuf[n].flux.size()
                    + pbval_tmunu->recvbuf[n].flux.size())*sizeof(Real);
      }
      std::cout << "[tmunu-filter] enabled: passes=" << nfilter_passes
                << " selftest=" << (filter_selftest ? 1 : 0)
                << " extra device memory/rank: scratch+coarse="
                << bytes/1048576.0 << " MiB, bvals buffers="
                << buf_bytes/1048576.0 << " MiB, total="
                << (bytes + buf_bytes)/1048576.0 << " MiB" << std::endl;
    }
  }
}

Tmunu::~Tmunu() {
  if (pbval_tmunu != nullptr) {delete pbval_tmunu;}
}
