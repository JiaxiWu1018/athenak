//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_newdt.cpp
//! \brief function to compute Particle timestep across all MeshBlock(s) in a MeshBlockPack

#include <cmath>
#include <limits>
#include <iostream>
#include <algorithm>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "particles.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::NewTimeStep
//! \brief Wrapper task list function to calculate new timestep

TaskStatus Particles::NewTimeStep(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;

  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();

  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  const int nm = pmy_pack->nmb_thispack;
  auto &mbsize = pmy_pack->pmb->mb_size;

  Kokkos::parallel_reduce("MHDNudt2",Kokkos::RangePolicy<>(DevExeSpace(), 0, nm),
    KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3) {
      // compute m,k,j,i indices of thread and call function
      int m = idx;

      min_dt1 = fmin(mbsize.d_view(m).dx1, min_dt1);
      min_dt2 = fmin(mbsize.d_view(m).dx2, min_dt2);
      min_dt3 = fmin(mbsize.d_view(m).dx3, min_dt3);
  }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));

  dtnew = dt1;
  if (pmy_pack->pmesh->multi_d) { dtnew = std::min(dtnew, dt2); }
  if (pmy_pack->pmesh->three_d) { dtnew = std::min(dtnew, dt3); }

  return TaskStatus::complete;
}
} //end namespace particles