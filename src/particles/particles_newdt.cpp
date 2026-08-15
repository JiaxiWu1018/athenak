//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_newdt.cpp
//! \brief compute the particle timestep across all MeshBlock(s) in a MeshBlockPack as the cell
//! light-crossing CFL condition dt = cfl * min(dx,dy,dz). This is the robust choice for
//! particles that ACCELERATE under gravity or the Lorentz force: a velocity-based crossing
//! time dx/v would (a) inflate dt without bound as v->0 -- so an initially-at-rest particle
//! (radial geodesic infall) would take a single enormous step -- and (b) under-resolve the
//! acceleration even when v>0, since dx/v assumes constant velocity. Because the coordinate
//! speed cannot exceed c=1, the light-crossing time dx is always the binding constraint, so no
//! velocity factor is needed. Mesh::NewTimeStep mins in this dtnew WITHOUT a cfl_no factor, so
//! the CFL number is baked in here. (A cyclotron limit for strongly magnetized charges, dt <=
//! C*2*pi*gamma/(|q/m| B), can be added later; the Stage-2 tests resolve the gyro-period.)

#include <cmath>
#include <limits>
#include <algorithm>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "particles.hpp"

namespace particles {

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::NewTimeStep

TaskStatus Particles::NewTimeStep(Driver *pdrive, int stage) {
  bool multi_d = pmy_pack->pmesh->multi_d;
  bool three_d = pmy_pack->pmesh->three_d;
  int nmb = pmy_pack->nmb_thispack;
  auto &mbsize = pmy_pack->pmb->mb_size;
  Real cfl = pmy_pack->pmesh->cfl_no;

  const Real big = std::numeric_limits<Real>::max();

  // light-crossing CFL: dt = cfl * min cell size over all MeshBlocks (independent of the
  // particle state, so it is robust to accelerating and initially-at-rest particles)
  Real dgrid = big;
  Kokkos::parallel_reduce("part_newdt",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb),
    KOKKOS_LAMBDA(const int m, Real &min_dt) {
      Real d = mbsize.d_view(m).dx1;
      if (multi_d) {d = fmin(d, mbsize.d_view(m).dx2);}
      if (three_d) {d = fmin(d, mbsize.d_view(m).dx3);}
      min_dt = fmin(min_dt, d);
    }, Kokkos::Min<Real>(dgrid));

  dtnew = cfl*dgrid;
  return TaskStatus::complete;
}

} // namespace particles
