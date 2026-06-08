//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_newdt.cpp
//! \brief compute the particle timestep across all MeshBlock(s) in a MeshBlockPack as a
//! CFL condition: dt = cfl * min_p( cell size / particle coordinate speed ). The coordinate
//! speed is estimated from the stored covariant 4-velocity u_i via a flat Lorentz factor
//! W = sqrt(1 + delta^{ij} u_i u_j), v^d ~ |u_d|/W (a conservative estimate adequate for a
//! timestep; it avoids a second metric interpolation). Mesh::NewTimeStep mins in this
//! dtnew WITHOUT a cfl_no factor, so the CFL number is baked in here.

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
  int gids = pmy_pack->gids;
  int nmb = pmy_pack->nmb_thispack;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  Real cfl = pmy_pack->pmesh->cfl_no;

  const Real big = std::numeric_limits<Real>::max();
  Real dt_part = big;

  // CFL over particles: crossing time = cell size / coordinate speed. A particle at rest in
  // a given direction contributes a large-but-finite (geometry-tied) time so it cannot pin
  // dt to zero.
  if (nprtcl_thispack > 0) {
    Kokkos::parallel_reduce("part_newdt",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nprtcl_thispack),
      KOKKOS_LAMBDA(const int p, Real &min_dt) {
        int m = pi(PGID,p) - gids;
        Real ux = pr(IPVX,p);
        Real uy = multi_d ? pr(IPVY,p) : 0.0;
        Real uz = three_d ? pr(IPVZ,p) : 0.0;
        Real W = std::sqrt(1.0 + ux*ux + uy*uy + uz*uz);

        Real dx1 = mbsize.d_view(m).dx1;
        Real t1 = (std::fabs(ux) > 0.0) ? dx1*W/std::fabs(ux) : 1.0e3*dx1;
        min_dt = fmin(min_dt, t1);
        if (multi_d) {
          Real dx2 = mbsize.d_view(m).dx2;
          Real t2 = (std::fabs(uy) > 0.0) ? dx2*W/std::fabs(uy) : 1.0e3*dx2;
          min_dt = fmin(min_dt, t2);
        }
        if (three_d) {
          Real dx3 = mbsize.d_view(m).dx3;
          Real t3 = (std::fabs(uz) > 0.0) ? dx3*W/std::fabs(uz) : 1.0e3*dx3;
          min_dt = fmin(min_dt, t3);
        }
      }, Kokkos::Min<Real>(dt_part));
  }

  // Fallback so dt stays finite when there are no particles on this pack (or the reduction
  // yielded a non-finite value): tie dt to the local grid spacing, as the prototype did.
  if (nprtcl_thispack == 0 || !std::isfinite(dt_part) || dt_part >= big) {
    Real dgrid = big;
    Kokkos::parallel_reduce("part_newdt_grid",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb),
      KOKKOS_LAMBDA(const int m, Real &min_dt) {
        Real d = mbsize.d_view(m).dx1;
        if (multi_d) {d = fmin(d, mbsize.d_view(m).dx2);}
        if (three_d) {d = fmin(d, mbsize.d_view(m).dx3);}
        min_dt = fmin(min_dt, d);
      }, Kokkos::Min<Real>(dgrid));
    dt_part = dgrid;
  }

  dtnew = cfl*dt_part;
  // TODO(Stage 2): for charged particles also impose the cyclotron limit
  //   dt <= C * 2*pi/Omega_c, Omega_c = |q| B / (m gamma).
  return TaskStatus::complete;
}

} // namespace particles
