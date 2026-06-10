//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particle_pushers.cpp
//  \brief

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "particles.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn  void Particles::ParticlesPush
//  \brief

TaskStatus Particles::Push(Driver *pdriver, int stage) {
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &pr = prtcl_rdata;
  auto dt_ = (pmy_pack->pmesh->dt);

  switch (pusher) {
    case ParticlesPusher::drift:

      // free streaming over the full timestep (particle tasks run once per cycle in the
      // after_timeintegrator list, so there is no per-stage 0.5*dt splitting)
      par_for("part_update",DevExeSpace(),0,(nprtcl_thispack-1),
      KOKKOS_LAMBDA(const int p) {
        pr(IPX,p) += dt_*pr(IPVX,p);

        if (multi_d) {
          pr(IPY,p) += dt_*pr(IPVY,p);
        }

        if (three_d) {
          pr(IPZ,p) += dt_*pr(IPVZ,p);
        }
      });

    break;
  case ParticlesPusher::boris:
    BorisPush();
    break;
  case ParticlesPusher::gr_boris:
    GR_BorisPush();
    break;
  default:
    break;
  }

  return TaskStatus::complete;
}
} // namespace particles
