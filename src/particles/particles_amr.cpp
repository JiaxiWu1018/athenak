//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_amr.cpp
//! \brief Particle redistribution entry points used by dynamic AMR.

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "particles.hpp"

namespace particles {

TaskStatus Particles::RelabelForAMR(const DualArray1D<int> &oldtonew,
                                    const DualArray1D<int> &newrank,
                                    const DualArray1D<int> &refflag, int old_gids) {
  return pbval_part->SetPrtclGIDForAMR(oldtonew, newrank, refflag, old_gids);
}

void Particles::ShipAfterAMR() {
  pbval_part->CountSendsAndRecvs();
  pbval_part->InitPrtclRecv();
  pbval_part->PackAndSendPrtcls();
  while (pbval_part->RecvAndUnpackPrtcls() == TaskStatus::incomplete) {}
  pbval_part->ClearPrtclSend();
  pbval_part->ClearPrtclRecv();
  CheckMigration(nullptr, 0);
}

} // namespace particles
